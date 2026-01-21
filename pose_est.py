import argparse
import os
import threading
from collections import Counter, deque
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import get_source_type
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

from metrics_utils import MetricsCollector, MetricWindows


SMOOTH_WINDOW_LEN = 7
MIN_KP_CONF = 0.35

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


# Metriken

def start_metrics_tick_thread(metrics: MetricsCollector):
    stop_evt = threading.Event()

    def loop():
        while not stop_evt.wait(1.0):
            try:
                metrics.on_tick_1hz()
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()
    return stop_evt


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-source", default="libcamera0")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--run-seconds", type=int, default=600)
    p.add_argument("--metrics-step-sec", type=int, default=600)
    p.add_argument("--metrics-max-sec", type=int, default=3600)
    p.add_argument("--print-every-frame", action="store_true", default=True)
    return p.parse_args()



# State

class AppState:
    def __init__(self, metrics: MetricsCollector, print_every_frame: bool):
        self.metrics = metrics
        self.print_every_frame = print_every_frame
        self.pose_hist = deque(maxlen=SMOOTH_WINDOW_LEN)
        self.current_pose = "UNKNOWN"


# Posen Logik

def kp_ok(kps, i: int) -> bool:
    return kps is not None and 0 <= i < len(kps) and kps[i][2] >= MIN_KP_CONF


def majority(hist) -> str:
    if not hist:
        return "UNKNOWN"
    return Counter(hist).most_common(1)[0][0]


def classify_pose(kps_xyc, w: int, h: int) -> str:
    if kps_xyc is None:
        return "UNKNOWN"

    need = [NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP]
    if any(not kp_ok(kps_xyc, i) for i in need):
        return "UNKNOWN"

    nose_y = kps_xyc[NOSE][1]
    lsx, lsy = kps_xyc[L_SHOULDER][0], kps_xyc[L_SHOULDER][1]
    rsx, rsy = kps_xyc[R_SHOULDER][0], kps_xyc[R_SHOULDER][1]
    lwx, lwy = kps_xyc[L_WRIST][0], kps_xyc[L_WRIST][1]
    rwx, rwy = kps_xyc[R_WRIST][0], kps_xyc[R_WRIST][1]
    lhy, rhy = kps_xyc[L_HIP][1], kps_xyc[R_HIP][1]

    has_elb = kp_ok(kps_xyc, L_ELBOW) and kp_ok(kps_xyc, R_ELBOW)
    ley = kps_xyc[L_ELBOW][1] if has_elb else None
    rey = kps_xyc[R_ELBOW][1] if has_elb else None

    mean_sh = 0.5 * (lsy + rsy)
    mean_hip = 0.5 * (lhy + rhy)
    torso_w = abs(rsx - lsx) or 0.1 * w
    torso_h = max(mean_hip - mean_sh, 0.2 * h)

    up_margin = 0.10 * h
    if (lwy < nose_y - up_margin and rwy < nose_y - up_margin) or \
       (lwy < mean_sh - up_margin and rwy < mean_sh - up_margin):
        return "HANDS_UP"

    tol_y_level = 0.18 * h
    out_min_x = max(0.6 * torso_w, 0.15 * w)

    left_lvl = abs(lwy - lsy) < tol_y_level and (ley is None or abs(ley - lsy) < tol_y_level * 1.1)
    right_lvl = abs(rwy - rsy) < tol_y_level and (rey is None or abs(rey - rsy) < tol_y_level * 1.1)
    left_out = abs(lwx - lsx) > out_min_x
    right_out = abs(rwx - rsx) > out_min_x

    if left_lvl and right_lvl and (left_out or right_out):
        return "T_POSE"
    if (left_lvl and left_out and right_lvl) or (right_lvl and right_out and left_lvl):
        return "T_POSE"

    below_sh_thr = mean_sh + 0.35 * torso_h
    hands_below = (lwy > below_sh_thr) and (rwy > below_sh_thr)

    near_body_max_x = max(0.5 * torso_w, 0.10 * w)
    hands_near = (abs(lwx - lsx) < near_body_max_x) and (abs(rwx - rsx) < near_body_max_x)

    elbows_ok = True
    if has_elb:
        elbows_ok = (ley > mean_sh + 0.15 * torso_h) and (rey > mean_sh + 0.15 * torso_h)

    if hands_below and hands_near and elbows_ok:
        return "ARMS_DOWN"

    return "UNKNOWN"


def extract_best_person_keypoints(detections, w: int, h: int):
    best_kps, best_conf = None, 0.0

    for det in detections:
        if det.get_label() != "person":
            continue

        bbox = det.get_bbox()
        conf = float(det.get_confidence())

        lmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not lmarks:
            continue

        pts = lmarks[0].get_points()
        kps_xyc = []
        for p in pts:
            x = (p.x() * bbox.width() + bbox.xmin()) * w
            y = (p.y() * bbox.height() + bbox.ymin()) * h
            kps_xyc.append((x, y, conf))

        if conf > best_conf:
            best_conf = conf
            best_kps = kps_xyc

    return best_kps


# Callback

def app_callback(pad, info, state: AppState):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    try:
        state.metrics.on_frame()
    except Exception:
        pass

    _fmt, w, h = get_caps_from_pad(pad)
    if not w or not h:
        return Gst.PadProbeReturn.OK

    try:
        roi = hailo.get_roi_from_buffer(buf)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        detections = []

    best_kps = extract_best_person_keypoints(detections, int(w), int(h))
    pose = classify_pose(best_kps, int(w), int(h))

    state.pose_hist.append(pose)
    state.current_pose = majority(state.pose_hist)

    if state.print_every_frame:
        print(state.current_pose)

    return Gst.PadProbeReturn.OK


# Pipeline lifecycle

def build_pipeline(args, state: AppState):
    app = GStreamerPoseEstimationApp(app_callback, state)
    app.video_source = args.video_source
    app.source_type = get_source_type(app.video_source)
    app.frame_rate = args.fps
    app.sync = "false"
    app.options_menu.show_fps = False
    app.options_menu.use_frame = False
    app.create_pipeline()
    return app, app.pipeline


def configure_hailo_elements(pipeline):
    if pipeline is None:
        return
    _hnet = pipeline.get_by_name("hnet")
    _overlay = pipeline.get_by_name("overlay")


def attach_callback(app, pipeline, state: AppState, run_seconds: int):
    identity = pipeline.get_by_name("identity_callback")
    if identity is not None:
        src_pad = identity.get_static_pad("src")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, app.app_callback, app.user_data)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, msg):
        if msg.type == Gst.MessageType.QOS:
            try:
                state.metrics.on_qos()
            except Exception:
                pass
        elif msg.type == Gst.MessageType.ERROR:
            try:
                state.metrics.on_error()
            except Exception:
                pass
        app.bus_call(_bus, msg, app.loop)

    bus.connect("message", on_bus)
    GLib.timeout_add_seconds(int(run_seconds), lambda: (app.shutdown() or False))


# main

def main():
    args = parse_args()

    project_root = Path.home() / "Projekt"
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")

    Gst.init(None)

    metrics = MetricsCollector(MetricWindows(step_sec=args.metrics_step_sec, max_total_sec=args.metrics_max_sec))
    stop_evt = start_metrics_tick_thread(metrics)

    state = AppState(metrics, args.print_every_frame)
    app, pipeline = build_pipeline(args, state)
    configure_hailo_elements(pipeline)
    attach_callback(app, pipeline, state, args.run_seconds)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        app.loop.run()
    finally:
        stop_evt.set()
        pipeline.set_state(Gst.State.NULL)
        print("\nMETRICS SUMMARY")
        print(metrics.summary_text())
        print(f"\nLast pose (smoothed): {state.current_pose}")


if __name__ == "__main__":
    main()
