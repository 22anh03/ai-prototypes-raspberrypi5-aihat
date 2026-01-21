import argparse
import math
import os
import signal
import sys
import threading
import time
from collections import deque

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cairo
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
from hailo_apps.hailo_app_python.core.gstreamer import gstreamer_helper_pipelines as ghp
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

from metrics_utils import MetricsCollector, MetricWindows


VEHICLE_LABELS_BASE = {"car", "bus", "truck", "motorcycle"}

MIN_BBOX_AREA_FRAC = float(os.environ.get("MIN_BBOX_AREA_FRAC", "0.0002"))
MIN_CROSS_INTERVAL = float(os.environ.get("MIN_CROSS_INTERVAL", "0.6"))
TRACK_TTL = float(os.environ.get("PASS_TRACK_TTL", "3.0"))
ASSOC_MAX_DIST = float(os.environ.get("ASSOC_MAX_DIST", "160"))
FALLBACK_LOSS_SEC = float(os.environ.get("FALLBACK_LOSS_SEC", "1.5"))


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

# 

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# CLI

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--orient", choices=["H", "V"], default=os.environ.get("PASS_LINE_ORIENT", "H"))
    p.add_argument("--pos", type=float, default=float(os.environ.get("PASS_LINE_POS", "0.60")))
    p.add_argument("--no-line", action="store_true")
    p.add_argument("--include-person", action="store_true")
    p.add_argument("--video-sink", default=os.environ.get("VIDEO_SINK", ""))
    p.add_argument("--env", default=os.environ.get("HAILO_ENV_FILE", "/home/Projekt/.env"))
    p.add_argument("--metrics-step-sec", type=int, default=int(os.environ.get("METRICS_STEP_SEC", "600")))
    p.add_argument("--metrics-max-sec", type=int, default=int(os.environ.get("METRICS_MAX_SEC", "3600")))
    p.add_argument("-h", "--help", action="store_true")
    args, passthrough = p.parse_known_args()

    if args.help:
        p.print_help()
        sys.exit(0)

    if not any(a == "--input" or a.startswith("--input=") for a in passthrough):
        passthrough += ["--input", "rpi"]

    return args, passthrough


def apply_env(args, passthrough):
    os.environ["HAILO_ENV_FILE"] = args.env
    if args.video_sink:
        os.environ["VIDEO_SINK"] = args.video_sink

    os.environ["PASS_LINE_ORIENT"] = args.orient.upper()
    os.environ["PASS_LINE_POS"] = str(args.pos)
    os.environ["PASS_DRAW_LINE"] = "0" if args.no_line else "1"

    sys.argv = [sys.argv[0]] + passthrough


def _get_input_value(argv):
    for i, a in enumerate(argv):
        if a == "--input" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--input="):
            return a.split("=", 1)[1]
    return None


# Tracker

class DrawCache:
    def __init__(self, maxlen=60):
        self.items = deque(maxlen=maxlen)

    def put(self, pts_ns: int, boxes, cars_now: int, passed_lr: int, passed_rl: int):
        self.items.append((pts_ns, boxes, cars_now, passed_lr, passed_rl))

    def best(self, ts_ns: int):
        if not self.items:
            return None
        best_item, best_d = None, 1e30
        for it in self.items:
            d = abs(int(it[0]) - int(ts_ns))
            if d < best_d:
                best_d = d
                best_item = it
        return best_item


class Tracker:
    def __init__(self):
        self.next_tid = 1
        self.tracks = {}

    def assign_tid(self, label: str, cx: int, cy: int, now: float) -> int:
        label_tracks = self.tracks.setdefault(label, {})
        alive = {tid: st for tid, st in label_tracks.items() if now - st["t"] <= FALLBACK_LOSS_SEC}

        best_tid, best_d = None, 1e9
        for tid, st in alive.items():
            d = math.hypot(cx - st["cx"], cy - st["cy"])
            if d < best_d:
                best_d, best_tid = d, tid

        if best_tid is not None and best_d <= ASSOC_MAX_DIST:
            label_tracks[best_tid] = {"cx": cx, "cy": cy, "t": now}
            return best_tid

        tid = self.next_tid
        self.next_tid += 1
        label_tracks[tid] = {"cx": cx, "cy": cy, "t": now}
        return tid

    def gc(self, now: float):
        for label in list(self.tracks.keys()):
            for tid in list(self.tracks[label].keys()):
                if now - self.tracks[label][tid]["t"] > FALLBACK_LOSS_SEC:
                    self.tracks[label].pop(tid, None)



# State

class AppState(app_callback_class):
    def __init__(self, orient: str, pos: float, vehicle_labels):
        super().__init__()
        self.cur_size = None
        self.line_orient = orient
        self.line_pos = pos
        self.vehicle_labels = set(vehicle_labels)

        self.passed_lr = 0
        self.passed_rl = 0

        self.last_side = {}
        self.last_seen = {}
        self.last_cross_ts = {}

        self.tracker = Tracker()
        self.draw_cache = DrawCache(maxlen=60)

        self.count_text = None
        self.metrics = None


# Pipeline 

def overlay_pipeline_nop(name="hailo_display_overlay"):
    return f'{ghp.QUEUE(name=f"{name}_q")} ! identity name={name}_noop'


def display_pipeline(is_file_input: bool, name="hailo_display", sync="false", show_fps="false"):
    sink = os.environ.get("VIDEO_SINK") or "autovideosink"
    if is_file_input:
        sync = "true"

    return (
        f'{ghp.OVERLAY_PIPELINE(name=f"{name}_overlay")} ! '
        f'{ghp.QUEUE(name=f"{name}_vc_q")} ! '
        'videoconvert qos=false ! video/x-raw,pixel-aspect-ratio=1/1 ! '
        'cairooverlay name=passline_draw ! '
        'textoverlay name=count_text text="Cars now: 0 | Passed LR: 0  RL: 0" '
        'valignment=top halignment=left shaded-background=true font-desc="Sans 20" ! '
        f'{ghp.QUEUE(name=f"{name}_q")} ! '
        f'fpsdisplaysink name={name} video-sink={sink} sync={sync} text-overlay={show_fps}'
    )


def build_pipeline(state: AppState):
    Gst.init(None)

    inp = _get_input_value(sys.argv) or ""
    is_file_input = bool(inp and inp not in ("rpi", "libcamera", "libcamera0") and not inp.startswith(("v4l2", "rtsp://")))

    ghp.OVERLAY_PIPELINE = overlay_pipeline_nop
    ghp.DISPLAY_PIPELINE = lambda **kw: display_pipeline(is_file_input=is_file_input, **kw)  # type: ignore

    app = GStreamerDetectionApp(app_callback, state)
    pipeline = getattr(app, "pipeline", None)
    return app, pipeline


def configure_hailo_elements(pipeline):
    if pipeline is None:
        return
    _hnet = pipeline.get_by_name("hnet")
    _overlay = pipeline.get_by_name("overlay")


def attach_callback(pipeline, state: AppState):
    if pipeline is None:
        return
    state.count_text = pipeline.get_by_name("count_text")
    overlay = pipeline.get_by_name("passline_draw")
    if overlay is not None:
        overlay.connect("caps-changed", on_caps_changed)
        overlay.connect("draw", on_draw)



# Detection Helfer

def _bb_get(bb, name):
    v = getattr(bb, name, None)
    return float(v()) if callable(v) else float(v)


def bbox_to_frame_xyxy(bb, w, h):
    try:
        x = _bb_get(bb, "xmin")
        y = _bb_get(bb, "ymin")
        bw = _bb_get(bb, "width")
        bh = _bb_get(bb, "height")
    except Exception:
        return None

    if bw <= 0 or bh <= 0:
        return None

    norm_like = (0.0 <= x <= 1.2 and 0.0 <= y <= 1.2 and 0.0 < bw <= 1.2 and 0.0 < bh <= 1.2)
    if norm_like:
        x1 = x * w
        y1 = y * h
        x2 = (x + bw) * w
        y2 = (y + bh) * h
    else:
        x1, y1, x2, y2 = x, y, x + bw, y + bh

    return x1, y1, x2, y2


def side_rel_to_tripline(state: AppState, cx: int, cy: int, w: int, h: int) -> int:
    if state.line_orient == "H":
        y_line = int(h * state.line_pos)
        return -1 if cy < y_line else +1
    x_line = int(w * state.line_pos)
    return -1 if cx < x_line else +1


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
    if w and h:
        state.cur_size = (int(w), int(h))
    if not state.cur_size:
        return Gst.PadProbeReturn.OK
    w, h = state.cur_size

    try:
        roi = hailo.get_roi_from_buffer(buf)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        dets = []

    now = time.time()

    for tid in list(state.last_seen.keys()):
        if now - state.last_seen[tid] > TRACK_TTL:
            state.last_seen.pop(tid, None)
            state.last_side.pop(tid, None)
            state.last_cross_ts.pop(tid, None)

    state.tracker.gc(now)

    cars_now = 0
    draw_boxes = []

    for det in dets:
        try:
            label = det.get_label()
            if label not in state.vehicle_labels:
                continue
        except Exception:
            continue

        bb = det.get_bbox()
        xyxy = bbox_to_frame_xyxy(bb, w, h)
        if xyxy is None:
            continue

        cl = clamp_box(xyxy[0], xyxy[1], xyxy[2], xyxy[3], w, h)
        if not cl:
            continue

        x1, y1, x2, y2 = cl
        bw = x2 - x1
        bh = y2 - y1
        if bw * bh < MIN_BBOX_AREA_FRAC * (w * h):
            continue

        cx = int((x1 + x2) * 0.5)
        cy = int(y2)

        use_fallback = False
        try:
            ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            tid = int(ids[0].get_id()) if len(ids) == 1 else None
            if tid is None:
                use_fallback = True
        except Exception:
            use_fallback = True

        if use_fallback:
            tid = state.tracker.assign_tid(label, cx, cy, now)

        s_prev = state.last_side.get(tid)
        s_now = side_rel_to_tripline(state, cx, cy, w, h)

        if s_prev is not None and s_prev != s_now:
            last_t = state.last_cross_ts.get(tid, 0.0)
            if (now - last_t) >= MIN_CROSS_INTERVAL:
                if s_prev < s_now:
                    state.passed_lr += 1
                else:
                    state.passed_rl += 1
                state.last_cross_ts[tid] = now

        state.last_side[tid] = s_now
        state.last_seen[tid] = now

        draw_boxes.append((x1, y1, x2, y2, label))
        cars_now += 1

    if state.count_text is not None:
        try:
            state.count_text.set_property(
                "text",
                f"Cars now: {cars_now} | Passed LR: {state.passed_lr}  RL: {state.passed_rl}",
            )
        except Exception:
            pass

    pts_ns = int(buf.pts)
    if pts_ns == int(Gst.CLOCK_TIME_NONE) or pts_ns < 0:
        pts_ns = int(time.time() * 1e9)

    state.draw_cache.put(pts_ns, draw_boxes, cars_now, state.passed_lr, state.passed_rl)
    return Gst.PadProbeReturn.OK


# cairooverlay

LINE_W = 0
LINE_H = 0
STATE_REF = None


def on_caps_changed(_overlay, caps):
    global LINE_W, LINE_H
    try:
        s = caps.get_structure(0)
        ok_w, w = s.get_int("width")
        ok_h, h = s.get_int("height")
        if ok_w and ok_h:
            LINE_W, LINE_H = w, h
    except Exception:
        pass


def on_draw(_overlay, ctx, timestamp, _duration):
    if STATE_REF is None:
        return

    try:
        if os.environ.get("PASS_DRAW_LINE", "1") != "0":
            ctx.set_source_rgba(1.0, 1.0, 0.0, 0.9)
            ctx.set_line_width(3.0)
            if STATE_REF.line_orient == "H":
                y = int(LINE_H * STATE_REF.line_pos)
                ctx.move_to(0, y); ctx.line_to(LINE_W, y)
            else:
                x = int(LINE_W * STATE_REF.line_pos)
                ctx.move_to(x, 0); ctx.line_to(x, LINE_H)
            ctx.stroke()
    except Exception:
        pass

    best = STATE_REF.draw_cache.best(int(timestamp))
    if best is None:
        return
    _pts, boxes, *_ = best

    try:
        ctx.set_line_width(2.0)
        ctx.set_source_rgba(0.0, 1.0, 0.0, 0.95)

        for (x1, y1, x2, y2, label) in boxes:
            ctx.rectangle(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            ctx.stroke()

            text = str(label)
            ctx.save()
            ctx.set_source_rgba(0.0, 0.0, 0.0, 0.6)
            ctx.rectangle(x1, max(0, y1 - 18), max(40, len(text) * 8), 18)
            ctx.fill()
            ctx.restore()

            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(12)
            ctx.move_to(x1 + 3, max(0, y1 - 18) + 13)
            ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            ctx.show_text(text)
    except Exception:
        pass


# main

def main():
    args, passthrough = parse_args()
    apply_env(args, passthrough)

    vehicle_labels = set(VEHICLE_LABELS_BASE)
    if args.include_person:
        vehicle_labels.add("person")

    Gst.init(None)

    metrics = MetricsCollector(MetricWindows(step_sec=args.metrics_step_sec, max_total_sec=args.metrics_max_sec))
    stop_evt = start_metrics_tick_thread(metrics)

    state = AppState(args.orient.upper(), float(args.pos), vehicle_labels)
    state.metrics = metrics

    global STATE_REF
    STATE_REF = state

    app, pipeline = build_pipeline(state)
    configure_hailo_elements(pipeline)
    attach_callback(pipeline, state)

    def shutdown(*_):
        stop_evt.set()
        print("\nMETRICS SUMMARY")
        print(metrics.summary_text())
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        app.run()
    finally:
        stop_evt.set()
        print("\nMETRICS SUMMARY")
        print(metrics.summary_text())


if __name__ == "__main__":
    main()
