import argparse
import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib

try:
    import hailo
    HAILO_OK = True
except Exception:
    HAILO_OK = False

from metrics_utils import MetricsCollector, MetricWindows


# Common

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
    p = argparse.ArgumentParser()
    p.add_argument("--hef", default="/models/scrfd_10g.hef")
    p.add_argument("--postproc-so", default="/resources/so/libscrfd.so")
    p.add_argument("--config", default="/resources/json/scrfd.json")
    p.add_argument("--function", default="scrfd_10g")

    p.add_argument("--cam-w", type=int, default=1280)
    p.add_argument("--cam-h", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam-format", default="NV12")

    p.add_argument("--sink", default="auto")

    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--pad", type=float, default=0.08)
    p.add_argument("--pixel-downscale", type=int, default=20)

    p.add_argument("--net-w", type=int, default=0)
    p.add_argument("--net-h", type=int, default=0)

    p.add_argument("--metrics-step-sec", type=int, default=600)
    p.add_argument("--metrics-max-sec", type=int, default=3600)
    return p.parse_args()


def pick_sink(s: str) -> str:
    if s != "auto":
        return s
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    return "waylandsink" if session == "wayland" else "autovideosink"


def detect_net_size(args):
    if args.net_w > 0 and args.net_h > 0:
        return args.net_w, args.net_h

    try:
        out = subprocess.check_output(["hailortcli", "parse-hef", args.hef], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"NHWC\((\d+)x(\d+)x(\d+)\)", out)
        if m and int(m.group(3)) == 3:
            return int(m.group(2)), int(m.group(1))
    except Exception:
        pass

    return 640, 640


# State

class AppState:
    def __init__(self, args, metrics: MetricsCollector):
        self.args = args
        self.metrics = metrics

        self.running = True
        self.loop = GLib.MainLoop()

        self.net_w, self.net_h = detect_net_size(args)
        self.sink = pick_sink(args.sink)

        self.in_pipe = None
        self.out_pipe = None
        self.appsrc = None
        self.out_caps = ""

        self.frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)

        self.dets_lock = threading.Lock()
        self.dets = []
        self.dets_ts = 0.0


# Pipeline

def build_pipeline(state: AppState):
    a = state.args
    q = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    cam_caps = f"video/x-raw,format={a.cam_format},width={a.cam_w},height={a.cam_h},framerate={a.fps}/1"

    in_desc = f"""
        libcamerasrc !
        {cam_caps} !
        tee name=t

        t. ! {q} !
            videoconvert !
            video/x-raw,format=RGB,width={a.cam_w},height={a.cam_h},pixel-aspect-ratio=1/1 !
            appsink name=preview_sink emit-signals=true max-buffers=1 drop=true sync=false enable-last-sample=false

        t. ! {q} !
            videoconvert !
            videoscale add-borders=true !
            video/x-raw,format=RGB,width={state.net_w},height={state.net_h},pixel-aspect-ratio=1/1 !
            {q} !
            hailonet name=hnet hef-path={a.hef} batch-size=1 force-writable=true !
            hailofilter name=hfilter so-path={a.postproc_so} function-name={a.function} config-path={a.config} !
            queue !
            identity name=identity_callback !
            fakesink sync=false
    """

    state.out_caps = f"video/x-raw,format=RGB,width={a.cam_w},height={a.cam_h},framerate={a.fps}/1"
    out_desc = f"""
        appsrc name=src is-live=true do-timestamp=true format=time block=false !
        {q} !
        videoconvert !
        video/x-raw,format=BGRx,width={a.cam_w},height={a.cam_h},framerate={a.fps}/1 !
        {state.sink} sync=true qos=false max-lateness=0
    """

    state.in_pipe = Gst.parse_launch(" ".join(in_desc.split()))
    state.out_pipe = Gst.parse_launch(" ".join(out_desc.split()))
    return state.in_pipe, state.out_pipe


def configure_hailo_elements(pipeline):
    if pipeline is None:
        return
    _hnet = pipeline.get_by_name("hnet")
    _hfilter = pipeline.get_by_name("hfilter")


def attach_callback(state: AppState):
    preview = state.in_pipe.get_by_name("preview_sink")
    preview.connect("new-sample", lambda s: on_preview_sample(s, state))

    identity = state.in_pipe.get_by_name("identity_callback")
    src_pad = identity.get_static_pad("src")
    src_pad.add_probe(Gst.PadProbeType.BUFFER, lambda p, i: app_callback(p, i, state))

    for pipe in (state.in_pipe, state.out_pipe):
        bus = pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda b, m: on_bus(b, m, state))

    state.appsrc = state.out_pipe.get_by_name("src")
    state.appsrc.set_property("caps", Gst.Caps.from_string(state.out_caps))


# Callbacks + Worker

def on_bus(_bus, msg, state: AppState):
    if msg.type == Gst.MessageType.QOS:
        try:
            state.metrics.on_qos()
        except Exception:
            pass
    if msg.type == Gst.MessageType.ERROR:
        try:
            state.metrics.on_error()
        except Exception:
            pass
        state.running = False
        try:
            state.loop.quit()
        except Exception:
            pass
    elif msg.type == Gst.MessageType.EOS:
        state.running = False
        try:
            state.loop.quit()
        except Exception:
            pass


def on_preview_sample(sink, state: AppState):
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    w = int(s.get_value("width"))
    h = int(s.get_value("height"))

    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK

    try:
        raw = np.frombuffer(mapinfo.data, dtype=np.uint8)
        stride = raw.size // h
        needed = w * 3
        if stride < needed:
            return Gst.FlowReturn.OK
        frame = raw.reshape((h, stride))[:, :needed].copy().reshape((h, w, 3))
    finally:
        buf.unmap(mapinfo)

    try:
        state.frame_q.put_nowait(frame)
    except queue.Full:
        try:
            _ = state.frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            state.frame_q.put_nowait(frame)
        except queue.Full:
            pass

    return Gst.FlowReturn.OK


def map_net_to_preview_letterbox(x1n, y1n, x2n, y2n, prev_w, prev_h, net_w, net_h):
    scale = min(net_w / prev_w, net_h / prev_h)
    new_w = prev_w * scale
    new_h = prev_h * scale
    pad_x = (net_w - new_w) / 2.0
    pad_y = (net_h - new_h) / 2.0

    x1 = (x1n - pad_x) / scale
    x2 = (x2n - pad_x) / scale
    y1 = (y1n - pad_y) / scale
    y2 = (y2n - pad_y) / scale
    return x1, y1, x2, y2


def pixelate_roi_rgb(rgb: np.ndarray, x1, y1, x2, y2, downscale: int):
    roi = rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    step_x = max(1, w // downscale)
    step_y = max(1, h // downscale)
    small = roi[::step_y, ::step_x]
    up = np.repeat(np.repeat(small, step_y, axis=0), step_x, axis=1)
    rgb[y1:y2, x1:x2] = up[:h, :w]


def app_callback(_pad, info, state: AppState):
    if not HAILO_OK:
        return Gst.PadProbeReturn.OK

    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    dets = []
    try:
        roi = hailo.get_roi_from_buffer(buf)
        objs = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for det in objs:
            conf = float(det.get_confidence())
            if conf < state.args.conf:
                continue
            bb = det.get_bbox()
            x1n = float(bb.xmin() * state.net_w)
            y1n = float(bb.ymin() * state.net_h)
            x2n = float(bb.xmax() * state.net_w)
            y2n = float(bb.ymax() * state.net_h)
            dets.append((x1n, y1n, x2n, y2n, conf))
    except Exception:
        dets = []

    with state.dets_lock:
        state.dets = dets
        state.dets_ts = time.time()

    return Gst.PadProbeReturn.OK


def worker_loop(state: AppState):
    while state.running:
        try:
            frame = state.frame_q.get(timeout=1.0)
        except queue.Empty:
            continue

        with state.dets_lock:
            dets = list(state.dets)
            dets_ts = state.dets_ts

        if time.time() - dets_ts > 0.7:
            dets = []

        out = frame
        h, w = out.shape[:2]

        for (x1n, y1n, x2n, y2n, _conf) in dets:
            x1, y1, x2, y2 = map_net_to_preview_letterbox(x1n, y1n, x2n, y2n, w, h, state.net_w, state.net_h)

            pad_px = int(min(x2 - x1, y2 - y1) * float(state.args.pad))
            x1 -= pad_px
            y1 -= pad_px
            x2 += pad_px
            y2 += pad_px

            cl = clamp_box(x1, y1, x2, y2, w, h)
            if not cl:
                continue

            x1i, y1i, x2i, y2i = cl
            pixelate_roi_rgb(out, x1i, y1i, x2i, y2i, int(state.args.pixel_downscale))

        data = out.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        ret = state.appsrc.emit("push-buffer", buf)
        if ret == Gst.FlowReturn.OK:
            try:
                state.metrics.on_frame()
            except Exception:
                pass


# main

def main():
    args = parse_args()

    for p in (args.hef, args.postproc_so, args.config):
        if not Path(p).exists():
            print(f"File not found: {p}")
            return

    Gst.init(None)

    metrics = MetricsCollector(MetricWindows(step_sec=args.metrics_step_sec, max_total_sec=args.metrics_max_sec))
    stop_evt = start_metrics_tick_thread(metrics)

    state = AppState(args, metrics)
    in_pipe, out_pipe = build_pipeline(state)
    configure_hailo_elements(in_pipe)
    attach_callback(state)

    out_pipe.set_state(Gst.State.PLAYING)
    in_pipe.set_state(Gst.State.PLAYING)

    threading.Thread(target=worker_loop, args=(state,), daemon=True).start()

    def shutdown(*_):
        state.running = False
        try:
            state.loop.quit()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        state.loop.run()
    finally:
        stop_evt.set()
        try:
            in_pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            out_pipe.set_state(Gst.State.NULL)
        except Exception:
            pass

        print("\nMETRICS SUMMARY")
        print(metrics.summary_text())


if __name__ == "__main__":
    main()