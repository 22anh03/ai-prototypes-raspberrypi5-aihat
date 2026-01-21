from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import psutil
    _PSUTIL_OK = True
except Exception:
    _PSUTIL_OK = False


# Probes

def read_cpu_temp_c() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_cpu_percent() -> Optional[float]:
    if not _PSUTIL_OK:
        return None
    try:
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def read_ram_mb() -> Optional[float]:
    if not _PSUTIL_OK:
        return None
    try:
        rss = psutil.Process(os.getpid()).memory_info().rss
        return float(rss) / (1024.0 * 1024.0)
    except Exception:
        return None


# Data

@dataclass
class MetricWindows:
    step_sec: int = 600
    max_total_sec: int = 3600


@dataclass
class SysSample:
    t_mon: float
    minute: int
    fps_avg: Optional[float]
    latency_ms_avg: Optional[float]
    cpu_temp_c: Optional[float]
    cpu_util_pct: Optional[float]
    ram_mb: Optional[float]


#Collector

@dataclass
class MetricsCollector:
    windows: MetricWindows = field(default_factory=MetricWindows)

    frames_processed: int = 0
    qos_events: int = 0
    errors: int = 0

    fps_instant: float = 0.0
    latency_ms_instant: float = 0.0

    samples: List[SysSample] = field(default_factory=list)

    _prev_wall: float = field(default_factory=time.time)
    _start_mon: float = field(default_factory=time.monotonic)
    _last_sample_mon: float = 0.0

    _interval_frames: int = 0
    _interval_latency_sum_ms: float = 0.0
    _interval_first_frame_mon: Optional[float] = None
    _interval_last_frame_mon: Optional[float] = None

    def __post_init__(self):
        if _PSUTIL_OK:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass


    # Hooks

    def on_frame(self) -> None:
        self.frames_processed += 1

        now_wall = time.time()
        dt = now_wall - self._prev_wall
        if dt > 0:
            self.fps_instant = 1.0 / dt
            self.latency_ms_instant = dt * 1000.0
        self._prev_wall = now_wall

        now_mon = time.monotonic()
        if self._interval_first_frame_mon is None:
            self._interval_first_frame_mon = now_mon
        self._interval_last_frame_mon = now_mon

        self._interval_frames += 1
        self._interval_latency_sum_ms += self.latency_ms_instant

    def on_qos(self, n: int = 1) -> None:
        self.qos_events += int(n)

    def on_error(self, n: int = 1) -> None:
        self.errors += int(n)

    def on_tick_1hz(self) -> None:
        now_mon = time.monotonic()
        if self._last_sample_mon != 0.0 and (now_mon - self._last_sample_mon) < float(self.windows.step_sec):
            return
        self._last_sample_mon = now_mon
        self._append_sample(now_mon)
        self._trim_samples(now_mon)


    # Internals

    def _interval_frame_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        if self._interval_frames <= 0:
            return None, None

        lat_avg = self._interval_latency_sum_ms / float(self._interval_frames)

        if self._interval_first_frame_mon is None or self._interval_last_frame_mon is None:
            return None, lat_avg

        dur = self._interval_last_frame_mon - self._interval_first_frame_mon
        if dur <= 0:
            return None, lat_avg

        fps_avg = float(self._interval_frames) / dur
        return fps_avg, lat_avg

    def _append_sample(self, now_mon: float) -> None:
        fps_avg, lat_avg = self._interval_frame_metrics()

        cpu_temp = read_cpu_temp_c()
        cpu_pct = read_cpu_percent()
        ram_mb = read_ram_mb()

        minute = int((now_mon - self._start_mon) // 60)

        self.samples.append(
            SysSample(
                t_mon=now_mon,
                minute=minute,
                fps_avg=fps_avg,
                latency_ms_avg=lat_avg,
                cpu_temp_c=cpu_temp,
                cpu_util_pct=cpu_pct,
                ram_mb=ram_mb,
            )
        )

        self._interval_frames = 0
        self._interval_latency_sum_ms = 0.0
        self._interval_first_frame_mon = None
        self._interval_last_frame_mon = None

    def _trim_samples(self, now_mon: float) -> None:
        cutoff = now_mon - float(self.windows.max_total_sec)
        i = 0
        while i < len(self.samples) and self.samples[i].t_mon < cutoff:
            i += 1
        if i > 0:
            self.samples = self.samples[i:]


    # Report

    @staticmethod
    def _fmt(x: Optional[float], nd: int = 2) -> str:
        if x is None:
            return "n/a"
        return f"{x:.{nd}f}"

    def summary_text(self) -> str:
        qos_rate = (self.qos_events / self.frames_processed * 100.0) if self.frames_processed else None

        lines = []
        lines.append(f"Frames processed: {self.frames_processed}")
        lines.append(f"Errors: {self.errors}")
        lines.append(f"QoS events: {self.qos_events}; QoS rate: {self._fmt(qos_rate, 2)}%")
        lines.append("")
        lines.append("[Instant]")
        lines.append(f" FPS:        {self._fmt(self.fps_instant, 2)}")
        lines.append(f" Latency ms: {self._fmt(self.latency_ms_instant, 2)}")
        lines.append("")
        lines.append(f"[Samples every {self.windows.step_sec}s (kept ~{self.windows.max_total_sec}s)]")

        if not self.samples:
            lines.append(" (no samples yet)")
            return "\n".join(lines)

        show = self.samples[-12:]

        header = "Time(min) | FPS(avg) | Lat(ms) | CPU(°C) | CPU(%) | RAM(MB)"
        lines.append(header)
        lines.append("-" * len(header))

        for s in show:
            lines.append(
                f"{s.minute:8d} | "
                f"{self._fmt(s.fps_avg, 2):7s} | "
                f"{self._fmt(s.latency_ms_avg, 2):6s} | "
                f"{self._fmt(s.cpu_temp_c, 2):7s} | "
                f"{self._fmt(s.cpu_util_pct, 2):6s} | "
                f"{self._fmt(s.ram_mb, 2):7s}"
            )

        return "\n".join(lines)