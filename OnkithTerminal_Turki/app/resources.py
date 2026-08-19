"""What the board is spending on us, sampled while it runs.

The question this exists to answer is "is this too much for a Pi 5",
and that question is not answerable from a single number taken at the
end: the load is bursty. Whisper runs for a second and stops, mpv
decodes continuously, the masker wakes up per utterance. So this
samples once a second, keeps the peaks alongside the means, and prints
both when the run ends.

No new dependency. psutil would be the obvious tool and it is not worth
another wheel on an armhf-adjacent board, so this reads /proc directly
on Linux and degrades to what os and resource can tell it elsewhere.
"""

from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

_STAT = "/proc/stat"
_MEMINFO = "/proc/meminfo"
_STATM = "/proc/self/statm"
_THERMAL = "/sys/class/thermal/thermal_zone0/temp"

SAMPLE_INTERVAL_SECONDS = 1.0

# Beyond these the board is not keeping up, and saying so is the whole
# point of the report. 85% system CPU sustained leaves nothing for the
# capture callback; the Pi 5 starts throttling at 80C and hard-limits
# at 85C.
BUSY_CPU_PERCENT = 85.0
HOT_CELSIUS = 80.0


@dataclass(frozen=True)
class Sample:
    """One second of the machine."""

    system_cpu_percent: float | None
    process_cpu_percent: float
    rss_bytes: int | None
    memory_used_percent: float | None
    temperature_celsius: float | None
    load_one_minute: float | None


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _cpu_jiffies() -> tuple[int, int] | None:
    """(busy, total) from the aggregate line of /proc/stat."""
    text = _read(_STAT)
    if not text:
        return None
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        fields = [int(value) for value in line.split()[1:] if value.isdigit()]
        if len(fields) < 4:
            return None
        total = sum(fields)
        # idle and iowait. iowait counts as idle for our purposes: the
        # CPU is available to the capture callback either way.
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return total - idle, total
    return None


def _process_cpu_seconds() -> float:
    """CPU seconds burned by us and by children we have reaped.

    Children only land here once they are waited on, so a long-running
    mpv contributes nothing until it exits. That is why the system
    number above exists and why it is the one the report leads with.
    """
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


def _rss_bytes() -> int | None:
    text = _read(_STATM)
    if text:
        fields = text.split()
        if len(fields) >= 2:
            try:
                return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
            except (ValueError, OSError):
                return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not peak:
        return None
    # Linux reports kilobytes here, Darwin reports bytes. This branch is
    # only reached off Linux, so bytes.
    return int(peak)


def _memory_used_percent() -> float | None:
    text = _read(_MEMINFO)
    if not text:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name] = int(parts[0])
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return (total - available) / total * 100.0


def _temperature_celsius() -> float | None:
    text = _read(_THERMAL)
    if not text:
        return None
    try:
        return int(text.strip()) / 1000.0
    except ValueError:
        return None


def _throttled_flags() -> int | None:
    """The sticky throttle word from the Pi firmware, if there is one."""
    try:
        output = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    _, _, value = output.strip().partition("=")
    try:
        return int(value, 0)
    except ValueError:
        return None


def describe_throttling(flags: int | None) -> str | None:
    if flags is None:
        return None
    if flags == 0:
        return "no undervoltage or throttling since boot"
    reasons = []
    if flags & 0x1:
        reasons.append("undervoltage now")
    if flags & 0x2:
        reasons.append("frequency capped now")
    if flags & 0x4:
        reasons.append("throttled now")
    if flags & 0x10000:
        reasons.append("undervoltage earlier")
    if flags & 0x40000:
        reasons.append("throttled earlier")
    return ", ".join(reasons) or f"flags {flags:#x}"


def _bytes_human(count: int | None) -> str:
    if count is None:
        return "n/a"
    megabytes = count / (1024 * 1024)
    if megabytes >= 1024:
        return f"{megabytes / 1024:.1f} GB"
    return f"{megabytes:.0f} MB"


class _Track:
    """Peak and mean of one series, without keeping the series."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.peak: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self.count += 1
        self.total += value
        self.peak = value if self.peak is None else max(self.peak, value)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


class ResourceMonitor:
    """Samples the machine on its own thread until asked to stop."""

    def __init__(
        self,
        *,
        interval: float = SAMPLE_INTERVAL_SECONDS,
        on_sample: Callable[[Sample], None] | None = None,
        cores: int | None = None,
    ) -> None:
        self.interval = interval
        self._on_sample = on_sample
        self.cores = cores or os.cpu_count() or 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: Sample | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._system = _Track()
        self._process = _Track()
        self._rss = _Track()
        self._memory = _Track()
        self._temperature = _Track()
        self._load = _Track()
        self._jiffies = _cpu_jiffies()
        self._cpu_seconds = _process_cpu_seconds()
        self._wall = time.monotonic()
        self._throttled_at_start = _throttled_flags()

    @property
    def latest(self) -> Sample | None:
        return self._latest

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            return self
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._loop, name="resource-monitor", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 0.5)
            self._thread = None
        if self._stopped_at is None:
            self._stopped_at = time.time()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                sample = self.sample()
            except Exception:
                continue
            if self._on_sample is not None:
                try:
                    self._on_sample(sample)
                except Exception:
                    pass

    def sample(self) -> Sample:
        """Take one reading and fold it into the running aggregates."""
        now = time.monotonic()
        elapsed = max(now - self._wall, 1e-6)
        self._wall = now

        system_percent: float | None = None
        jiffies = _cpu_jiffies()
        if jiffies is not None and self._jiffies is not None:
            busy = jiffies[0] - self._jiffies[0]
            total = jiffies[1] - self._jiffies[1]
            if total > 0:
                system_percent = busy / total * 100.0
        if jiffies is not None:
            self._jiffies = jiffies

        cpu_seconds = _process_cpu_seconds()
        # Of the whole machine, not of one core: 400% of a core means
        # nothing to somebody asking whether the board is coping.
        process_percent = (
            (cpu_seconds - self._cpu_seconds) / (elapsed * self.cores) * 100.0
        )
        self._cpu_seconds = cpu_seconds

        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):
            load = None

        sample = Sample(
            system_cpu_percent=system_percent,
            process_cpu_percent=max(0.0, process_percent),
            rss_bytes=_rss_bytes(),
            memory_used_percent=_memory_used_percent(),
            temperature_celsius=_temperature_celsius(),
            load_one_minute=load,
        )
        self._latest = sample
        self._system.add(sample.system_cpu_percent)
        self._process.add(sample.process_cpu_percent)
        self._rss.add(sample.rss_bytes)
        self._memory.add(sample.memory_used_percent)
        self._temperature.add(sample.temperature_celsius)
        self._load.add(sample.load_one_minute)
        return sample

    def status_line(self) -> str:
        """The one-line form, for the bar at the bottom of the terminal."""
        sample = self._latest
        if sample is None:
            return ""
        parts = []
        percent = (
            sample.system_cpu_percent
            if sample.system_cpu_percent is not None
            else sample.process_cpu_percent
        )
        parts.append(f"cpu {percent:.0f}%")
        if sample.rss_bytes is not None:
            parts.append(f"rss {_bytes_human(sample.rss_bytes)}")
        if sample.temperature_celsius is not None:
            parts.append(f"{sample.temperature_celsius:.0f}C")
        return "  ".join(parts)

    def report(self) -> str:
        """The whole run, as lines to print after the screen is released."""
        if self._system.count == 0 and self._process.count == 0:
            return "hardware usage: no samples taken"

        started = self._started_at or time.time()
        stopped = self._stopped_at or time.time()
        lines = [
            "Hardware usage",
            f"  ran for            {stopped - started:.0f}s over "
            f"{self._process.count} sample(s) on {self.cores} core(s)",
        ]

        def row(label: str, track: _Track, unit: str, digits: int = 0) -> None:
            if track.count == 0:
                lines.append(f"  {label:<18} not available on this platform")
                return
            lines.append(
                f"  {label:<18} mean {track.mean:.{digits}f}{unit}   "
                f"peak {track.peak:.{digits}f}{unit}"
            )

        row("whole machine CPU", self._system, "%")
        row("this process CPU", self._process, "%")
        if self._rss.count:
            lines.append(
                f"  {'resident memory':<18} mean "
                f"{_bytes_human(int(self._rss.mean or 0))}   "
                f"peak {_bytes_human(int(self._rss.peak or 0))}"
            )
        row("memory in use", self._memory, "%")
        row("temperature", self._temperature, "C")
        row("load average", self._load, "", digits=2)

        throttled = describe_throttling(_throttled_flags())
        if throttled is not None:
            lines.append(f"  {'firmware':<18} {throttled}")

        lines.append("")
        lines.extend(f"  {verdict}" for verdict in self._verdicts())
        return "\n".join(lines)

    def _verdicts(self) -> list[str]:
        """Whether this was too much, said in words rather than numbers."""
        verdicts: list[str] = []
        busy = self._system if self._system.count else self._process
        mean = busy.mean
        peak = busy.peak
        if mean is not None and mean >= BUSY_CPU_PERCENT:
            verdicts.append(
                f"CPU averaged {mean:.0f}%. The board was saturated for most "
                "of this run; capture overflows are expected."
            )
        elif peak is not None and peak >= BUSY_CPU_PERCENT:
            verdicts.append(
                f"CPU peaked at {peak:.0f}% but averaged {mean:.0f}%. Bursty, "
                "which is the shape transcription makes. Headroom overall."
            )
        elif mean is not None:
            verdicts.append(f"CPU averaged {mean:.0f}%. Comfortable headroom.")

        hottest = self._temperature.peak
        if hottest is not None and hottest >= HOT_CELSIUS:
            verdicts.append(
                f"Peaked at {hottest:.0f}C, inside the throttling range. It "
                "needs a fan or a heatsink before a long demo."
            )
        elif hottest is not None:
            verdicts.append(f"Peaked at {hottest:.0f}C, well under throttling.")

        memory = self._memory.peak
        if memory is not None and memory >= 90.0:
            verdicts.append(
                f"Memory reached {memory:.0f}% used. Close to swapping, which "
                "on an SD card is worse than a slow model."
            )
        return verdicts or ["Nothing measurable was recorded."]
