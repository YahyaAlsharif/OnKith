"""The hardware usage readout, and the arithmetic underneath it.

None of this can be tested by reading the machine it runs on: a laptop
has no /proc/stat and no thermal zone, and a Pi under test is not
reliably busy. So the kernel files are written into a temporary
directory and the module is pointed at them, which also proves the
degradation path when a file is simply absent.

The number that matters is a rate, not a reading, and rates are where
this kind of code goes wrong: a first sample with nothing to subtract
from, a counter that wrapped, a wall clock that moved while the CPU did
not. Those are the cases below.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import resources
from app.resources import ResourceMonitor, describe_throttling


def stat_file(busy: int, idle: int) -> str:
    """One aggregate line, in the field order the kernel uses."""
    return (
        f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\n"
        f"cpu0 {busy} 0 0 {idle} 0 0 0 0 0 0\n"
        "intr 12345\n"
    )


class KernelFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="onkith-usage-"))
        self.addCleanup(shutil.rmtree, directory, True)
        self.directory = directory

    def pointed_at(self, name: str, attribute: str, contents: str) -> Path:
        path = self.directory / name
        path.write_text(contents, encoding="utf-8")
        previous = getattr(resources, attribute)
        setattr(resources, attribute, str(path))
        self.addCleanup(setattr, resources, attribute, previous)
        return path


class WhatTheKernelReports(KernelFileTests):
    def test_busy_and_total_come_out_of_the_aggregate_line(self) -> None:
        self.pointed_at("stat", "_STAT", stat_file(busy=300, idle=700))
        self.assertEqual(resources._cpu_jiffies(), (300, 1000))

    def test_iowait_counts_as_idle(self) -> None:
        # The CPU is free for the capture callback while a disk is busy,
        # and a report that calls that "in use" would send somebody
        # looking for load that is not there.
        self.pointed_at(
            "stat", "_STAT", "cpu  100 0 0 500 400 0 0 0 0 0\n"
        )
        self.assertEqual(resources._cpu_jiffies(), (100, 1000))

    def test_a_missing_file_is_not_an_error(self) -> None:
        previous = resources._STAT
        resources._STAT = str(self.directory / "absent")
        self.addCleanup(setattr, resources, "_STAT", previous)
        self.assertIsNone(resources._cpu_jiffies())

    def test_memory_is_reported_as_used_not_as_free(self) -> None:
        self.pointed_at(
            "meminfo",
            "_MEMINFO",
            "MemTotal:        8000000 kB\n"
            "MemFree:         1000000 kB\n"
            "MemAvailable:    6000000 kB\n",
        )
        self.assertAlmostEqual(resources._memory_used_percent(), 25.0)

    def test_temperature_is_millidegrees(self) -> None:
        self.pointed_at("temp", "_THERMAL", "61234\n")
        self.assertAlmostEqual(resources._temperature_celsius(), 61.234)


class ThrottleWordTests(unittest.TestCase):
    def test_zero_says_so_in_words(self) -> None:
        self.assertEqual(
            describe_throttling(0), "no undervoltage or throttling since boot"
        )

    def test_now_and_earlier_are_told_apart(self) -> None:
        # The sticky bits are the ones that catch a demo: the board
        # recovered, so nothing is wrong at the moment it is read, and
        # something was wrong ten minutes ago.
        self.assertEqual(
            describe_throttling(0x50000),
            "undervoltage earlier, throttled earlier",
        )
        self.assertEqual(describe_throttling(0x1), "undervoltage now")

    def test_no_firmware_reports_nothing_rather_than_zero(self) -> None:
        self.assertIsNone(describe_throttling(None))


class SamplingTests(KernelFileTests):
    def monitor(self, cores: int = 4) -> ResourceMonitor:
        self.pointed_at("stat", "_STAT", stat_file(busy=0, idle=1000))
        return ResourceMonitor(interval=0.01, cores=cores)

    def test_the_first_sample_has_a_baseline_to_subtract_from(self) -> None:
        # Taken in the constructor, so the first reading is a rate over a
        # short interval rather than the whole uptime of the machine.
        monitor = self.monitor()
        self.pointed_at("stat", "_STAT", stat_file(busy=250, idle=1750))
        sample = monitor.sample()
        self.assertAlmostEqual(sample.system_cpu_percent, 25.0)

    def test_percent_is_of_the_whole_board_not_of_one_core(self) -> None:
        # 400% of a core is not an answer to "is this too much for a Pi".
        monitor = self.monitor(cores=4)
        monitor._cpu_seconds = 0.0
        monitor._wall -= 1.0
        real = resources._process_cpu_seconds
        self.addCleanup(setattr, resources, "_process_cpu_seconds", real)
        resources._process_cpu_seconds = lambda: 2.0
        sample = monitor.sample()
        self.assertAlmostEqual(sample.process_cpu_percent, 50.0, places=0)

    def test_an_unchanged_counter_reads_as_idle_not_as_a_divide_by_zero(
        self,
    ) -> None:
        monitor = self.monitor()
        sample = monitor.sample()
        self.assertIsNone(sample.system_cpu_percent)

    def test_the_status_line_is_empty_until_something_is_measured(self) -> None:
        self.assertEqual(ResourceMonitor(interval=0.01).status_line(), "")

    def test_the_status_line_leads_with_the_board_not_the_process(self) -> None:
        monitor = self.monitor()
        self.pointed_at("stat", "_STAT", stat_file(busy=910, idle=1090))
        self.pointed_at("temp", "_THERMAL", "58000\n")
        monitor.sample()
        line = monitor.status_line()
        self.assertIn("cpu 91%", line)
        self.assertIn("58C", line)


class ReportTests(KernelFileTests):
    def loaded(self, percents: list[int]) -> ResourceMonitor:
        self.pointed_at("stat", "_STAT", stat_file(busy=0, idle=1000))
        monitor = ResourceMonitor(interval=0.01, cores=4)
        busy = 0
        idle = 1000
        for percent in percents:
            busy += percent * 10
            idle += (100 - percent) * 10
            self.pointed_at("stat", "_STAT", stat_file(busy, idle))
            monitor.sample()
        return monitor

    def test_nothing_measured_says_nothing_measured(self) -> None:
        self.assertIn("no samples", ResourceMonitor().report())

    def test_a_saturated_board_is_named_as_saturated(self) -> None:
        report = self.loaded([94, 96, 95]).report()
        self.assertIn("saturated", report)

    def test_a_burst_is_told_apart_from_a_sustained_load(self) -> None:
        # Transcription is bursty by design. A peak alone is not a
        # verdict, and reporting it as one would send somebody shopping
        # for a faster board they do not need.
        report = self.loaded([5, 95, 5, 5]).report()
        self.assertIn("Bursty", report)
        self.assertNotIn("saturated", report)

    def test_the_peak_survives_alongside_the_mean(self) -> None:
        report = self.loaded([10, 90, 10]).report()
        self.assertIn("peak 90%", report)
        self.assertIn("mean 37%", report)

    def test_a_hot_board_is_named_before_the_demo_not_during_it(self) -> None:
        self.pointed_at("temp", "_THERMAL", "84000\n")
        self.assertIn("throttling range", self.loaded([10, 10]).report())

    def test_a_cool_board_says_so_rather_than_staying_silent(self) -> None:
        self.pointed_at("temp", "_THERMAL", "47000\n")
        self.assertIn("under throttling", self.loaded([10, 10]).report())


if __name__ == "__main__":
    unittest.main()
