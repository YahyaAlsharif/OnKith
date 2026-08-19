"""The lamp, on hardware PWM, owned by the core process.

Faris got this working on the Pi 5 from a terminal script driving PWM0
directly, and what is kept here is that flow rather than the program
around it. His version was a curses UI with four animation modes; this
device needs one thing from it, which is a level that voice can move,
so the animator and the keyboard are gone and the hardware path is the
same: one hardware PWM channel at a fixed frequency, brightness written
as a duty cycle, and the channel stopped on the way out.

Two of his findings are load-bearing and are recorded here rather than
in a note somewhere. The overlay has to be in /boot/firmware/config.txt
before any of this exists:

    dtoverlay=pwm,pin=12,func=4

and the chip number that overlay creates has to be confirmed on the
board rather than assumed, with `ls /sys/class/pwm/`. It was pwmchip0
for GPIO12 on his setup and the kernel is free to number it otherwise.
Both are configurable here for exactly that reason.

Off the Pi there is no PWM and no sysfs to talk to, so this falls to a
backend that does nothing and says so once. That is deliberate: the
voice path must not fail on a laptop because a lamp is missing, and
half of this project is developed on one.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Protocol

LogFunction = Callable[[str], None]

PWM_SYSFS_ROOT = Path("/sys/class/pwm")


class LightBackend(Protocol):
    name: str

    def set_duty(self, percent: float) -> None:
        ...

    def close(self) -> None:
        ...


class NullLight:
    """No lamp on this machine. Say so once and carry on."""

    name = "none"

    def __init__(self, reason: str, log: LogFunction = lambda message: None) -> None:
        self.reason = reason
        self.log = log
        self._warned = False

    def set_duty(self, percent: float) -> None:
        if not self._warned:
            self._warned = True
            self.log(f"No lamp is wired on this machine: {self.reason}")

    def close(self) -> None:
        return


class HardwarePwmLight:
    """One hardware PWM channel, brightness written as a duty cycle.

    The channel is opened on the first instruction rather than at
    startup. Nothing should touch a GPIO because the app happened to
    boot, and it means an unwired board costs one warning instead of
    stopping the process.
    """

    name = "pwm"

    def __init__(
        self,
        *,
        chip: int,
        channel: int,
        frequency_hz: int,
        log: LogFunction = lambda message: None,
    ) -> None:
        self.chip = int(chip)
        self.channel = int(channel)
        self.frequency_hz = int(frequency_hz)
        self.log = log
        self._pwm: Any | None = None

    def _open(self) -> Any:
        if self._pwm is None:
            from rpi_hardware_pwm import HardwarePWM

            pwm = HardwarePWM(
                pwm_channel=self.channel,
                hz=self.frequency_hz,
                chip=self.chip,
            )
            pwm.start(0)
            self._pwm = pwm
        return self._pwm

    def set_duty(self, percent: float) -> None:
        self._open().change_duty_cycle(max(0.0, min(100.0, float(percent))))

    def close(self) -> None:
        if self._pwm is None:
            return
        try:
            # Dark before released, so a lamp is never left burning by a
            # process that has gone away.
            self._pwm.change_duty_cycle(0)
            self._pwm.stop()
        except Exception as error:
            self.log(f"The lamp could not be released cleanly: {error}")
        finally:
            self._pwm = None


def detect_backend(
    *,
    chip: int,
    channel: int,
    frequency_hz: int,
    log: LogFunction = lambda message: None,
) -> LightBackend:
    """The real lamp if this board has one, otherwise a quiet stand-in.

    Both conditions are checked rather than caught later. The import
    fails on any machine without the library, and the sysfs directory is
    missing on a Pi whose overlay is not in config.txt yet, which is the
    one that would otherwise look like a wiring fault.
    """

    try:
        import rpi_hardware_pwm  # noqa: F401
    except Exception:
        return NullLight(
            "rpi-hardware-pwm is not installed, which is expected off the Pi",
            log,
        )
    chip_path = PWM_SYSFS_ROOT / f"pwmchip{int(chip)}"
    if not chip_path.exists():
        return NullLight(
            f"{chip_path} does not exist. Add dtoverlay=pwm,pin=12,func=4 to "
            "/boot/firmware/config.txt, reboot, and confirm the chip number "
            "with ls /sys/class/pwm/",
            log,
        )
    return HardwarePwmLight(
        chip=chip,
        channel=channel,
        frequency_hz=frequency_hz,
        log=log,
    )


class LightController:
    """On, off and a level, which is all voice can ask for.

    The brightness is remembered across off, so "light off" then "light
    on" returns to where it was rather than to a default. Somebody who
    set it low did that on purpose.
    """

    def __init__(
        self,
        bus: Any = None,
        settings: dict[str, Any] | None = None,
        *,
        backend: LightBackend | None = None,
        log: LogFunction = lambda message: None,
    ) -> None:
        values = dict(settings or {})
        self.step_size = int(values.get("step", 15))
        self.default_brightness = _bounded(values.get("default_brightness", 70))
        self.log = log
        self.bus = bus
        self._lock = threading.Lock()
        self._brightness = self.default_brightness
        self._on = False
        self._backend = backend if backend is not None else detect_backend(
            chip=int(values.get("chip", 0)),
            channel=int(values.get("channel", 0)),
            frequency_hz=int(values.get("frequency_hz", 1000)),
            log=log,
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def brightness(self) -> int:
        with self._lock:
            return self._brightness

    @property
    def is_on(self) -> bool:
        with self._lock:
            return self._on

    def on(self) -> None:
        with self._lock:
            self._on = True
            # Turned off by stepping all the way down, then turned back
            # on, would otherwise light nothing and look broken.
            if self._brightness <= 0:
                self._brightness = self.default_brightness
        self._apply()

    def off(self) -> None:
        with self._lock:
            self._on = False
        self._apply()

    def step(self, delta: int) -> None:
        """Move the level, and treat asking for brighter as asking for on.

        Somebody says "brighter" at a dark lamp because they want light,
        not because they want a number changed behind the scenes.
        """

        with self._lock:
            self._brightness = _bounded(self._brightness + int(delta))
            if delta > 0:
                self._on = True
            elif self._brightness <= 0:
                self._on = False
        self._apply()

    def set_brightness(self, value: float) -> None:
        with self._lock:
            self._brightness = _bounded(value)
            self._on = self._brightness > 0
        self._apply()

    def _apply(self) -> None:
        with self._lock:
            duty = float(self._brightness) if self._on else 0.0
            # The backend travels with every update rather than being
            # announced once at startup. A surface can connect at any
            # moment, and "the lamp says 0%" means two entirely different
            # things depending on whether a wire is attached: on a Pi it is
            # off, on a laptop there was never a lamp.
            state = {
                "on": self._on,
                "brightness": self._brightness,
                "backend": self._backend.name,
            }
        try:
            self._backend.set_duty(duty)
        except Exception as error:
            # A lamp that cannot be driven must never reach back into the
            # voice path. It is the least important thing on this device.
            self.log(f"The lamp could not be set: {error}")
        if self.bus is not None:
            try:
                self.bus.publish("light", state)
            except Exception:
                pass

    def announce(self) -> None:
        """Publish the lamp's state without touching the lamp.

        Deliberately not ``_apply``. That drives the hardware, and opening
        a PWM channel because a surface happened to connect is exactly what
        the lazy backend exists to avoid. This is the read-only half: it
        lets the devices panel draw an unlit lamp, and say whether that is
        because it is switched off or because no lamp is wired at all.
        """

        if self.bus is None:
            return
        with self._lock:
            state = {
                "on": self._on,
                "brightness": self._brightness,
                "backend": self._backend.name,
            }
        try:
            self.bus.publish("light", state)
        except Exception:
            pass

    def close(self) -> None:
        self._backend.close()


def _bounded(value: Any) -> int:
    return max(0, min(100, int(round(float(value)))))
