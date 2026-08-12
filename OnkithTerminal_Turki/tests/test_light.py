"""The lamp: the lane that reaches it, and the controller behind it.

Nothing here touches a GPIO. The hardware path is Faris's, already
working on the Pi from a terminal, and what can go wrong in this repo is
the wiring around it: which words reach the lamp rather than the music,
and whether the level behaves the way somebody standing in front of it
expects.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.light import LightController, NullLight, detect_backend
from app.onkith_music import intercept_onkith_command


class RecordingLight:
    """Stand in for the PWM channel and remember every duty written."""

    name = "recording"

    def __init__(self) -> None:
        self.duties: list[float] = []
        self.closed = False

    def set_duty(self, percent: float) -> None:
        self.duties.append(percent)

    def close(self) -> None:
        self.closed = True


def controller(**settings) -> tuple[LightController, RecordingLight]:
    backend = RecordingLight()
    values = {"step": 15, "default_brightness": 70}
    values.update(settings)
    return LightController(None, values, backend=backend), backend


class LightLaneTests(unittest.TestCase):
    def lane(self, transcript: str):
        return intercept_onkith_command(transcript).intent

    def test_the_ways_people_ask_for_the_lamp(self) -> None:
        for transcript, expected in [
            ("Onkith turn the light on", "LIGHT_ON"),
            ("Onkith turn on the light", "LIGHT_ON"),
            ("Onkith lights on", "LIGHT_ON"),
            ("Onkith turn the light off", "LIGHT_OFF"),
            ("Onkith turn off the lights", "LIGHT_OFF"),
            ("Onkith light off", "LIGHT_OFF"),
            ("Onkith brighter", "BRIGHTER"),
            ("Onkith dimmer", "DIMMER"),
            ("Onkith dim the light", "DIMMER"),
            ("Onkith make the light brighter", "BRIGHTER"),
            ("Onkith turn the light up", "BRIGHTER"),
            ("Onkith turn the light down", "DIMMER"),
            ("Onkith increase the brightness", "BRIGHTER"),
            ("Onkith lower the lights", "DIMMER"),
        ]:
            with self.subTest(transcript=transcript):
                self.assertEqual(self.lane(transcript), expected)

    def test_close_the_light_is_how_people_here_say_it(self) -> None:
        # Everyone who will be standing in front of this on the day says
        # it this way, and it costs nothing to accept.
        self.assertEqual(self.lane("Onkith close the light"), "LIGHT_OFF")
        self.assertEqual(self.lane("Onkith open the light"), "LIGHT_ON")

    def test_the_lamp_never_takes_a_command_meant_for_the_music(self) -> None:
        # These share every verb with the lamp. The noun is the only
        # thing separating them, which is why the lamp insists on one.
        for transcript, expected in [
            ("Onkith turn it off", "CLOSE"),
            ("Onkith stop", "CLOSE"),
            ("Onkith turn it up", "LOUDER"),
            ("Onkith turn it down", "QUIETER"),
            ("Onkith louder", "LOUDER"),
            ("Onkith lower the volume", "QUIETER"),
            ("Onkith close the player", "HIDE"),
        ]:
            with self.subTest(transcript=transcript):
                self.assertEqual(self.lane(transcript), expected)

    def test_a_question_about_the_light_is_still_a_question(self) -> None:
        self.assertEqual(self.lane("Onkith why is the light so bright"), "ASK")
        self.assertEqual(self.lane("Onkith what colour is the light"), "ASK")

    def test_the_hint_shows_the_recogniser_the_word_light(self) -> None:
        from integrated_pipeline.whisper_adapter import name_hint

        # The word, not both wordings. One example per lane: eight
        # examples cost a digit out of an id number, and "on" needs no
        # help from a prompt.
        hint = name_hint(["onkith"])
        self.assertIn("light", hint)
        self.assertIn("Onkith, turn the light off.", hint)

    def test_every_lane_the_device_has_is_somewhere_in_the_hint(self) -> None:
        """The hint and the lanes have to grow together.

        A lane with no word in the hint is a command the recogniser was
        never told about, and it will mishear it into conversation. This
        is the check that survives the hint being trimmed, which it was:
        it names the vocabulary each lane needs rather than a sentence.
        """

        from integrated_pipeline.whisper_adapter import name_hint

        hint = name_hint(["onkith"]).lower()
        for lane, word in (
            ("PLAY", "play"),
            ("CLOSE", "stop"),
            ("SHOW", "show"),
            ("HIDE", "hide"),
            ("LOUDER/QUIETER", "volume"),
            ("LIGHT_*", "light"),
        ):
            with self.subTest(lane=lane):
                self.assertIn(word, hint)


class LightControllerTests(unittest.TestCase):
    def test_on_lights_it_at_the_remembered_level(self) -> None:
        light, backend = controller()
        light.on()
        self.assertTrue(light.is_on)
        self.assertEqual(backend.duties[-1], 70.0)

    def test_off_goes_dark_but_does_not_forget(self) -> None:
        # Somebody who set it low did that on purpose, so coming back on
        # returns to where they left it rather than to a default.
        light, backend = controller()
        light.on()
        light.step(-15)
        self.assertEqual(light.brightness, 55)
        light.off()
        self.assertEqual(backend.duties[-1], 0.0)
        self.assertFalse(light.is_on)
        light.on()
        self.assertEqual(backend.duties[-1], 55.0)

    def test_brighter_at_a_dark_lamp_turns_it_on(self) -> None:
        # "Brighter" at a lamp that is off means somebody wants light,
        # not a number changed behind the scenes.
        light, backend = controller()
        self.assertFalse(light.is_on)
        light.step(15)
        self.assertTrue(light.is_on)
        self.assertGreater(backend.duties[-1], 0.0)

    def test_dimming_all_the_way_down_is_off(self) -> None:
        light, backend = controller(default_brightness=20)
        light.on()
        light.step(-15)
        light.step(-15)
        self.assertEqual(light.brightness, 0)
        self.assertFalse(light.is_on)
        self.assertEqual(backend.duties[-1], 0.0)
        # And asking for it back gives light rather than a dark lamp
        # reporting itself on.
        light.on()
        self.assertEqual(backend.duties[-1], 20.0)

    def test_the_level_never_leaves_its_range(self) -> None:
        light, backend = controller()
        for _ in range(10):
            light.step(15)
        self.assertEqual(light.brightness, 100)
        self.assertEqual(backend.duties[-1], 100.0)
        for _ in range(20):
            light.step(-15)
        self.assertEqual(light.brightness, 0)

    def test_a_backend_that_throws_never_reaches_the_caller(self) -> None:
        """The lamp is the least important thing on this device.

        The voice path runs through here, so a loose wire must not be
        able to take a transcript down with it.
        """

        class Broken:
            name = "broken"

            def set_duty(self, percent: float) -> None:
                raise RuntimeError("the lamp fell off")

            def close(self) -> None:
                raise RuntimeError("and again on the way out")

        messages: list[str] = []
        light = LightController(None, {}, backend=Broken(), log=messages.append)
        light.on()
        light.step(-15)
        light.off()
        self.assertTrue(messages)

    def test_it_goes_dark_when_it_is_released(self) -> None:
        light, backend = controller()
        light.on()
        light.close()
        self.assertTrue(backend.closed)


class SpokenToDutyTests(unittest.TestCase):
    """Words in, duty cycle out, through the real dispatch.

    The lane, the core's handler and the controller are each covered
    above. This is the one that would catch an intent that everything
    agrees on and nothing acts upon.
    """

    def setUp(self) -> None:
        from app.core_runtime import CoreRuntime

        self.light, self.backend = controller()
        holder = self

        class Core:
            light = holder.light
            handle_light = CoreRuntime.handle_light

        self.core = Core()

    def say(self, transcript: str) -> None:
        intent = intercept_onkith_command(transcript).intent
        self.assertIn(
            intent,
            ("LIGHT_ON", "LIGHT_OFF", "BRIGHTER", "DIMMER"),
            f"{transcript!r} did not reach the lamp",
        )
        self.core.handle_light(intent)

    def test_a_spoken_instruction_reaches_the_duty_cycle(self) -> None:
        self.say("Onkith turn the light on")
        self.assertEqual(self.backend.duties[-1], 70.0)
        self.say("Onkith dimmer")
        self.assertEqual(self.backend.duties[-1], 55.0)
        self.say("Onkith brighter")
        self.assertEqual(self.backend.duties[-1], 70.0)
        self.say("Onkith turn the light off")
        self.assertEqual(self.backend.duties[-1], 0.0)


class BackendDetectionTests(unittest.TestCase):
    def test_a_machine_with_no_pwm_gets_a_lamp_that_does_nothing(self) -> None:
        # Half of this project is developed on a laptop. A missing lamp
        # is not an error there, and it must not be one here.
        backend = detect_backend(chip=0, channel=0, frequency_hz=1000)
        if backend.name == "pwm":
            self.skipTest("this machine has hardware PWM")
        self.assertIsInstance(backend, NullLight)

    def test_the_reason_says_what_to_do_about_it(self) -> None:
        # A Pi with no overlay in config.txt looks exactly like a wiring
        # fault unless the message says otherwise.
        messages: list[str] = []
        NullLight("no pwmchip0", messages.append).set_duty(50)
        self.assertEqual(len(messages), 1)
        self.assertIn("no pwmchip0", messages[0])
        # Said once, not on every instruction.
        NullLight("no pwmchip0", messages.append).set_duty(50)
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
