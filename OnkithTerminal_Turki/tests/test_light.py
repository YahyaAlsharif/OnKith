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


class LightLevelTests(unittest.TestCase):
    """Setting the lamp to a percentage rather than nudging it.

    The lane had only directions and switches in it, so a number had
    nowhere to travel. This is the one intent that carries a value.
    """

    def read(self, transcript: str):
        result = intercept_onkith_command(transcript)
        return result.intent, result.level

    def test_the_ways_people_name_a_level(self) -> None:
        for transcript, level in [
            ("Onkith set the light to 10 percent", 10),
            ("Onkith set light to 10", 10),
            ("Onkith set the light at 40%", 40),
            ("Onkith light 25 percent", 25),
            ("Onkith set the lamp to 60 percent", 60),
            ("Onkith put the light at half", 50),
            ("Onkith set the light to ninety percent", 90),
            ("Onkith set the light to zero percent", 0),
            ("Onkith set the light to one hundred percent", 100),
        ]:
            with self.subTest(transcript=transcript):
                self.assertEqual(self.read(transcript), ("LIGHT_LEVEL", level))

    def test_a_level_beats_the_direction_it_was_asked_with(self) -> None:
        # "Dim the light to 20" names a level. Reading it as one step
        # down would ignore the number the person actually said.
        self.assertEqual(self.read("Onkith dim the light to 20"), ("LIGHT_LEVEL", 20))
        self.assertEqual(
            self.read("Onkith brighten the light to 80"), ("LIGHT_LEVEL", 80)
        )

    def test_a_number_elsewhere_in_the_sentence_is_not_a_level(self) -> None:
        # The guard that earns its place. Without it this sets the lamp
        # to 20% instead of turning it on.
        self.assertEqual(
            self.read("Onkith turn the light on I paid 20 dollars for it"),
            ("LIGHT_ON", None),
        )

    def test_off_still_wins(self) -> None:
        self.assertEqual(self.read("Onkith turn the light off"), ("LIGHT_OFF", None))

    def test_a_question_containing_a_percentage_is_still_a_question(self) -> None:
        self.assertEqual(self.read("Onkith what is 10 percent of 40"), ("ASK", None))

    def test_a_level_outside_the_range_is_not_one(self) -> None:
        # "Set the light to 2024" is not somebody asking for a
        # brightness, and clamping it would pretend otherwise.
        intent, level = self.read("Onkith set the light to 2024")
        self.assertNotEqual(intent, "LIGHT_LEVEL")
        self.assertIsNone(level)

    def test_the_level_survives_the_five_word_command_cap(self) -> None:
        # command_text is capped at five words and would drop the number
        # off the end of the longest phrasing, so the level is read from
        # the full command instead.
        self.assertEqual(
            self.read("Onkith please set the light to ninety percent"),
            ("LIGHT_LEVEL", 90),
        )

    def test_the_hint_shows_the_recogniser_a_percentage(self) -> None:
        from integrated_pipeline.whisper_adapter import name_hint

        self.assertIn("percent", name_hint(["onkith"]))


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
        result = intercept_onkith_command(transcript)
        self.assertIn(
            result.intent,
            ("LIGHT_ON", "LIGHT_OFF", "BRIGHTER", "DIMMER", "LIGHT_LEVEL"),
            f"{transcript!r} did not reach the lamp",
        )
        self.core.handle_light(result.intent, result.level)

    def test_a_spoken_level_reaches_the_duty_cycle(self) -> None:
        self.say("Onkith set the light to 25 percent")
        self.assertEqual(self.backend.duties[-1], 25.0)
        self.say("Onkith set the light to zero percent")
        self.assertEqual(self.backend.duties[-1], 0.0)
        self.assertFalse(self.light.is_on)
        # And a level puts it back on without needing "on" said first.
        self.say("Onkith set the light to 70 percent")
        self.assertEqual(self.backend.duties[-1], 70.0)
        self.assertTrue(self.light.is_on)

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


class RecordingBus:
    """Collect published sections so the panel's inputs can be asserted."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    def publish(self, section: str, patch: dict) -> None:
        self.published.append((section, patch))


class LampOnTheScreenTests(unittest.TestCase):
    """What the devices panel is given to draw."""

    def test_the_backend_travels_with_every_update(self) -> None:
        # 0% means "switched off" on a Pi and "no lamp exists" on a
        # laptop, and the panel cannot tell those apart from the number.
        bus = RecordingBus()
        light = LightController(bus, {"default_brightness": 70}, backend=RecordingLight())
        light.on()
        section, patch = bus.published[-1]
        self.assertEqual(section, "light")
        self.assertEqual(patch["backend"], "recording")
        self.assertTrue(patch["on"])
        self.assertEqual(patch["brightness"], 70)

    def test_announcing_does_not_touch_the_lamp(self) -> None:
        # The whole point of the lazy backend is that nothing reaches a
        # GPIO because a browser connected. announce publishes and stops.
        bus = RecordingBus()
        backend = RecordingLight()
        light = LightController(bus, {}, backend=backend)
        light.announce()
        self.assertEqual(backend.duties, [])
        self.assertEqual(bus.published[-1][0], "light")

    def test_an_unwired_lamp_says_so_rather_than_reading_dark(self) -> None:
        bus = RecordingBus()
        light = LightController(bus, {}, backend=NullLight("no chip here"))
        light.announce()
        self.assertEqual(bus.published[-1][1]["backend"], "none")

    def test_off_reports_the_level_it_will_return_to(self) -> None:
        # The controller remembers brightness through an off so that on
        # comes back to it. The panel has to render that as dark anyway,
        # which is the surface's job, so the number must survive here.
        bus = RecordingBus()
        light = LightController(bus, {"default_brightness": 70}, backend=RecordingLight())
        light.set_brightness(40)
        light.off()
        patch = bus.published[-1][1]
        self.assertFalse(patch["on"])
        self.assertEqual(patch["brightness"], 40)


if __name__ == "__main__":
    unittest.main()
