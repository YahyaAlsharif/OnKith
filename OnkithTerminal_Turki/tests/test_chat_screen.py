"""The screen is a conversation, and the player is a companion to it.

The page used to be the player. It is now the room: the creature in the
middle, what was heard and what was said around it, and the recorder in a
panel that arrives when there is something to play.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE = (ROOT / "app" / "player_template.html").read_text(encoding="utf-8")
APP = (ROOT / "assets" / "player" / "player-app-v1.mjs").read_text(encoding="utf-8")


class ChatScreenMarkupTest(unittest.TestCase):
    def test_the_conversation_is_the_screen_and_the_player_is_beside_it(
        self,
    ) -> None:
        self.assertIn('<main class="room"', PAGE)
        self.assertIn('<aside class="player-panel" id="player-panel"', PAGE)
        # The recorder and everything that drives it moved inside the panel
        # rather than being rebuilt, so the player keeps every id it had.
        panel = PAGE.split('id="player-panel"', maxsplit=1)[1].split("</aside>")[0]
        for kept in (
            'id="machine-viewer"',
            'id="machine-hotspots"',
            'id="track-flow"',
            'id="progress"',
            'id="fallback-controls"',
        ):
            with self.subTest(kept=kept):
                self.assertIn(kept, panel)

    def test_the_creature_is_the_mark_and_its_shelter_never_moves(self) -> None:
        room = PAGE.split('<main class="room"', maxsplit=1)[1].split("</main>")[0]
        for part in ('class="cover"', 'class="wall"', 'class="sheltered"'):
            with self.subTest(part=part):
                self.assertIn(part, room)
        # Only the sheltered block is animated. The eave it lives under is
        # the constant, and the life inside it is what changes.
        for state in ("resting", "listening", "thinking", "speaking"):
            with self.subTest(state=state):
                self.assertIn(f'[data-pet="{state}"] .mark .sheltered', PAGE)
        self.assertNotIn(".mark .cover { animation", PAGE)
        self.assertNotIn(".mark .wall { animation", PAGE)

    def test_the_resting_device_is_colourless(self) -> None:
        """The rule the whole system hangs on.

        Colour says something is happening. Listening, thinking and speaking
        take the accent; at rest it takes none.
        """

        for state in ("listening", "thinking", "speaking"):
            with self.subTest(state=state):
                self.assertIn(f'[data-pet="{state}"] .mark .sheltered', PAGE)
        accent_rule = PAGE.split(
            '[data-pet="listening"] .mark .sheltered,', maxsplit=1
        )[1].split("}", maxsplit=1)[0]
        self.assertIn("fill: var(--lift)", accent_rule)
        self.assertNotIn('[data-pet="resting"] .mark .sheltered { fill: var(--lift)', PAGE)

    def test_the_panel_is_not_animated_by_its_grid_track(self) -> None:
        """Measured in a browser, that transition never progresses.

        It reports the closed value forever and leaves the panel a sliver
        wide, with fr tracks and with percentages alike. The panel snaps to
        width and fades itself in, which is also far cheaper on a Pi given
        there is a WebGL canvas inside the track being resized.
        """

        shell = re.search(r"\n    \.app-shell \{\n(.*?)\n    \}", PAGE, re.S)
        self.assertIsNotNone(shell)
        self.assertNotIn("transition: grid-template", shell.group(1))
        panel = re.search(r"\n    \.player-panel \{\n(.*?)\n    \}", PAGE, re.S)
        self.assertIsNotNone(panel)
        self.assertIn("opacity: 0;", panel.group(1))
        self.assertIn("transition: opacity", panel.group(1))

    def test_the_panel_sizes_against_itself_not_the_window(self) -> None:
        # The player was written when it was the whole page, so its type and
        # rail are measured in vw. In a panel that is roughly twice too big.
        panel = re.search(r"\n    \.player-panel \{\n(.*?)\n    \}", PAGE, re.S)
        self.assertIn("container-type: inline-size;", panel.group(1))
        self.assertIn("cqi", PAGE.split(".player-panel .section-heading h1")[1][:200])


class ChatScreenWiringTest(unittest.TestCase):
    def test_every_topic_the_screen_needs_is_handled(self) -> None:
        for topic in ("listening", "gate", "conversation"):
            with self.subTest(topic=topic):
                # Both on first connection and as it changes.
                self.assertIn(f"if (message.{topic})", APP)
                self.assertIn(f'type === "{topic}"', APP)

    def test_hearing_someone_outranks_anything_else(self) -> None:
        """While a person is mid sentence, nothing about the last one matters.

        Otherwise the device carries on showing that it is speaking while
        somebody is plainly talking over it.
        """

        pet = APP.split("function refreshPet()", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("hearing", pet)
        # Hearing is read first, and what it yields is the listening state.
        self.assertLess(pet.index("hearing"), pet.index("assistantState"))
        self.assertRegex(pet, r'hearing\s*\n?\s*\?\s*"listening"')

    def test_a_question_waiting_for_a_reply_shows_no_reply(self) -> None:
        # Otherwise a new question stands over the answer to the last one.
        body = APP.split("function applyConversation(", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("replyAnswersIt", body)
        self.assertIn("latestReply > latestHeard", body)

    def test_the_newest_question_is_not_shown_twice(self) -> None:
        body = APP.split("function applyConversation(", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("turns.slice(0, exchangeStart)", body)

    def test_asking_for_the_panel_out_loud_outranks_the_screen(self) -> None:
        body = APP.split("function refreshPlayerPanel()", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("panelAskedFor === null", body)
        # And a new search hands the decision back, so asking for it to go
        # away once does not mean it stays away for good.
        tracks = APP.split("function applyTracks(", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("panelAskedFor = null", tracks)

    def test_the_words_that_were_held_back_are_marked_on_screen(self) -> None:
        """The one place full vermillion belongs, and why it is reserved.

        These are the exact words the masker kept from the model, sitting in
        the sentence on a screen that is in the room. Anyone watching can see
        what was held and that the answer came back anyway.
        """

        self.assertIn(".revealed { color: var(--lift); }", PAGE)
        self.assertIn("function drawTurn(", APP)
        # Every turn on the screen is drawn the same way, so the marking is
        # not only on the newest line.
        drawn = APP.count("drawTurn(")
        self.assertGreaterEqual(drawn, 4)

    def test_a_sentence_somebody_said_can_never_become_markup(self) -> None:
        # The core sends offsets, not markup, and every piece goes in as
        # text. A transcript is whatever was said in the room.
        body = APP.split("function drawTurn(", maxsplit=1)[1].split("\n}")[0]
        self.assertNotIn("innerHTML", body)
        self.assertIn("textContent", body)
        self.assertIn("createElement", body)

    def test_a_span_that_does_not_fit_its_text_is_dropped(self) -> None:
        # Offsets arrive over a websocket. One that runs off the end, or
        # backwards, must not tear the sentence apart.
        body = APP.split("function drawTurn(", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("Number.isInteger(start)", body)
        self.assertIn("end > text.length", body)
        self.assertIn("end <= start", body)

    def test_a_closed_panel_cannot_swallow_a_tap(self) -> None:
        """It still overflows its zero width track by a few dozen pixels.

        Invisible is not gone: on a touchscreen that strip would quietly eat
        taps at the edge of the conversation.
        """

        panel = re.search(r"\n    \.player-panel \{\n(.*?)\n    \}", PAGE, re.S)
        self.assertIn("pointer-events: none;", panel.group(1))
        opened = PAGE.split('[data-player="open"] .player-panel {', maxsplit=1)[
            1
        ].split("}", maxsplit=1)[0]
        self.assertIn("pointer-events: auto;", opened)

    def test_the_panel_follows_results_rather_than_whether_audio_is_running(
        self,
    ) -> None:
        """A paused track is still the thing on the deck.

        Closing the panel underneath somebody who just pressed pause would
        be the screen arguing with them.
        """

        body = APP.split("function refreshPlayerPanel()", maxsplit=1)[1].split("\n}")[0]
        self.assertIn("tracks.length > 0", body)
        self.assertNotIn("playback.playing", body)
        # The recorder measures itself against its box, and the box changed.
        self.assertIn('new Event("resize")', body)


if __name__ == "__main__":
    unittest.main()
