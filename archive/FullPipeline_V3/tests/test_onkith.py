from __future__ import annotations

import unittest

from app.onkith_music import (
    append_song_to_search_query,
    intercept_onkith_command,
    safe_json_for_html,
)


class OnkithPreservationTests(unittest.TestCase):
    def test_play_command_and_remaining_text(self) -> None:
        result = intercept_onkith_command(
            "normal words Onkith play Imagine Dragons Bones"
        )
        self.assertTrue(result.command_detected)
        self.assertEqual(result.intent, "PLAY")
        self.assertEqual(result.command_text, "play Imagine Dragons Bones")
        self.assertEqual(result.remaining_text, "normal words")

    def test_close_command(self) -> None:
        result = intercept_onkith_command("On Kith stop the music")
        self.assertEqual(result.intent, "CLOSE")

    def test_common_name_is_not_wake_word(self) -> None:
        self.assertFalse(
            intercept_onkith_command("Someone named Ankit called").command_detected
        )

    def test_search_suffix_does_not_modify_transcript(self) -> None:
        self.assertEqual(append_song_to_search_query("play believer"), "believer song")

    def test_player_json_is_html_safe(self) -> None:
        encoded = safe_json_for_html({"value": "</script><script>alert(1)</script>"})
        self.assertNotIn("</script>", encoded)


if __name__ == "__main__":
    unittest.main()
