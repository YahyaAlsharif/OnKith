"""Onkith wake-word interception and isolated YouTube music ownership."""

from __future__ import annotations

import html
import json
import os
import queue
import re
import shutil
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
DEFAULT_REGION = "SA"
MAX_SEARCH_RESULTS = 25
MAX_PLAYER_RESULTS = 5
PLAYER_TEMPLATE_PATH = Path(__file__).resolve().with_name(
    "player_template.html"
)
PLAYER_ASSET_SOURCE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "assets" / "player"
)
PLAYER_ASSET_FILENAMES = (
    "archivo-medium-v1.ttf",
    "archivo-regular-v1.ttf",
    "archivo-semibold-v1.ttf",
    "ibm-plex-mono-v1.ttf",
    "onkith-favicon.svg",
    "player-app-v1.mjs",
    "player-spline-v1.mjs",
    # The recorder and the runtime that draws it. spline-viewer.js imports
    # the other two by relative path, so all four stay in one directory and
    # the device never asks a CDN for any of them.
    "spline/opentype.js",
    "spline/process.js",
    "spline/scene.splinecode",
    "spline/spline-viewer.js",
)
_THUMBNAIL_PRIORITY = ("maxres", "standard", "high", "medium", "default")

Intent = Literal[
    "PLAY",
    "CLOSE",
    "ASK",
    "SHOW",
    "HIDE",
    "SHOW_DEVICES",
    "HIDE_DEVICES",
    "LOUDER",
    "QUIETER",
    "LIGHT_ON",
    "LIGHT_OFF",
    "BRIGHTER",
    "DIMMER",
    "LIGHT_LEVEL",
]
LogFunction = Callable[[str], None]

_WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
@dataclass(frozen=True)
class WakeWords:
    """What this device answers to.

    ``names`` is what it is called. ``spellings`` are the ways a recogniser
    writes those names down, which is a different list: a speech model hears
    "on ket" and there is no rule that derives that from the letters.

    Near misses are allowed only for names long enough to survive one, which
    is the whole reason for ``fuzzy_minimum_length``. Onkith is six letters
    and can spare an edit. Sada cannot: at one edit it would answer to "soda"
    and to "sad", and a device that wakes on "sad" is worse than one that
    misses a call.
    """

    names: tuple[str, ...] = ("onkith", "sada", "sitr")
    spellings: tuple[str, ...] = (
        "onkuth",
        "onkoth",
        "onketh",
        "ankith",
        "unkith",
        "onkit",
        "onket",
        "صدى",
        "sitter",
        "setter",
        "ستر",
    )
    # Words that must never wake it however close they land. Ankit is a name
    # people have, and being called by it is not being spoken to.
    never: tuple[str, ...] = ("ankit",)
    fuzzy_minimum_length: int = 6
    edit_distance: int = 1

    @classmethod
    def from_config(cls, section: dict[str, Any] | None) -> "WakeWords":
        if not section:
            return cls()
        default = cls()

        def words(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
            supplied = section.get(key)
            if supplied is None:
                return fallback
            # Stripped as well as folded. A name typed with a stray space
            # would otherwise never match anything, because the words it is
            # compared against come out of a tokeniser and have none.
            return tuple(
                _normalise_word(str(item).strip())
                for item in supplied
                if str(item).strip()
            )

        return cls(
            names=words("names", default.names) or default.names,
            spellings=words("spellings", default.spellings),
            never=words("never", default.never),
            fuzzy_minimum_length=int(
                section.get("fuzzy_minimum_length", default.fuzzy_minimum_length)
            ),
            edit_distance=int(section.get("edit_distance", default.edit_distance)),
        )

    def spoken_forms(self) -> set[str]:
        return set(self.names) | set(self.spellings)


_WAKE = WakeWords()


def configure_wake_words(wake: WakeWords) -> None:
    """Set what the device answers to, once, at startup."""

    global _WAKE
    _WAKE = wake


def wake_words() -> WakeWords:
    return _WAKE
_CLOSE_WORDS = {
    "stop",
    "close",
    "pause",
    "skip",
    "cancel",
    "quit",
    "end",
    "prevent",
}
_LEADING_SEARCH_CONTROL_WORDS = {
    "play",
    "plays",
    "playing",
    "open",
    "opens",
    "opening",
    "do",
    "does",
    "start",
    "starts",
    "starting",
    "put",
    "listen",
    "find",
    "search",
    "launch",
}
_COMMAND_BOUNDARIES = {
    ("please", "continue"),
    ("continue", "now"),
}
# Asking for the player is asking about the screen, not about the music.
# The noun is what tells the two apart: "close it" stops the track, "close
# the player" puts the panel away and leaves the track playing.
_SCREEN_NOUNS = {"player", "panel", "screen", "recorder", "deck"}
# The second panel needs a noun of its own, because it shares every verb
# with the first one. "Devices" and "hardware" are the two words a person
# reaches for, and neither appears in _SCREEN_NOUNS, so "show the devices"
# and "show the player" cannot be confused for one another.
#
# Singular "device" is left out deliberately. This machine calls itself a
# device throughout, so "hide the device panel" would be ambiguous with
# every other panel on it, and "turn the device off" is not a request to
# close a window.
_DEVICE_NOUNS = {"devices", "hardware"}
_SHOW_VERBS = {
    "show",
    "shows",
    "open",
    "opens",
    "bring",
    "display",
    "reveal",
    "pop",
}
_HIDE_VERBS = {
    "close",
    "closes",
    "hide",
    "hides",
    "dismiss",
    "remove",
    "shut",
}
# Volume is the one control with no noun of its own worth insisting on.
# Nobody says "raise the volume of the music", they say "louder", so a
# single word has to be enough, and the two-word forms have to cover the
# ways a person reaches for it without thinking.
#
# "quiet" and "loud" are left out on purpose. "Onkith quiet" is somebody
# telling it to be quiet, which is the transport, not the level.
_LOUDER_WORDS = {"louder"}
_QUIETER_WORDS = {"quieter", "softer"}
_VOLUME_NOUNS = {"volume", "sound", "music", "it", "that", "this"}
_RAISE_VERBS = {"increase", "raise", "boost", "crank", "pump"}
_LOWER_VERBS = {"lower", "decrease", "reduce", "soften", "drop"}
_LEVEL_VERBS = {"turn", "put", "bring", "make", "set"}
# How far one instruction moves it, out of 100. Small enough that asking
# twice is normal, large enough that asking once is audible.
VOLUME_STEP = 12

# The lamp. Unlike volume this one insists on its noun, because "turn it
# up" in a room with music playing means the music, and a device that
# guesses the lamp there is a device that has to be argued with.
_LIGHT_NOUNS = {"light", "lights", "lamp", "led", "leds", "brightness"}
_BRIGHTER_WORDS = {"brighter", "brighten"}
_DIMMER_WORDS = {"dimmer", "dim"}
# "close the light" and "open the light" are how a great many people say
# this in English, including everyone who will be standing in front of it
# on the day. They cost nothing to accept and they are the first thing
# somebody tries.
_LIGHT_OFF_WORDS = {"off", "out"}
_LIGHT_ON_WORDS = {"on"}

# "Set the light to ten percent" arrives as a word about as often as it
# arrives as a numeral, and which one depends on the recogniser rather
# than on the person, so both are read. Only the round tens: nobody asks
# a lamp for thirty seven percent, and every extra word here is another
# way to mishear an ordinary sentence as a command.
_SPOKEN_LEVELS = {
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
    # What people say instead of a number.
    "half": 50, "full": 100, "max": 100, "maximum": 100, "minimum": 0,
}


@dataclass(frozen=True)
class OnkithInterception:
    command_detected: bool
    matched_wake_word: str | None
    intent: Intent | None
    command_text: str
    command_span_start: int | None
    command_span_end: int | None
    remaining_text: str
    # Only LIGHT_LEVEL carries one. Every other lane is a direction or a
    # switch, and a number would mean nothing to it.
    level: int | None = None


@dataclass(frozen=True)
class MusicAction:
    intent: Intent
    command_text: str
    generation: int


def _normalise_word(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


# Nothing here knows what the device is called. The previous matcher did:
# it required a candidate to start on, an or un and to end in h, which is the
# shape of the word Onkith and of no other word. Rename the device and it
# stopped answering. What is left is a rule about names in general, so the
# name itself is only ever data.
def _answers_to(candidate: str, wake: WakeWords, *, allow_near: bool = True) -> bool:
    if not candidate or candidate in wake.never:
        return False
    if candidate in wake.spoken_forms():
        return True
    if not allow_near:
        return False
    # A near miss is only allowed against a name it was given, never against
    # a spelling, so the accepted forms cannot drift outward a letter at a
    # time. And only for a name long enough to spare the edit.
    return any(
        len(name) >= wake.fuzzy_minimum_length
        and _edit_distance(candidate, name) <= wake.edit_distance
        for name in wake.names
    )


# How far off a word can be and still be worth reporting as a near miss.
# Wide enough to catch a recogniser writing the name as two ordinary words,
# which is what happens in practice, and it never wakes anything: this only
# ever produces a line in the log.
NEAR_MISS_LIMIT = 2
# A short word is close to everything. Without a floor and a length bound
# the report is noise: "it" is two edits from "sitr", so every sentence
# containing the word it would be offered as the device's name.
NEAR_MISS_MINIMUM_LENGTH = 4
NEAR_MISS_LENGTH_SPREAD = 2


def nearest_wake_miss(
    transcript: str, wake: WakeWords | None = None
) -> tuple[str, int] | None:
    """The closest this transcript came to being the device's name.

    A speech model writes a coined name however it likes, and it is
    consistent about it: this one hears Onkith and writes "on kiss". Without
    saying so, the device simply never answers and there is nothing on
    screen to explain why, which leaves somebody repeating themselves at it.
    Reported so the spelling can be added to the config, which is what that
    list is for.
    """

    settings = wake if wake is not None else _WAKE
    words = [_normalise_word(match.group(0)) for match in _WORD_RE.finditer(transcript)]
    candidates = list(words)
    candidates.extend(
        words[index] + words[index + 1] for index in range(len(words) - 1)
    )
    best: tuple[str, int] | None = None
    for candidate in candidates:
        if len(candidate) < NEAR_MISS_MINIMUM_LENGTH or candidate in settings.never:
            continue
        for name in settings.names:
            # Only against a name long enough to be distinctive, which is the
            # same rule the matcher itself uses. A four letter name sits two
            # edits from half the language: measured against Sitr, "is it"
            # and "stop" both come back as plausible, which is worse than
            # saying nothing.
            if len(name) < settings.fuzzy_minimum_length:
                continue
            if abs(len(candidate) - len(name)) > NEAR_MISS_LENGTH_SPREAD:
                continue
            distance = _edit_distance(candidate, name)
            if distance > NEAR_MISS_LIMIT:
                continue
            if best is None or distance < best[1]:
                best = (candidate, distance)
    return best


def _find_wake(
    words: list[str], wake: WakeWords | None = None
) -> tuple[int, int, str] | None:
    settings = wake if wake is not None else _WAKE
    for index, word in enumerate(words):
        if _answers_to(word, settings):
            return index, 1, word
        # A recogniser often writes one spoken name as two words, and there
        # is no rule that derives "on ket" from the letters of Onkith, so the
        # pair is joined and tried as though it had been heard whole. The
        # join has to be exact. Allowing an edit on top of it stacks two
        # guesses and starts hearing its own name in ordinary speech: with
        # the device called Jarvis, "the jar is empty" joins to "jaris",
        # which is one letter away.
        if index + 1 < len(words) and _answers_to(
            word + words[index + 1], settings, allow_near=False
        ):
            return index, 2, " ".join(words[index : index + 2])
    return None


# The first word after the wake word picks the lane, and anything that is not
# a verb this device knows is a question. The ordering matters more than it
# looks. A rule that guesses wrong sends a question to YouTube, which is what
# this used to do to "Onkith who was Alan Turing": it searched for "who was
# Alan Turing song". Falling through to the assistant is the failure that
# costs nothing, because the assistant can answer anything, including that it
# cannot do something.
#
# Only the first word is read. Scanning the whole command for a stop word
# would send "Onkith what time do the shops close" to the transport.
def _sounds_like_close(word: str) -> bool:
    return (
        word.startswith(("st", "cl", "pa", "sk", "ca", "qu"))
        and 3 <= len(word) <= 8
        and min(_edit_distance(word, close) for close in _CLOSE_WORDS) <= 1
    )


def _leads_with_close(words: list[str]) -> bool:
    lead = words[0]
    if lead in _CLOSE_WORDS or lead == "shut" or _sounds_like_close(lead):
        return True
    # "turn it off" and "switch the music off" carry the sense on a later
    # word, so those two verbs alone are allowed to look further along.
    return lead in {"turn", "switch"} and "off" in words[1:4]


# How far into the command a level can sit, and what marks it as one.
# "Set the light to ten percent" is the longest anybody says.
_LEVEL_WINDOW = 6
_LEVEL_MARKERS = {"to", "at", "percent", "%"}


def requested_level(words: list[str]) -> int | None:
    """A brightness out of 100, if these words are asking for one.

    Two guards, and both earn their place. The number has to be near the
    front, inside the command rather than anywhere in the sentence, and
    something has to mark it as a level: "to", "at" or "percent".
    Without them "turn the light on, I paid 20 dollars for it" sets the
    lamp to 20%.

    Strictly 0 to 100. Anything outside that is not a percentage somebody
    meant for a lamp, and reading "set the light to 2024" as a level is
    worse than not hearing it.

    The largest match wins rather than the first, because "one hundred"
    is two tokens and reads as 1 then 100, and a sentence carries one
    level.
    """

    window = words[:_LEVEL_WINDOW]
    if not _LEVEL_MARKERS.intersection(words[: _LEVEL_WINDOW + 1]):
        return None
    found: list[int] = []
    for word in window:
        if word.isdigit():
            value = int(word)
            if 0 <= value <= 100:
                found.append(value)
        elif word in _SPOKEN_LEVELS:
            found.append(_SPOKEN_LEVELS[word])
    return max(found) if found else None


def _light_command(words: list[str]) -> Intent | None:
    """The lamp, which has to be named before it is touched.

    Read before the volume and before the transport, because it shares
    every verb with both of them. "Turn it off" is the music and "turn
    the light off" is the lamp, and the only thing separating those is
    the noun, so the noun is required. The two comparatives are the
    exception: nothing else on this device gets brighter or dimmer.
    """

    lead = words[0]
    names_the_lamp = any(word in _LIGHT_NOUNS for word in words[:4])
    comparative = lead in _BRIGHTER_WORDS or (
        lead in _DIMMER_WORDS and not _mentions_screen(words)
    )

    # A number beats every direction. "Dim the light to 20" names a level
    # and reading it as one step down ignores what was asked for, so this
    # has to come before the comparatives rather than after them. "Off"
    # still wins: "turn the light off" with a stray number later in the
    # sentence is off.
    if (
        (names_the_lamp or comparative)
        and not _LIGHT_OFF_WORDS.intersection(words)
        and requested_level(words) is not None
    ):
        return "LIGHT_LEVEL"

    if lead in _BRIGHTER_WORDS:
        return "BRIGHTER"
    if lead in _DIMMER_WORDS and not _mentions_screen(words):
        return "DIMMER"
    if not names_the_lamp:
        return None
    if _LIGHT_OFF_WORDS.intersection(words) or lead in _CLOSE_WORDS:
        return "LIGHT_OFF"
    if _BRIGHTER_WORDS.intersection(words) or lead in _RAISE_VERBS:
        return "BRIGHTER"
    if _DIMMER_WORDS.intersection(words) or lead in _LOWER_VERBS:
        return "DIMMER"
    if "up" in words:
        return "BRIGHTER"
    if "down" in words:
        return "DIMMER"
    if _LIGHT_ON_WORDS.intersection(words) or lead in _SHOW_VERBS:
        return "LIGHT_ON"
    return None


def _mentions_screen(words: list[str]) -> bool:
    return any(word in _SCREEN_NOUNS for word in words[:4])


def _volume_command(words: list[str]) -> Intent | None:
    """Louder or quieter, however somebody reached for it.

    There is no natural noun to insist on the way there is for the
    screen. Nobody says "raise the volume of the music", they say
    "louder", so one word has to be enough, and the rest of it is the
    handful of ways a person asks without thinking about it.

    Read before the transport, because "turn it down" and "turn it off"
    share a verb and only the last word says which one was meant.
    """

    lead = words[0]
    # Three words is as far as the direction ever sits from the verb:
    # "turn the volume down" is the longest form anybody uses.
    rest = words[1:4]
    if lead in _LOUDER_WORDS:
        return "LOUDER"
    if lead in _QUIETER_WORDS:
        return "QUIETER"
    if lead == "volume":
        if "up" in rest:
            return "LOUDER"
        if "down" in rest:
            return "QUIETER"
        return None
    if lead in _RAISE_VERBS and any(word in _VOLUME_NOUNS for word in rest):
        return "LOUDER"
    if lead in _LOWER_VERBS and any(word in _VOLUME_NOUNS for word in rest):
        return "QUIETER"
    if lead in _LEVEL_VERBS:
        if "up" in rest or _LOUDER_WORDS.intersection(rest):
            return "LOUDER"
        if "down" in rest or _QUIETER_WORDS.intersection(rest):
            return "QUIETER"
    return None


def _lane_for(words: list[str]) -> Intent:
    if not words:
        # The wake word by itself is somebody getting the device's attention,
        # which is the assistant's to answer.
        return "ASK"
    # Before the player, because the two panels share every verb and
    # "show me the devices panel" holds a noun from each set. The more
    # specific noun has to be read first or that sentence opens the wrong
    # one.
    devices = _devices_command(words)
    if devices is not None:
        return devices
    # Before the transport, because the verbs overlap and only the noun
    # separates them. "Close it" stops the music; "close the player" puts the
    # panel away and leaves the music running.
    screen = _screen_command(words)
    if screen is not None:
        return screen
    # The lamp names itself, so it is safe to read first and it has to
    # be: it shares "turn", "off" and "lower" with both of the lanes
    # below, and it is the only one of the three with a noun to check.
    lamp = _light_command(words)
    if lamp is not None:
        return lamp
    level = _volume_command(words)
    if level is not None:
        return level
    if _leads_with_close(words):
        return "CLOSE"
    if words[0] in _LEADING_SEARCH_CONTROL_WORDS:
        return "PLAY"
    return "ASK"


def _screen_command(words: list[str]) -> Intent | None:
    """Show or hide the player panel, which is a screen and not a track.

    The noun has to be there. Without it "open believer" would put the panel
    up instead of playing anything, and "close it" would leave the music
    running while the panel disappeared.
    """

    if not any(word in _SCREEN_NOUNS for word in words[:4]):
        return None
    if words[0] in _SHOW_VERBS:
        return "SHOW"
    if words[0] in _HIDE_VERBS:
        return "HIDE"
    return None


def _devices_command(words: list[str]) -> Intent | None:
    """Show or hide the panel that says what is plugged into this board.

    Same shape as the player's, and same insistence on the noun. The verb
    alone cannot decide: "show" reaches three different things on this
    device and only the word after it says which.
    """

    if not any(word in _DEVICE_NOUNS for word in words[:4]):
        return None
    if words[0] in _SHOW_VERBS:
        return "SHOW_DEVICES"
    if words[0] in _HIDE_VERBS:
        return "HIDE_DEVICES"
    return None


def _is_close_intent(command_words: list[str]) -> bool:
    if not command_words:
        return False
    words = [_normalise_word(word) for word in command_words]
    if any(word in _CLOSE_WORDS for word in words):
        return True
    if "off" in words and any(
        word in {"turn", "switch", "music", "song", "it"}
        for word in words
    ):
        return True
    if "shut" in words:
        return True
    for word in words:
        if (
            word.startswith(("st", "cl", "pa", "sk", "ca", "qu"))
            and 3 <= len(word) <= 8
            and min(_edit_distance(word, close) for close in _CLOSE_WORDS)
            <= 1
        ):
            return True
    return False


def _command_word_count(words: list[str], command_start: int) -> int:
    available = min(5, len(words) - command_start)
    for offset in range(available):
        remaining = tuple(words[command_start + offset : command_start + offset + 2])
        if remaining in _COMMAND_BOUNDARIES:
            return offset
    return available


def _remove_command_span(
    transcript: str,
    start_character: int,
    end_character: int,
) -> str:
    prefix = transcript[:start_character].rstrip()
    suffix = transcript[end_character:].lstrip(" \t\r\n,;:.!?-")
    if prefix and suffix:
        return f"{prefix} {suffix}".strip()
    return (prefix or suffix).strip()


def append_song_to_search_query(command_text: str) -> str:
    """Build the search-only phrase without changing transcript output."""

    words = [
        match.group(0)
        for match in _WORD_RE.finditer(command_text)
    ][:5]
    if words and _normalise_word(words[0]) in _LEADING_SEARCH_CONTROL_WORDS:
        words = words[1:]
    if not words:
        return ""
    query = " ".join(words)
    if any(_normalise_word(word) == "song" for word in words):
        return query
    return f"{query} song"


def play_query_from(transcript: str) -> str:
    """Lift the title out of a sentence the device did not route itself.

    The deterministic lane never has to look for the verb: the wake word
    is directly in front of it, so the words after the name are the
    command and nothing else. This one has no such landmark. The model
    decided PLAY from a whole utterance, and anything at all may sit in
    front of the instruction.

    Without this, "how are you doing? Play dilemma." was handed over
    entire, the first five words became the search, and the device
    played a music quiz called "Most Popular Songs" instead of Nelly.

    The last verb wins rather than the first. A compound sentence puts
    the instruction at the end -- somebody talks, then asks for
    something -- so the later verb is the one that meant it.
    """

    matches = list(_WORD_RE.finditer(transcript))
    for match in reversed(matches):
        if _normalise_word(match.group(0)) not in _LEADING_SEARCH_CONTROL_WORDS:
            continue
        title = transcript[match.end() :].strip(" \t\r\n,;:.!?-")
        if title:
            return transcript[match.start() :].strip()
        # A verb with nothing behind it names no song. "Onkith, play"
        # falls through to whatever else was said rather than searching
        # for the empty string.
        break
    return without_wake_word(transcript).strip()


def intercept_onkith_command(transcript: str) -> OnkithInterception:
    """Decide which lane an utterance belongs in, and lift out a command.

    A music command is cut out of the sentence, because the words either side
    of it are ordinary speech that still belongs in the transcript. A question
    is not: it is the whole reason the person spoke, it is not five words
    long, and there is nothing to preserve around it. So an ASK leaves the
    transcript exactly as it was and only reports that the device was spoken
    to.
    """

    matches = list(_WORD_RE.finditer(transcript))
    normalised_words = [_normalise_word(match.group(0)) for match in matches]
    wake = _find_wake(normalised_words)
    if wake is None:
        return OnkithInterception(
            command_detected=False,
            matched_wake_word=None,
            intent=None,
            command_text="",
            command_span_start=None,
            command_span_end=None,
            remaining_text=transcript,
        )

    wake_start, wake_word_count, matched_wake_word = wake
    command_start = wake_start + wake_word_count

    lane = _lane_for(normalised_words[command_start:])
    if lane == "ASK":
        return OnkithInterception(
            command_detected=True,
            matched_wake_word=matched_wake_word,
            intent="ASK",
            command_text=" ".join(
                match.group(0) for match in matches[command_start:]
            ),
            command_span_start=command_start,
            command_span_end=len(matches),
            remaining_text=transcript,
        )

    command_word_count = _command_word_count(normalised_words, command_start)
    command_end = command_start + command_word_count
    command_words = [
        match.group(0) for match in matches[command_start:command_end]
    ]
    command_text = " ".join(command_words)
    intent: Intent = lane

    span_end_word = command_end - 1 if command_word_count else wake_start + wake_word_count - 1
    remaining_text = _remove_command_span(
        transcript,
        matches[wake_start].start(),
        matches[span_end_word].end(),
    )
    return OnkithInterception(
        command_detected=True,
        matched_wake_word=matched_wake_word,
        intent=intent,
        command_text=command_text,
        command_span_start=wake_start,
        command_span_end=command_end,
        remaining_text=remaining_text,
        # Read from the same words the lane was decided on, not from
        # command_text, which is capped at five words and would drop the
        # number out of "set the light to ten percent".
        level=(
            requested_level(normalised_words[command_start:])
            if intent == "LIGHT_LEVEL"
            else None
        ),
    )


def without_wake_word(transcript: str) -> str:
    """The question with the name that summoned it taken off the front.

    "Onkith, what is your purpose?" reads to a language model as somebody
    called Onkith asking a question, and it answers in kind: "my purpose
    is to assist users like you, Onkith." The name is how the device was
    addressed, not who was speaking, and the model has no way to know
    that from the text alone.

    Only the first occurrence goes, and only the word itself. A name that
    turns up later in a sentence is part of what was asked, and "what
    does Onkith mean" has to survive being asked.
    """

    matches = list(_WORD_RE.finditer(transcript))
    wake = _find_wake([_normalise_word(match.group(0)) for match in matches])
    if wake is None:
        return transcript
    wake_start, wake_word_count, _ = wake
    command_start = wake_start + wake_word_count
    before = transcript[: matches[wake_start].start()]
    after = (
        transcript[matches[command_start].start() :]
        if command_start < len(matches)
        else ""
    )
    # The comma the name was wearing goes with it. Left behind it opens
    # the sentence on punctuation, which reads as a fragment.
    question = f"{before.strip()} {after.strip()}".strip().lstrip(",.;: ")
    return question or transcript


def youtube_get(
    url: str,
    parameters: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    parameters = dict(parameters)
    parameters["key"] = api_key
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": "OnkithMusicPlayer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if api_key:
            body = body.replace(api_key, "[REDACTED]")
        raise RuntimeError(
            f"YouTube API returned HTTP {error.code}: {body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to YouTube: {error.reason}"
        ) from error


def search_youtube(
    query: str,
    api_key: str,
    region: str,
) -> list[dict[str, str]]:
    data = youtube_get(
        YOUTUBE_SEARCH_URL,
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoCategoryId": "10",
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
            "regionCode": region,
            "order": "relevance",
            "maxResults": str(MAX_SEARCH_RESULTS),
        },
        api_key,
    )
    results: list[dict[str, str]] = []
    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if video_id:
            results.append(
                {
                    "video_id": video_id,
                    "title": html.unescape(
                        snippet.get("title", "Unknown title")
                    ),
                    "channel": html.unescape(
                        snippet.get("channelTitle", "Unknown channel")
                    ),
                    "thumbnail": best_youtube_thumbnail(
                        snippet,
                        video_id,
                    ),
                }
            )
    return results


def best_youtube_thumbnail(
    snippet: dict[str, Any],
    video_id: str,
) -> str:
    thumbnails = snippet.get("thumbnails", {})
    if isinstance(thumbnails, dict):
        for size in _THUMBNAIL_PRIORITY:
            thumbnail = thumbnails.get(size)
            if not isinstance(thumbnail, dict):
                continue
            url = thumbnail.get("url")
            if isinstance(url, str) and url:
                return url
    safe_video_id = urllib.parse.quote(video_id, safe="")
    return f"https://i.ytimg.com/vi/{safe_video_id}/hqdefault.jpg"


def is_available_in_region(
    content_details: dict[str, Any],
    region: str,
) -> bool:
    restriction = content_details.get("regionRestriction", {})
    allowed_regions = restriction.get("allowed")
    blocked_regions = restriction.get("blocked")
    if isinstance(allowed_regions, list) and region not in allowed_regions:
        return False
    if isinstance(blocked_regions, list) and region in blocked_regions:
        return False
    return True


def verify_results(
    search_results: list[dict[str, str]],
    api_key: str,
    region: str,
) -> list[dict[str, str]]:
    if not search_results:
        return []
    data = youtube_get(
        YOUTUBE_VIDEOS_URL,
        {
            "part": "snippet,status,contentDetails",
            "id": ",".join(item["video_id"] for item in search_results),
        },
        api_key,
    )
    videos_by_id = {
        item["id"]: item
        for item in data.get("items", [])
        if item.get("id")
    }
    verified: list[dict[str, str]] = []
    for search_result in search_results:
        item = videos_by_id.get(search_result["video_id"])
        if item is None:
            continue
        status = item.get("status", {})
        content_details = item.get("contentDetails", {})
        if status.get("privacyStatus") != "public":
            continue
        if status.get("uploadStatus") != "processed":
            continue
        if status.get("embeddable") is not True:
            continue
        if not is_available_in_region(content_details, region):
            continue
        snippet = item.get("snippet", {})
        verified_result = dict(search_result)
        verified_result["title"] = html.unescape(
            snippet.get("title", search_result["title"])
        )
        verified_result["channel"] = html.unescape(
            snippet.get("channelTitle", search_result["channel"])
        )
        verified_result["thumbnail"] = best_youtube_thumbnail(
            snippet,
            search_result["video_id"],
        )
        verified.append(verified_result)
        if len(verified) >= MAX_PLAYER_RESULTS:
            break
    return verified


def normalize_videos(videos: list[dict[str, str]]) -> list[dict[str, str]]:
    """Guarantee every verified result carries a usable thumbnail."""

    normalized: list[dict[str, str]] = []
    for video in videos:
        item = dict(video)
        item["thumbnail"] = (
            video.get("thumbnail")
            or best_youtube_thumbnail({}, video["video_id"])
        )
        normalized.append(item)
    return normalized


def create_player_page(output_directory: Path) -> Path:
    """Install the player assets and write the static surface page.

    The page no longer carries baked-in results. Track data, playback
    state, and output levels arrive over the websocket instead, so one
    file serves every search, the surface can be reloaded at any moment,
    and nothing about audio depends on the page existing.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    output_assets_directory = output_directory / "assets"
    output_assets_directory.mkdir(parents=True, exist_ok=True)
    for filename in PLAYER_ASSET_FILENAMES:
        source = PLAYER_ASSET_SOURCE_DIRECTORY / filename
        if not source.is_file():
            raise FileNotFoundError(f"Player asset was not found: {source}")
        destination = output_assets_directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    page_path = output_directory / "player.html"
    page_path.write_text(
        PLAYER_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return page_path


class OnkithMusicController:
    """Own the API configuration and hand verified results to playback.

    This object no longer owns a browser. Audio belongs to the core
    playback engine, so a PLAY command replaces the queue in a process
    that keeps running whether or not any screen is attached.
    """

    def __init__(
        self,
        api_key: str,
        region: str = DEFAULT_REGION,
        *,
        playback: Any,
        log: LogFunction = lambda message: None,
        search_function: Callable[
            [str, str, str], list[dict[str, str]]
        ] = search_youtube,
        verify_function: Callable[
            [list[dict[str, str]], str, str], list[dict[str, str]]
        ] = verify_results,
    ) -> None:
        self._api_key = api_key
        self.region = region.upper()
        self.playback = playback
        self.log = log
        self.search_function = search_function
        self.verify_function = verify_function
        self.lock = threading.Lock()
        self._shutdown = False

    @classmethod
    def from_environment(
        cls,
        *,
        playback: Any,
        log: LogFunction = lambda message: None,
    ) -> "OnkithMusicController":
        try:
            api_key = os.environ["YOUTUBE_API_KEY"]
        except KeyError:
            api_key = ""
        return cls(
            api_key=api_key,
            region=os.environ.get("YOUTUBE_REGION", DEFAULT_REGION),
            playback=playback,
            log=log,
        )

    def play(
        self,
        query: str,
        *,
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> None:
        query = query.strip()
        if not query:
            raise ValueError("The Onkith music query is empty")
        if not self._api_key:
            raise RuntimeError(
                "YOUTUBE_API_KEY is not set in the process environment"
            )
        search_query = append_song_to_search_query(query)
        if not search_query:
            raise ValueError(
                "The Onkith command has no searchable words after its "
                "leading control word"
            )
        self.log(f'Searching YouTube for: "{search_query}"')
        search_results = self.search_function(
            search_query, self._api_key, self.region
        )
        verified_videos = normalize_videos(
            self.verify_function(search_results, self._api_key, self.region)
        )
        if should_cancel():
            self.log("A newer music action superseded this PLAY request")
            return
        if not verified_videos:
            raise RuntimeError("No playable YouTube result was found")
        with self.lock:
            if self._shutdown or should_cancel():
                return
            self.log(f"Opening: {verified_videos[0]['title']}")
            self.playback.load(verified_videos)

    def close(self) -> None:
        with self.lock:
            if self._shutdown:
                return
        self.playback.stop()
        self.log("Music stopped")

    def shutdown(self) -> None:
        """Stop accepting music actions.

        The playback process itself belongs to the core runtime, which
        outlives this controller, so it is not torn down here.
        """

        with self.lock:
            self._shutdown = True


class OnkithMusicWorker:
    """Serialise music actions away from the STT worker."""

    def __init__(
        self,
        controller: OnkithMusicController,
        *,
        log: LogFunction = lambda message: None,
    ) -> None:
        self.controller = controller
        self.log = log
        self.queue: queue.Queue[MusicAction | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.errors: list[str] = []
        self._generation = 0
        self._generation_lock = threading.Lock()
        self._shutdown = False

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._run,
            name="onkith-music-worker",
            daemon=False,
        )
        self.thread.start()

    def _next_generation(self) -> int:
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def _is_stale(self, generation: int) -> bool:
        with self._generation_lock:
            return generation != self._generation

    def submit_play(self, query: str) -> None:
        generation = self._next_generation()
        self.queue.put(MusicAction("PLAY", query, generation))

    def submit_close(self) -> None:
        generation = self._next_generation()
        self.queue.put(MusicAction("CLOSE", "", generation))

    def _run(self) -> None:
        while True:
            action = self.queue.get()
            try:
                if action is None:
                    return
                if action.intent == "PLAY":
                    if not self._is_stale(action.generation):
                        self.controller.play(
                            action.command_text,
                            should_cancel=lambda: self._is_stale(
                                action.generation
                            ),
                        )
                else:
                    self.controller.close()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self.errors.append(message)
                self.log(f"Music action failed: {message}")
            finally:
                self.queue.task_done()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        if self.thread is None:
            self.controller.shutdown()
            self._shutdown = True
            return
        self._next_generation()
        self.queue.join()
        self.queue.put(None)
        if self.thread is not None:
            self.thread.join()
        self.controller.shutdown()
        self._shutdown = True
