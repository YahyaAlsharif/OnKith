"""Onkith wake-word interception and isolated YouTube music ownership."""

from __future__ import annotations

import html
import http.client
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Literal


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
# Use the literal loopback address for both the local server and browser URL.
# On affected Windows hosts, Chrome resolves ``localhost`` to ::1 while
# ThreadingHTTPServer is listening on IPv4, producing a full-page connection
# reset before the embedded YouTube player is ever reached.
HOST = "127.0.0.1"
DEFAULT_REGION = "SA"
MAX_SEARCH_RESULTS = 10
MAX_PLAYER_RESULTS = 5

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
_SINGLE_WAKE_ALIASES = {
    "onkith",
    "onkuth",
    "onkoth",
    "onketh",
    "ankith",
    "unkith",
}
_TWO_TOKEN_WAKE_ALIASES = {
    ("on", "kith"),
    ("on", "kit"),
    ("on", "ket"),
    ("on", "kuth"),
}
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
_SCREEN_NOUNS = {"player", "panel", "screen", "recorder", "deck"}
_DEVICE_NOUNS = {"devices", "hardware"}
_SHOW_VERBS = {"show", "shows", "open", "opens", "bring", "display", "reveal", "pop"}
_HIDE_VERBS = {"close", "closes", "hide", "hides", "dismiss", "remove", "shut"}
_LOUDER_WORDS = {"louder"}
_QUIETER_WORDS = {"quieter", "softer"}
_VOLUME_NOUNS = {"volume", "sound", "music", "it", "that", "this"}
_RAISE_VERBS = {"increase", "raise", "boost", "crank", "pump"}
_LOWER_VERBS = {"lower", "decrease", "reduce", "soften", "drop"}
_LEVEL_VERBS = {"turn", "put", "bring", "make", "set"}
_LIGHT_NOUNS = {"light", "lights", "lamp", "led", "leds", "brightness"}
_BRIGHTER_WORDS = {"brighter", "brighten"}
_DIMMER_WORDS = {"dimmer", "dim"}
_LIGHT_OFF_WORDS = {"off", "out"}
_LIGHT_ON_WORDS = {"on"}
_SPOKEN_LEVELS = {
    "zero": 0,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "half": 50,
    "full": 100,
    "max": 100,
    "maximum": 100,
    "minimum": 0,
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


def _is_fuzzy_single_wake(candidate: str) -> bool:
    # "Ankit" is a common name and is deliberately not a wake alias.
    if candidate == "ankit":
        return False
    if not candidate.startswith(("on", "an", "un")):
        return False
    if not candidate.endswith("h") or not 5 <= len(candidate) <= 7:
        return False
    return min(
        _edit_distance(candidate, alias)
        for alias in _SINGLE_WAKE_ALIASES
    ) <= 1


def _find_wake(words: list[str]) -> tuple[int, int, str] | None:
    for index, word in enumerate(words):
        if word in _SINGLE_WAKE_ALIASES or _is_fuzzy_single_wake(word):
            return index, 1, word
        if index + 1 >= len(words):
            continue
        pair = (word, words[index + 1])
        if pair in _TWO_TOKEN_WAKE_ALIASES:
            return index, 2, " ".join(pair)
        if (
            word == "on"
            and words[index + 1].startswith("k")
            and 3 <= len(words[index + 1]) <= 5
            and min(
                _edit_distance(words[index + 1], suffix)
                for suffix in {"kith", "kit", "ket", "kuth"}
            )
            <= 1
        ):
            return index, 2, " ".join(pair)
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


def _leads_with_close(words: list[str]) -> bool:
    """Require a close instruction at the front, not inside a question."""

    if not words:
        return False
    lead = words[0]
    if lead in _CLOSE_WORDS:
        return True
    if (
        lead.startswith(("st", "cl", "pa", "sk", "ca", "qu"))
        and 3 <= len(lead) <= 8
        and min(_edit_distance(lead, close) for close in _CLOSE_WORDS) <= 1
    ):
        return True
    return lead in {"turn", "switch", "shut"} and "off" in words[1:4]


def requested_level(words: list[str]) -> int | None:
    """Read a local light percentage from nearby numeral or spoken words."""

    normalised = [_normalise_word(str(word)).strip(".,;:!?%") for word in words]
    for word in normalised[:8]:
        if word in _SPOKEN_LEVELS:
            return _SPOKEN_LEVELS[word]
        if word.isdigit():
            value = int(word)
            if 0 <= value <= 100:
                return value
    return None


def _mentions_screen(words: list[str]) -> bool:
    return any(word in _SCREEN_NOUNS for word in words[:4])


def _devices_command(words: list[str]) -> Intent | None:
    if not any(word in _DEVICE_NOUNS for word in words[:4]):
        return None
    if words[0] in _SHOW_VERBS:
        return "SHOW_DEVICES"
    if words[0] in _HIDE_VERBS:
        return "HIDE_DEVICES"
    return None


def _screen_command(words: list[str]) -> Intent | None:
    if not any(word in _SCREEN_NOUNS for word in words[:4]):
        return None
    if words[0] in _SHOW_VERBS:
        return "SHOW"
    if words[0] in _HIDE_VERBS:
        return "HIDE"
    return None


def _light_command(words: list[str]) -> Intent | None:
    if not words:
        return None
    lead = words[0]
    names_light = any(word in _LIGHT_NOUNS for word in words[:5])
    comparative = lead in _BRIGHTER_WORDS or (
        lead in _DIMMER_WORDS and not _mentions_screen(words)
    )
    if (
        (names_light or comparative)
        and not _LIGHT_OFF_WORDS.intersection(words)
        and requested_level(words) is not None
    ):
        return "LIGHT_LEVEL"
    if lead in _BRIGHTER_WORDS:
        return "BRIGHTER"
    if lead in _DIMMER_WORDS and not _mentions_screen(words):
        return "DIMMER"
    if not names_light:
        return None
    if _LIGHT_OFF_WORDS.intersection(words) or lead in _CLOSE_WORDS:
        return "LIGHT_OFF"
    if _BRIGHTER_WORDS.intersection(words) or lead in _RAISE_VERBS or "up" in words:
        return "BRIGHTER"
    if _DIMMER_WORDS.intersection(words) or lead in _LOWER_VERBS or "down" in words:
        return "DIMMER"
    if _LIGHT_ON_WORDS.intersection(words) or lead in _SHOW_VERBS:
        return "LIGHT_ON"
    return None


def _volume_command(words: list[str]) -> Intent | None:
    if not words:
        return None
    lead = words[0]
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
    """Choose a deterministic local action or the masked assistant lane."""

    if not words:
        return "ASK"
    for parser in (_devices_command, _screen_command, _light_command, _volume_command):
        intent = parser(words)
        if intent is not None:
            return intent
    if _leads_with_close(words):
        return "CLOSE"
    # Do/does are deliberately excluded here: "do you know" is a question,
    # not a request to search YouTube.
    if words[0] in _LEADING_SEARCH_CONTROL_WORDS - {"do", "does"}:
        return "PLAY"
    return "ASK"


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
    suffix = transcript[end_character:].lstrip(" \t\r\n,;:.!?-–—")
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


def without_wake_word(transcript: str) -> str:
    """Remove only the first wake word from a question copy."""

    matches = list(_WORD_RE.finditer(transcript))
    wake = _find_wake([_normalise_word(match.group(0)) for match in matches])
    if wake is None:
        return transcript
    wake_start, wake_word_count, _ = wake
    after_index = wake_start + wake_word_count
    before = transcript[: matches[wake_start].start()]
    after = (
        transcript[matches[after_index].start() :]
        if after_index < len(matches)
        else ""
    )
    return f"{before.strip()} {after.strip()}".strip().lstrip(",.;: ") or transcript


def play_query_from(transcript: str) -> str:
    """Extract the final explicit play/search clause from local raw text."""

    matches = list(_WORD_RE.finditer(transcript))
    play_words = _LEADING_SEARCH_CONTROL_WORDS - {"do", "does"}
    for match in reversed(matches):
        if _normalise_word(match.group(0)) not in play_words:
            continue
        title = transcript[match.end() :].strip(" \t\r\n,;:.!?-")
        if title:
            return transcript[match.start() :].strip()
        break
    return without_wake_word(transcript).strip()


def route_onkith_request(transcript: str) -> OnkithInterception:
    """Classify addressed speech while preserving questions for masking."""

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
        level=(
            requested_level(normalised_words[command_start:])
            if intent == "LIGHT_LEVEL"
            else None
        ),
    )


def intercept_onkith_command(transcript: str) -> OnkithInterception:
    """Preserve the original pre-mask seam: intercept music commands only.

    Questions, light/device commands and every other addressed utterance are
    reported as ordinary speech here so the unchanged STT worker forwards the
    complete transcript into the existing fail-closed masking pipeline. The
    additive post-mask layer calls :func:`route_onkith_request` instead.
    """

    routed = route_onkith_request(transcript)
    if routed.command_detected and routed.intent in {"PLAY", "CLOSE"}:
        return routed
    return OnkithInterception(
        command_detected=False,
        matched_wake_word=None,
        intent=None,
        command_text="",
        command_span_start=None,
        command_span_end=None,
        remaining_text=transcript,
    )


def _create_ipv4_connection(
    address: tuple[str, int],
    timeout: float | object = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Connect to one of a host's IPv4 addresses without changing global DNS."""

    host, port = address
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
        host,
        port,
        socket.AF_INET,
        socket.SOCK_STREAM,
    ):
        connection: socket.socket | None = None
        try:
            connection = socket.socket(family, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                connection.settimeout(timeout)
            if source_address:
                connection.bind(source_address)
            connection.connect(sockaddr)
            return connection
        except OSError as error:
            last_error = error
            if connection is not None:
                connection.close()
    if last_error is not None:
        raise last_error
    raise OSError(f"No IPv4 address is available for {host}")


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP resolver is restricted to IPv4."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _create_ipv4_connection


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request: urllib.request.Request):
        return self.do_open(
            _IPv4HTTPSConnection,
            request,
            context=self._context,
            check_hostname=self._check_hostname,
        )


def _urlopen_ipv4(
    request: urllib.request.Request,
    *,
    timeout: float,
):
    opener = urllib.request.build_opener(_IPv4HTTPSHandler())
    return opener.open(request, timeout=timeout)


def _youtube_http_failure(
    error: urllib.error.HTTPError,
    api_key: str,
) -> RuntimeError:
    body = error.read().decode("utf-8", errors="replace")
    if api_key:
        body = body.replace(api_key, "[REDACTED]")
    return RuntimeError(f"YouTube API returned HTTP {error.code}: {body}")


def _connection_reason(error: BaseException) -> object:
    return error.reason if isinstance(error, urllib.error.URLError) else error


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
        raise _youtube_http_failure(error, api_key) from error
    except (urllib.error.URLError, OSError) as first_error:
        # Some Windows/router combinations advertise Google IPv6 routes that
        # accept TCP and then reset TLS. Retry this idempotent GET over IPv4
        # without changing DNS, the OS network stack, or any other pipeline.
        try:
            with _urlopen_ipv4(request, timeout=20) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise _youtube_http_failure(error, api_key) from error
        except (urllib.error.URLError, OSError) as fallback_error:
            raise RuntimeError(
                "Could not connect to YouTube after IPv4 fallback: "
                f"{_connection_reason(fallback_error)}"
            ) from fallback_error


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
                }
            )
    return results


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
        verified.append(search_result)
        if len(verified) >= MAX_PLAYER_RESULTS:
            break
    return verified


def safe_json_for_html(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def create_player_page(
    videos: list[dict[str, str]],
    output_directory: Path,
    port: int,
) -> Path:
    if not videos:
        raise ValueError("No videos were provided to the player")
    output_directory.mkdir(parents=True, exist_ok=True)
    origin = f"http://{HOST}:{port}"
    first_video_id = urllib.parse.quote(videos[0]["video_id"], safe="")
    encoded_origin = urllib.parse.quote(origin, safe="")
    videos_json = safe_json_for_html(videos)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="referrer" content="strict-origin-when-cross-origin">
  <title>Onkith YouTube Player</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #111; color: white; font-family: Arial, sans-serif; }}
    main {{ width: min(760px, 100%); margin: auto; padding: 18px; }}
    #channel, #status {{ color: #ccc; }}
    #player {{ width: 100%; aspect-ratio: 16/9; border: 0; background: #222; }}
    button {{ margin: 12px 8px 0 0; padding: 10px 16px; border: 1px solid #777;
      border-radius: 8px; background: #222; color: white; cursor: pointer; }}
  </style>
</head>
<body>
<main>
  <h1 id="title">Loading…</h1>
  <p id="channel"></p>
  <iframe id="player"
    src="https://www.youtube.com/embed/{first_video_id}?enablejsapi=1&autoplay=1&playsinline=1&origin={encoded_origin}"
    allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>
  <p id="status" aria-live="polite">Connecting to YouTube…</p>
  <button id="play-button" type="button">Play</button>
  <button id="next-button" type="button">Try next result</button>
</main>
<script>
const videos = {videos_json};
let player = null, currentIndex = 0, playbackStarted = false, watchdog = null;
const titleElement = document.getElementById("title");
const channelElement = document.getElementById("channel");
const statusElement = document.getElementById("status");
function setStatus(message) {{ statusElement.textContent = message; }}
function showCurrentVideo() {{
  const video = videos[currentIndex];
  titleElement.textContent = video.title;
  channelElement.textContent = video.channel;
  document.title = video.title;
}}
function clearWatchdog() {{
  if (watchdog !== null) {{ window.clearTimeout(watchdog); watchdog = null; }}
}}
function startWatchdog() {{
  clearWatchdog();
  watchdog = window.setTimeout(() => {{
    if (!playbackStarted) tryNextVideo("Playback did not start.");
  }}, 12000);
}}
function loadCurrentVideo() {{
  playbackStarted = false;
  showCurrentVideo();
  setStatus(`Loading result ${{currentIndex + 1}} of ${{videos.length}}…`);
  startWatchdog();
  if (player) player.loadVideoById(videos[currentIndex].video_id);
}}
function tryNextVideo(reason) {{
  clearWatchdog();
  if (currentIndex + 1 >= videos.length) {{
    setStatus(reason + " No remaining result could be played.");
    return;
  }}
  currentIndex += 1;
  setStatus(reason + " Trying the next result…");
  window.setTimeout(loadCurrentVideo, 600);
}}
window.onYouTubeIframeAPIReady = function () {{
  showCurrentVideo();
  player = new YT.Player("player", {{ events: {{
    onReady(event) {{ setStatus("Starting playback…"); startWatchdog(); event.target.playVideo(); }},
    onStateChange(event) {{
      if (event.data === YT.PlayerState.PLAYING) {{
        playbackStarted = true; clearWatchdog(); setStatus("Playing through your speakers.");
      }} else if (event.data === YT.PlayerState.BUFFERING) setStatus("Buffering…");
      else if (event.data === YT.PlayerState.PAUSED) setStatus("Paused.");
      else if (event.data === YT.PlayerState.ENDED) {{ clearWatchdog(); setStatus("Playback finished."); }}
    }},
    onError(event) {{ tryNextVideo(`YouTube player error ${{event.data}}.`); }},
    onAutoplayBlocked() {{ clearWatchdog(); setStatus("Autoplay was blocked. Click Play once."); }}
  }} }});
}};
document.getElementById("play-button").addEventListener("click", () => {{
  if (!player) {{ setStatus("The player is still loading."); return; }}
  playbackStarted = false; startWatchdog(); player.playVideo();
}});
document.getElementById("next-button").addEventListener("click", () => tryNextVideo("Skipped."));
</script>
<script src="https://www.youtube.com/iframe_api"></script>
</body>
</html>"""
    page_path = output_directory / "player.html"
    page_path.write_text(page, encoding="utf-8")
    return page_path


class PlayerHandler(SimpleHTTPRequestHandler):
    def log_message(self, format_string: str, *args: object) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()


def start_server(
    directory: Path,
) -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    def handler(*args: Any, **kwargs: Any) -> PlayerHandler:
        return PlayerHandler(*args, directory=str(directory), **kwargs)

    server = ThreadingHTTPServer((HOST, 0), handler)
    server.daemon_threads = True
    port = int(server.server_address[1])
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="onkith-player-server",
        daemon=False,
    )
    server_thread.start()
    return server, server_thread, port


def find_browser() -> str | None:
    for command in (
        "chrome",
        "google-chrome",
        "msedge",
        "chromium",
        "chromium-browser",
    ):
        executable = shutil.which(command)
        if executable:
            return executable
    if os.name == "nt":
        roots = [
            os.environ.get("PROGRAMFILES", ""),
            os.environ.get("PROGRAMFILES(X86)", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        relative_paths = [
            "Google/Chrome/Application/chrome.exe",
            "Microsoft/Edge/Application/msedge.exe",
        ]
        for root in roots:
            if not root:
                continue
            for relative_path in relative_paths:
                candidate = Path(root) / relative_path
                if candidate.is_file():
                    return str(candidate)
    return None


def open_player(
    url: str,
    profile_directory: Path,
) -> subprocess.Popen[Any]:
    browser = find_browser()
    if browser is None:
        raise RuntimeError(
            "Chrome or Edge was not found; a dedicated player cannot be opened"
        )
    profile_directory.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            browser,
            f"--app={url}",
            "--window-size=800,650",
            "--window-position=80,80",
            "--autoplay-policy=no-user-gesture-required",
            f"--user-data-dir={profile_directory}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
    )


def close_player_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


class OnkithMusicController:
    """Own the API configuration, local player server, and one browser."""

    def __init__(
        self,
        api_key: str,
        region: str = DEFAULT_REGION,
        *,
        log: LogFunction = lambda message: None,
        search_function: Callable[
            [str, str, str], list[dict[str, str]]
        ] = search_youtube,
        verify_function: Callable[
            [list[dict[str, str]], str, str], list[dict[str, str]]
        ] = verify_results,
        page_factory: Callable[
            [list[dict[str, str]], Path, int], Path
        ] = create_player_page,
        server_factory: Callable[
            [Path], tuple[Any, threading.Thread, int]
        ] = start_server,
        player_opener: Callable[
            [str, Path], subprocess.Popen[Any]
        ] = open_player,
        player_closer: Callable[
            [subprocess.Popen[Any] | None], None
        ] = close_player_process,
    ) -> None:
        self._api_key = api_key
        self.region = region.upper()
        self.log = log
        self.search_function = search_function
        self.verify_function = verify_function
        self.page_factory = page_factory
        self.server_factory = server_factory
        self.player_opener = player_opener
        self.player_closer = player_closer
        self.lock = threading.Lock()
        self.browser_process: subprocess.Popen[Any] | None = None
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="onkith-player-"
        )
        self.player_directory = Path(self._temporary_directory.name)
        self.profile_directory = self.player_directory / "browser-profile"
        self.server: Any | None = None
        self.server_thread: threading.Thread | None = None
        self.port: int | None = None
        self._shutdown = False

    @classmethod
    def from_environment(
        cls,
        *,
        log: LogFunction = lambda message: None,
    ) -> "OnkithMusicController":
        try:
            api_key = os.environ["YOUTUBE_API_KEY"]
        except KeyError:
            api_key = ""
        return cls(
            api_key=api_key,
            region=os.environ.get("YOUTUBE_REGION", DEFAULT_REGION),
            log=log,
        )

    def _ensure_server(self) -> None:
        if self.server is not None:
            return
        self.server, self.server_thread, self.port = self.server_factory(
            self.player_directory
        )

    def play(
        self,
        query: str,
        *,
        should_cancel: Callable[[], bool] = lambda: False,
    ) -> None:
        query = query.strip()
        self.close()
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
        verified_videos = self.verify_function(
            search_results, self._api_key, self.region
        )
        if should_cancel():
            self.log("A newer music action superseded this PLAY request")
            return
        if not verified_videos:
            raise RuntimeError("No playable YouTube result was found")
        with self.lock:
            if self._shutdown or should_cancel():
                return
            self._ensure_server()
            assert self.port is not None
            self.page_factory(
                verified_videos, self.player_directory, self.port
            )
            player_url = (
                f"http://{HOST}:{self.port}/player.html"
                f"?request={time.time_ns()}"
            )
            self.log(f"Opening: {verified_videos[0]['title']}")
            self.browser_process = self.player_opener(
                player_url, self.profile_directory
            )

    def close(self) -> None:
        with self.lock:
            process = self.browser_process
            self.browser_process = None
        self.player_closer(process)
        if process is not None:
            self.log("Player closed")

    def shutdown(self) -> None:
        with self.lock:
            if self._shutdown:
                return
            self._shutdown = True
        self.close()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread is not None:
            self.server_thread.join()
        self._temporary_directory.cleanup()


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
