"""Headless audio playback owned by the core process.

Playback runs in an ``mpv`` child process driven over its JSON IPC
socket, with ``yt-dlp`` resolving the YouTube stream. Nothing about
playback depends on a page being open: the display surface is a
subscriber, never the audio source. That is what lets the terminal mode
play music with no UI at all, lets Chromium be killed without
interrupting audio, and lets the core duck music while speech is
playing.

Only the audio stream is requested, so a Raspberry Pi never decodes
video for a track nobody is watching.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

DUCK_RATIO = 0.2
POSITION_PUBLISH_INTERVAL = 0.25
_CONNECT_TIMEOUT = 15.0
_REPLY_TIMEOUT = 5.0

_OBSERVED_PROPERTIES = (
    "pause",
    "time-pos",
    "duration",
    "volume",
    "mute",
    "playlist-pos",
    # paused-for-cache, not core-idle: core-idle is true whenever mpv is not
    # rendering a frame, which for audio-only playback is most of the time,
    # so it reported a track that was playing perfectly well as buffering.
    "paused-for-cache",
    "idle-active",
)

LogFunction = Callable[[str], None]


class PlaybackUnavailable(RuntimeError):
    """mpv or yt-dlp is missing, so headless playback cannot start."""


def find_executable(name: str, override: str | None = None) -> str | None:
    """Resolve a helper, including one installed beside this interpreter.

    The launcher runs .venv/bin/python directly rather than activating the
    environment, so .venv/bin is not on PATH. yt-dlp is installed there
    because it has to track YouTube changes and cannot be pinned.
    """

    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return shutil.which(override)
    beside_interpreter = Path(sys.executable).parent / name
    if beside_interpreter.is_file() and os.access(beside_interpreter, os.X_OK):
        return str(beside_interpreter)
    return shutil.which(name)


def describe_requirements(
    *, mpv_path: str | None = None, ytdl_path: str | None = None
) -> dict[str, str | None]:
    """Report which playback executables resolve on this machine."""

    return {
        "mpv": find_executable("mpv", mpv_path),
        "yt-dlp": find_executable("yt-dlp", ytdl_path),
    }


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


class MpvPlaybackEngine:
    """Own one mpv child process and translate commands to its IPC socket."""

    def __init__(
        self,
        bus: Any,
        *,
        mpv_path: str | None = None,
        ytdl_path: str | None = None,
        log: LogFunction = lambda message: None,
        initial_volume: int = 82,
        extra_arguments: tuple[str, ...] = (),
    ) -> None:
        self.bus = bus
        self.log = log
        self._mpv_path = mpv_path
        self._ytdl_path = ytdl_path
        self._extra_arguments = tuple(extra_arguments)
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._request_id = 0
        self._pending: dict[int, dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._replies: dict[int, threading.Event] = {}
        self._shutdown = threading.Event()
        self._user_volume = max(0, min(100, int(initial_volume)))
        self._ducked = False
        self._muted = False
        # Transport is reported from three independent mpv properties, so
        # they are held here and turned into one coherent state.
        self._paused = False
        self._stalled = False
        self._idle = True
        self._duration = 0.0
        self._position = 0.0
        self._last_position_publish = 0.0
        self._tracks: list[dict[str, str]] = []
        self._index = 0

    # ---------------------------------------------------------------- setup

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.running:
            return
        mpv = find_executable("mpv", self._mpv_path)
        if mpv is None:
            install = (
                "brew install mpv"
                if sys.platform == "darwin"
                else "sudo apt-get install mpv"
            )
            raise PlaybackUnavailable(
                "mpv was not found, so music cannot play and the transport "
                f"controls stay disabled. Install it with: {install}"
            )
        if not hasattr(socket, "AF_UNIX"):
            raise PlaybackUnavailable(
                "This platform has no Unix socket support for the mpv IPC "
                "channel."
            )

        self._directory = tempfile.TemporaryDirectory(prefix="onkith-mpv-")
        socket_path = Path(self._directory.name) / "mpv.sock"
        arguments = [
            mpv,
            "--idle=yes",
            "--no-video",
            "--no-terminal",
            "--no-config",
            "--audio-display=no",
            "--ytdl-format=bestaudio/best",
            "--cache=yes",
            f"--volume={self._user_volume}",
            f"--input-ipc-server={socket_path}",
        ]
        ytdl = find_executable("yt-dlp", self._ytdl_path)
        if ytdl is not None:
            arguments.append(f"--script-opts=ytdl_hook-ytdl_path={ytdl}")
        arguments.extend(self._extra_arguments)

        self._process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._socket = self._connect(socket_path)
        self._reader = threading.Thread(
            target=self._read_loop,
            name="onkith-mpv-reader",
            daemon=True,
        )
        self._reader.start()
        for index, name in enumerate(_OBSERVED_PROPERTIES, start=1):
            self._send({"command": ["observe_property", index, name]})
        self._publish({"available": True, "state": "idle"})

    def _connect(self, socket_path: Path) -> socket.socket:
        deadline = time.monotonic() + _CONNECT_TIMEOUT
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise PlaybackUnavailable(
                    "mpv exited before its IPC socket was ready."
                )
            if socket_path.exists():
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    connection.connect(str(socket_path))
                    return connection
                except OSError as error:
                    last_error = error
                    connection.close()
            time.sleep(0.05)
        raise PlaybackUnavailable(
            f"mpv did not open its IPC socket in time: {last_error}"
        )

    # -------------------------------------------------------------- plumbing

    def _send(
        self, message: dict[str, Any], *, wait: bool = False
    ) -> dict[str, Any] | None:
        connection = self._socket
        if connection is None:
            return None
        with self._pending_lock:
            self._request_id += 1
            request_id = self._request_id
            event = threading.Event()
            if wait:
                self._replies[request_id] = event
        payload = dict(message)
        payload["request_id"] = request_id
        line = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            with self._send_lock:
                connection.sendall(line)
        except OSError:
            with self._pending_lock:
                self._replies.pop(request_id, None)
            return None
        if not wait:
            return None
        if not event.wait(_REPLY_TIMEOUT):
            with self._pending_lock:
                self._replies.pop(request_id, None)
            return None
        with self._pending_lock:
            self._replies.pop(request_id, None)
            return self._pending.pop(request_id, None)

    def _read_loop(self) -> None:
        connection = self._socket
        if connection is None:
            return
        buffer = b""
        while not self._shutdown.is_set():
            try:
                chunk = connection.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                self._handle(message)
        if not self._shutdown.is_set():
            self._publish({"available": False, "playing": False,
                           "state": "idle"})

    def _handle(self, message: dict[str, Any]) -> None:
        request_id = message.get("request_id")
        if request_id is not None and "error" in message:
            with self._pending_lock:
                event = self._replies.get(int(request_id))
                if event is not None:
                    self._pending[int(request_id)] = message
                    event.set()
            return
        if message.get("event") == "property-change":
            self._handle_property(
                str(message.get("name") or ""), message.get("data")
            )
            return
        event_name = message.get("event")
        if event_name == "end-file":
            reason = str(message.get("reason") or "")
            if reason == "eof":
                self._publish({"state": "idle", "playing": False})

    def _publish_transport(self) -> None:
        """Report one coherent transport state from the separate flags."""

        if self._idle:
            state, playing = "idle", False
        elif self._paused:
            state, playing = "paused", False
        elif self._stalled:
            state, playing = "buffering", False
        else:
            state, playing = "playing", True
        self._publish({"playing": playing, "state": state})

    def _handle_property(self, name: str, data: Any) -> None:
        if name == "pause":
            self._paused = bool(data)
            self._publish_transport()
            return
        if name == "time-pos":
            if data is None:
                return
            with self._state_lock:
                self._position = float(data)
            now = time.monotonic()
            if now - self._last_position_publish < POSITION_PUBLISH_INTERVAL:
                return
            self._last_position_publish = now
            self._publish({"position": float(data)})
            return
        if name == "duration":
            duration = float(data) if data else 0.0
            with self._state_lock:
                self._duration = duration
            self._publish({"duration": duration})
            return
        if name == "volume":
            # mpv reports the ducked value while speech is playing; the
            # surface must keep showing the level the listener chose.
            return
        if name == "mute":
            self._muted = bool(data)
            self._publish({"muted": bool(data)})
            return
        if name == "playlist-pos":
            if data is None or int(data) < 0:
                return
            self._index = int(data)
            self._publish({"index": int(data)})
            self.bus.publish("tracks", {"index": int(data)})
            return
        if name == "paused-for-cache":
            self._stalled = bool(data)
            self._publish_transport()
            return
        if name == "idle-active":
            self._idle = bool(data)
            self._publish_transport()

    def _publish(self, patch: dict[str, Any]) -> None:
        try:
            self.bus.publish("playback", patch)
        except Exception:
            # Surface fan-out must never break the audio path.
            pass

    # -------------------------------------------------------------- commands

    def load(self, tracks: list[dict[str, str]], *, start_index: int = 0) -> None:
        """Replace the queue with verified results and begin the first one."""

        if not tracks:
            return
        self._tracks = list(tracks)
        self._index = 0
        self._last_position_publish = 0.0
        for offset, track in enumerate(self._tracks):
            mode = "replace" if offset == 0 else "append"
            self._send(
                {
                    "command": [
                        "loadfile",
                        watch_url(track["video_id"]),
                        mode,
                    ]
                }
            )
        self.bus.publish(
            "tracks",
            {
                "index": 0,
                "items": [
                    {
                        "video_id": track["video_id"],
                        "title": track.get("title", ""),
                        "channel": track.get("channel", ""),
                        "thumbnail": track.get("thumbnail", ""),
                    }
                    for track in self._tracks
                ],
            },
        )
        self._send({"command": ["set_property", "pause", False]})
        self._publish({"available": True, "state": "buffering", "index": 0})
        if start_index > 0:
            self.select(start_index)

    def select(self, index: int) -> None:
        if not self._tracks:
            return
        bounded = max(0, min(len(self._tracks) - 1, int(index)))
        self._send({"command": ["set_property", "playlist-pos", bounded]})
        self._send({"command": ["set_property", "pause", False]})

    def toggle(self) -> None:
        self._send({"command": ["cycle", "pause"]})

    def play(self) -> None:
        self._send({"command": ["set_property", "pause", False]})

    def pause(self) -> None:
        self._send({"command": ["set_property", "pause", True]})

    def stop(self) -> None:
        """Stop and return to the beginning, keeping the queue loaded."""

        self._send({"command": ["seek", 0, "absolute"]})
        self._send({"command": ["set_property", "pause", True]})
        with self._state_lock:
            self._position = 0.0
        self._publish({"playing": False, "position": 0.0, "state": "stopped"})

    def next(self) -> None:
        self._send({"command": ["playlist-next", "weak"]})

    def previous(self) -> None:
        self._send({"command": ["playlist-prev", "weak"]})

    def seek_ratio(self, ratio: float) -> None:
        with self._state_lock:
            duration = self._duration
        if duration <= 0:
            return
        bounded = max(0.0, min(1.0, float(ratio)))
        target = duration * bounded
        self._send({"command": ["seek", target, "absolute"]})
        with self._state_lock:
            self._position = target
        self._publish({"position": target})

    def set_volume(self, value: float) -> None:
        self._user_volume = max(0, min(100, int(round(float(value)))))
        self._apply_volume()
        self._publish({"volume": self._user_volume})

    def set_muted(self, muted: bool) -> None:
        self._send(
            {"command": ["set_property", "mute", bool(muted)]}
        )

    def toggle_mute(self) -> None:
        self._send({"command": ["cycle", "mute"]})

    # ---------------------------------------------------------------- duck

    def _apply_volume(self) -> None:
        level = self._user_volume * (DUCK_RATIO if self._ducked else 1.0)
        self._send({"command": ["set_property", "volume", round(level, 2)]})

    def duck(self) -> None:
        """Lower music to roughly a fifth while speech is playing."""

        if self._ducked:
            return
        self._ducked = True
        self._apply_volume()
        self._publish({"ducked": True})

    def restore(self) -> None:
        if not self._ducked:
            return
        self._ducked = False
        self._apply_volume()
        self._publish({"ducked": False})

    @property
    def ducked(self) -> bool:
        return self._ducked

    @property
    def volume(self) -> int:
        return self._user_volume

    @property
    def position(self) -> float:
        with self._state_lock:
            return self._position

    @property
    def duration(self) -> float:
        with self._state_lock:
            return self._duration

    # ------------------------------------------------------------- shutdown

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._send({"command": ["quit"]})
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2)
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except OSError:
                pass
        self._process = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        self._publish({"available": False, "playing": False, "state": "idle"})
