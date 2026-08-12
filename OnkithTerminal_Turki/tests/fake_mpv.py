#!/usr/bin/env python3
"""A minimal stand-in for mpv's JSON IPC server, used by the test suite.

It implements only the surface the playback engine relies on: an
``--input-ipc-server`` socket, newline-delimited JSON requests with
``request_id`` replies, ``observe_property`` registrations, and
``property-change`` events. Every received command is appended to a log
file so tests can assert on what the engine actually sent.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path


class FakeMpv:
    def __init__(self, socket_path: Path, log_path: Path, volume: float) -> None:
        self.socket_path = socket_path
        self.log_path = log_path
        self.properties: dict[str, object] = {
            "pause": False,
            "time-pos": 0.0,
            "duration": 0.0,
            "volume": volume,
            "mute": False,
            "playlist-pos": -1,
            "paused-for-cache": False,
            "idle-active": True,
        }
        self.observers: dict[str, int] = {}
        self.playlist: list[str] = []
        self.connection: socket.socket | None = None
        self.send_lock = threading.Lock()
        self.running = True

    def log(self, entry: object) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def send(self, message: dict) -> None:
        connection = self.connection
        if connection is None:
            return
        try:
            with self.send_lock:
                connection.sendall(
                    (json.dumps(message) + "\n").encode("utf-8")
                )
        except OSError:
            self.running = False

    def set_property(self, name: str, value: object) -> None:
        if self.properties.get(name) == value:
            return
        self.properties[name] = value
        observer_id = self.observers.get(name)
        if observer_id is not None:
            self.send(
                {
                    "event": "property-change",
                    "id": observer_id,
                    "name": name,
                    "data": value,
                }
            )

    def handle(self, command: list, request_id: int | None) -> object:
        if not command:
            return None
        name = command[0]
        self.log(command)

        if name == "observe_property":
            observer_id, property_name = int(command[1]), str(command[2])
            self.observers[property_name] = observer_id
            self.send(
                {
                    "event": "property-change",
                    "id": observer_id,
                    "name": property_name,
                    "data": self.properties.get(property_name),
                }
            )
            return None
        if name == "set_property":
            self.set_property(str(command[1]), command[2])
            return None
        if name == "cycle":
            property_name = str(command[1])
            self.set_property(
                property_name, not bool(self.properties.get(property_name))
            )
            return None
        if name == "loadfile":
            url, mode = str(command[1]), str(command[2])
            if mode == "replace":
                self.playlist = [url]
                self.set_property("playlist-pos", 0)
                self.set_property("duration", 210.0)
                self.set_property("idle-active", False)
                self.set_property("paused-for-cache", False)
            else:
                self.playlist.append(url)
            return None
        if name == "seek":
            target = float(command[1])
            self.set_property("time-pos", target)
            return None
        if name == "playlist-next":
            position = int(self.properties.get("playlist-pos") or 0)
            if position + 1 < len(self.playlist):
                self.set_property("playlist-pos", position + 1)
                self.set_property("time-pos", 0.0)
            return None
        if name == "playlist-prev":
            position = int(self.properties.get("playlist-pos") or 0)
            if position > 0:
                self.set_property("playlist-pos", position - 1)
                self.set_property("time-pos", 0.0)
            return None
        if name == "get_property":
            return self.properties.get(str(command[1]))
        if name == "quit":
            self.running = False
            return None
        return None

    def serve(self) -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(1)
        connection, _ = server.accept()
        self.connection = connection
        buffer = b""
        while self.running:
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
                except ValueError:
                    continue
                request_id = message.get("request_id")
                data = self.handle(message.get("command") or [], request_id)
                if request_id is not None:
                    self.send(
                        {
                            "error": "success",
                            "data": data,
                            "request_id": request_id,
                        }
                    )
        try:
            connection.close()
            server.close()
        except OSError:
            pass


def main() -> int:
    socket_path: Path | None = None
    log_path = Path(os.environ.get("FAKE_MPV_LOG", "fake-mpv.log"))
    volume = 100.0
    for argument in sys.argv[1:]:
        if argument.startswith("--input-ipc-server="):
            socket_path = Path(argument.split("=", 1)[1])
        elif argument.startswith("--volume="):
            volume = float(argument.split("=", 1)[1])
        elif argument.startswith("--onkith-log="):
            log_path = Path(argument.split("=", 1)[1])
    if socket_path is None:
        return 2
    FakeMpv(socket_path, log_path, volume).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
