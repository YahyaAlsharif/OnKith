"""Surface server tests: framing, snapshot delivery, and command intake."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core_server import CoreServer, accept_key, encode_frame
from app.event_bus import EventBus


class WebSocketTestClient:
    """A minimal RFC 6455 client used only by these tests."""

    def __init__(self, host: str, port: int, path: str = "/events") -> None:
        self.socket = socket.create_connection((host, port), timeout=5)
        self.socket.settimeout(5)
        self._buffer = b""
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        header = self._read_until(b"\r\n\r\n").decode("latin-1")
        self.status_line = header.split("\r\n", 1)[0]
        self.expected_accept = accept_key(key)
        self.headers = {}
        for line in header.split("\r\n")[1:]:
            if ": " in line:
                name, value = line.split(": ", 1)
                self.headers[name.lower()] = value

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self._buffer:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise AssertionError("Connection closed during handshake")
            self._buffer += chunk
        index = self._buffer.index(marker) + len(marker)
        head, self._buffer = self._buffer[:index], self._buffer[index:]
        return head

    def _read_exactly(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise AssertionError("Connection closed mid-frame")
            self._buffer += chunk
        head, self._buffer = self._buffer[:count], self._buffer[count:]
        return head

    def receive_json(self) -> dict:
        header = self._read_exactly(2)
        if header[0] & 0x80 == 0:
            raise AssertionError("Server sent a fragmented frame")
        if header[1] & 0x80:
            raise AssertionError("Server frames must not be masked")
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exactly(8))[0]
        return json.loads(self._read_exactly(length).decode("utf-8"))

    def send_json(self, message: dict) -> None:
        payload = json.dumps(message).encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        else:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        masked = bytes(
            byte ^ mask[index % 4] for index, byte in enumerate(payload)
        )
        self.socket.sendall(bytes(header) + mask + masked)

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass


class FrameTest(unittest.TestCase):
    def test_accept_key_matches_the_specification_example(self) -> None:
        # The example pair published in RFC 6455 section 1.3.
        self.assertEqual(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
        )

    def test_frame_lengths_use_the_smallest_header(self) -> None:
        self.assertEqual(encode_frame(b"hi")[:2], bytes([0x81, 2]))
        medium = encode_frame(b"x" * 200)
        self.assertEqual(medium[:2], bytes([0x81, 126]))
        self.assertEqual(struct.unpack("!H", medium[2:4])[0], 200)
        large = encode_frame(b"x" * 70000)
        self.assertEqual(large[:2], bytes([0x81, 127]))
        self.assertEqual(struct.unpack("!Q", large[2:10])[0], 70000)


class CoreServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="onkith-server-")
        self.received: list[tuple[str, dict]] = []
        self.command_seen = threading.Event()
        self.bus = EventBus()

        def on_command(name: str, payload: dict) -> None:
            self.received.append((name, payload))
            self.command_seen.set()

        self.server = CoreServer(
            Path(self.directory.name), self.bus, on_command=on_command
        )
        self.server.start()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.server.shutdown)

    def test_handshake_and_snapshot(self) -> None:
        client = WebSocketTestClient("127.0.0.1", self.server.port)
        self.addCleanup(client.close)
        self.assertIn("101", client.status_line)
        self.assertEqual(
            client.headers.get("sec-websocket-accept"), client.expected_accept
        )
        snapshot = client.receive_json()
        self.assertEqual(snapshot["type"], "snapshot")
        self.assertIn("playback", snapshot)
        self.assertFalse(snapshot["playback"]["playing"])

    def test_published_sections_reach_the_surface(self) -> None:
        client = WebSocketTestClient("127.0.0.1", self.server.port)
        self.addCleanup(client.close)
        client.receive_json()
        self.bus.publish("playback", {"playing": True, "position": 12.5})
        message = client.receive_json()
        self.assertEqual(message["type"], "playback")
        self.assertTrue(message["playing"])
        self.assertEqual(message["position"], 12.5)

    def test_unchanged_publish_sends_nothing_further(self) -> None:
        client = WebSocketTestClient("127.0.0.1", self.server.port)
        self.addCleanup(client.close)
        client.receive_json()
        self.bus.publish("playback", {"playing": True})
        self.assertTrue(client.receive_json()["playing"])
        # Republishing the same value must not produce a second frame.
        self.bus.publish("playback", {"playing": True})
        self.bus.publish("status", {"message": "distinct"})
        self.assertEqual(client.receive_json()["type"], "status")

    def test_surface_commands_reach_the_core(self) -> None:
        client = WebSocketTestClient("127.0.0.1", self.server.port)
        self.addCleanup(client.close)
        client.receive_json()
        client.send_json({"command": "toggle", "payload": {"source": "test"}})
        self.assertTrue(self.command_seen.wait(5))
        self.assertEqual(self.received[0][0], "toggle")
        self.assertEqual(self.received[0][1]["source"], "test")

    def test_malformed_commands_are_ignored(self) -> None:
        client = WebSocketTestClient("127.0.0.1", self.server.port)
        self.addCleanup(client.close)
        client.receive_json()
        client.send_json({"nonsense": True})
        client.send_json({"command": "", "payload": {}})
        client.send_json({"command": "stop"})
        self.assertTrue(self.command_seen.wait(5))
        self.assertEqual([name for name, _ in self.received], ["stop"])

    def test_static_assets_are_served(self) -> None:
        (Path(self.directory.name) / "player.html").write_text(
            "<!doctype html><title>ok</title>", encoding="utf-8"
        )
        import urllib.request

        with urllib.request.urlopen(self.server.url, timeout=5) as response:
            body = response.read().decode("utf-8")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertIn("<title>ok</title>", body)

    def test_a_disconnected_surface_does_not_stop_publishing(self) -> None:
        first = WebSocketTestClient("127.0.0.1", self.server.port)
        second = WebSocketTestClient("127.0.0.1", self.server.port)
        self.addCleanup(second.close)
        first.receive_json()
        second.receive_json()
        first.close()
        self.bus.publish("playback", {"position": 7.0})
        self.assertEqual(second.receive_json()["position"], 7.0)


if __name__ == "__main__":
    unittest.main()
