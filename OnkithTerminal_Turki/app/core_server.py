"""Localhost surface server: static player assets plus a websocket feed.

The core process owns this server. Surfaces connect to ``/events`` to
receive the authoritative state and to send transport commands back. No
surface is required for audio to run, so every failure here is contained
and never propagates into playback.

The websocket implementation is deliberately local and dependency free,
matching the vendored Three.js and the zero-dependency terminal UI.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import queue
import socket
import struct
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
EVENTS_PATH = "/events"
_OUTBOUND_LIMIT = 256

_OPCODE_CONTINUATION = 0x0
_OPCODE_TEXT = 0x1
_OPCODE_BINARY = 0x2
_OPCODE_CLOSE = 0x8
_OPCODE_PING = 0x9
_OPCODE_PONG = 0xA

CommandHandler = Callable[[str, dict[str, Any]], None]


class WebSocketClosed(Exception):
    """The peer closed the connection or the socket failed."""


def accept_key(client_key: str) -> str:
    """Derive the RFC 6455 handshake response for one client key."""

    digest = hashlib.sha1(
        (client_key.strip() + WEBSOCKET_GUID).encode("ascii")
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, opcode: int = _OPCODE_TEXT) -> bytes:
    """Build one unmasked server frame."""

    header = bytearray()
    header.append(0x80 | opcode)
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


class WebSocketConnection:
    """One framed connection with an independent writer thread."""

    def __init__(self, handler: SimpleHTTPRequestHandler) -> None:
        self._handler = handler
        self._rfile = handler.rfile
        self._socket = handler.connection
        self._outbound: queue.Queue[bytes | None] = queue.Queue(
            maxsize=_OUTBOUND_LIMIT
        )
        self._write_lock = threading.Lock()
        self._closed = threading.Event()
        self._writer = threading.Thread(
            target=self._write_loop,
            name="onkith-ws-writer",
            daemon=True,
        )
        self._writer.start()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def send_json(self, message: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            return
        try:
            self._outbound.put_nowait(encode_frame(payload))
        except queue.Full:
            # A stalled surface must not slow the core. State is published
            # as whole sections, so the next update restores correctness.
            pass

    def _write_loop(self) -> None:
        while True:
            frame = self._outbound.get()
            if frame is None:
                return
            try:
                with self._write_lock:
                    self._socket.sendall(frame)
            except OSError:
                self._closed.set()
                return

    def _send_control(self, opcode: int, payload: bytes = b"") -> None:
        try:
            with self._write_lock:
                self._socket.sendall(encode_frame(payload, opcode))
        except OSError:
            self._closed.set()

    def _read_exactly(self, count: int) -> bytes:
        data = self._rfile.read(count)
        if data is None or len(data) < count:
            raise WebSocketClosed()
        return data

    def read_message(self) -> str | None:
        """Return one complete text message, or None for a non-text frame."""

        fragments = bytearray()
        message_opcode: int | None = None
        while True:
            header = self._read_exactly(2)
            final = bool(header[0] & 0x80)
            opcode = header[0] & 0x0F
            masked = bool(header[1] & 0x80)
            length = header[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exactly(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exactly(8))[0]
            if length > 1 << 20:
                # A surface never sends anything large; refuse to buffer it.
                raise WebSocketClosed()
            mask = self._read_exactly(4) if masked else b""
            payload = self._read_exactly(length) if length else b""
            if masked:
                payload = bytes(
                    byte ^ mask[index % 4]
                    for index, byte in enumerate(payload)
                )

            if opcode == _OPCODE_CLOSE:
                self._send_control(_OPCODE_CLOSE)
                raise WebSocketClosed()
            if opcode == _OPCODE_PING:
                self._send_control(_OPCODE_PONG, payload)
                continue
            if opcode == _OPCODE_PONG:
                continue
            if opcode in (_OPCODE_TEXT, _OPCODE_BINARY):
                message_opcode = opcode
                fragments.extend(payload)
            elif opcode == _OPCODE_CONTINUATION:
                fragments.extend(payload)
            else:
                raise WebSocketClosed()

            if not final:
                continue
            if message_opcode != _OPCODE_TEXT:
                return None
            try:
                return fragments.decode("utf-8")
            except UnicodeDecodeError:
                raise WebSocketClosed() from None

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._outbound.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class CoreRequestHandler(SimpleHTTPRequestHandler):
    """Serve the player surface and upgrade ``/events`` to a websocket."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()

    def _is_local_peer(self) -> bool:
        host = self.client_address[0] if self.client_address else ""
        return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?", 1)[0] == EVENTS_PATH:
            self._handle_events()
            return
        super().do_GET()

    def _handle_events(self) -> None:
        if not self._is_local_peer():
            self.send_error(403, "Local surfaces only")
            return
        upgrade = (self.headers.get("Upgrade") or "").strip().lower()
        client_key = self.headers.get("Sec-WebSocket-Key")
        if upgrade != "websocket" or not client_key:
            self.send_error(400, "Expected a websocket upgrade")
            return

        # The handshake is written directly rather than through
        # send_response/end_headers: those buffer headers for HTTP/1.1 and
        # would also attach the cache headers this response must not carry.
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(client_key)}\r\n"
            "\r\n"
        )
        self.wfile.write(handshake.encode("ascii"))
        self.wfile.flush()
        self.close_connection = True

        server: "CoreServer" = self.server.core  # type: ignore[attr-defined]
        server.serve_connection(self)


class _ThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    core: "CoreServer"


class CoreServer:
    """Own one localhost port for player assets and the surface feed."""

    def __init__(
        self,
        directory: Path,
        bus: Any,
        *,
        on_command: CommandHandler | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bus = bus
        self.on_command = on_command
        self._connections: set[WebSocketConnection] = set()
        self._connections_lock = threading.Lock()

        def build_handler(*args: Any, **kwargs: Any) -> CoreRequestHandler:
            return CoreRequestHandler(
                *args, directory=str(self.directory), **kwargs
            )

        self._server = _ThreadingServer((host, port), build_handler)
        self._server.core = self
        self.host = host
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="onkith-core-server",
            daemon=True,
        )
        self._subscription: int | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/player.html"

    def start(self) -> None:
        if self._subscription is None:
            self._subscription = self.bus.subscribe(self._broadcast)
        self._thread.start()

    def _broadcast(self, message: dict[str, Any]) -> None:
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            connection.send_json(message)

    def serve_connection(self, handler: SimpleHTTPRequestHandler) -> None:
        connection = WebSocketConnection(handler)
        with self._connections_lock:
            self._connections.add(connection)
        try:
            snapshot = self.bus.snapshot()
            connection.send_json({"type": "snapshot", **snapshot})
            while True:
                raw = connection.read_message()
                if raw is None:
                    continue
                self._dispatch(raw)
        except (WebSocketClosed, OSError, ValueError):
            pass
        finally:
            with self._connections_lock:
                self._connections.discard(connection)
            connection.close()

    def _dispatch(self, raw: str) -> None:
        if self.on_command is None:
            return
        try:
            message = json.loads(raw)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        name = message.get("command")
        if not isinstance(name, str) or not name:
            return
        payload = message.get("payload")
        try:
            self.on_command(
                name, payload if isinstance(payload, dict) else {}
            )
        except Exception:
            # A malformed surface command must never stop the server.
            pass

    @property
    def connection_count(self) -> int:
        with self._connections_lock:
            return len(self._connections)

    def shutdown(self) -> None:
        if self._subscription is not None:
            self.bus.unsubscribe(self._subscription)
            self._subscription = None
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for connection in connections:
            connection.close()
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def free_port(host: str = "127.0.0.1") -> int:
    """Reserve one ephemeral port so a URL can be built before binding."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def is_headless_environment() -> bool:
    """Report whether no display server is reachable for a kiosk surface."""

    # macOS always has a window server and never sets DISPLAY, so the X11
    # and Wayland probes below would wrongly call a development Mac
    # headless.
    if platform.system() == "Darwin":
        return False
    if os.environ.get("WAYLAND_DISPLAY"):
        return False
    return not os.environ.get("DISPLAY")
