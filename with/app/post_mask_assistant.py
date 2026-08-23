"""Add wake routing, Groq, rehydration, TTS and UI after the privacy gate.

This module is intentionally ignorant of audio, VAD, STT inference, masking,
and SQLite.  Its only transcript input is ``LiveOutput`` after the existing
fail-closed masker has decided whether the text is safe to release.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from typing import Any, Callable

from app.assistant import AssistantWorker
from app.assistant_surface import AssistantSurface
from app.light import LightController
from app.llm import GroqClient
from app.onkith_music import (
    OnkithMusicWorker,
    play_query_from,
    requested_level,
    route_onkith_request,
    without_wake_word,
)
from app.speech import SpeechEngine
from app.wake_word_normalizer import normalize_wake_word
from integrated_pipeline.schemas import LiveOutput


LogFunction = Callable[[str], None]


class PostMaskAssistant:
    """A non-throwing additive consumer of privacy-cleared outputs."""

    def __init__(
        self,
        *,
        terminal_output: Callable[[LiveOutput], None],
        music: OnkithMusicWorker,
        port: int = 8765,
        open_window: bool = True,
        log: LogFunction = lambda message: None,
        llm: Any | None = None,
        speech: Any | None = None,
        surface: AssistantSurface | None = None,
        light: LightController | None = None,
    ) -> None:
        self.terminal_output = terminal_output
        self.music = music
        self.log = log
        self.surface = surface or AssistantSurface(
            port=port, open_window=open_window, log=log
        )
        self.llm = llm or GroqClient(log=log)
        self.speech = speech or SpeechEngine(self.surface, log=log)
        self.light = light or LightController(self.surface, log=log)
        self.assistant = AssistantWorker(
            llm=self.llm,
            speech=self.speech,
            bus=self.surface,
            log=log,
            on_reply=self._on_reply,
        )
        self._actions: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._action_thread: threading.Thread | None = None
        self._started = False
        self._shutdown = False

    @property
    def speaking(self) -> bool:
        return bool(getattr(self.speech, "speaking", False))

    @property
    def configured(self) -> bool:
        return bool(getattr(self.assistant, "configured", False))

    def start(self) -> None:
        if self._started:
            return
        # A screen is useful but not load-bearing. Voice and routing still
        # start if a port is occupied or Chromium is absent.
        try:
            self.surface.start()
        except Exception as error:
            self.log(f"Assistant UI unavailable: {type(error).__name__}: {error}")
        self.speech.start()
        self._action_thread = threading.Thread(
            target=self._action_loop,
            name="onkith-local-command-worker",
            daemon=False,
        )
        self._action_thread.start()
        self.assistant.start()
        self.light.announce()
        self.surface.publish(
            "devices",
            {
                "speech": getattr(self.speech, "backend_name", "none"),
                "groq": "configured" if self.configured else "missing API key",
            },
        )
        if not self.configured:
            self.surface.publish(
                "status",
                {
                    "message": (
                        "GROQ_API_KEY is not set. Local STT, masking, commands, "
                        "UI and TTS remain available."
                    )
                },
            )
        self._started = True

    def observe_event(self, name: str, data: dict[str, Any]) -> None:
        """Mirror safe runtime state; never alter the authoritative event path."""

        try:
            if name == "microphone_open":
                microphone = data.get("device_name") or data.get("name")
                if microphone:
                    self.surface.publish("devices", {"microphone": str(microphone)})
            elif name == "recording_started":
                self.surface.publish("assistant", {"state": "listening"})
            elif name in {"segment_closed", "stt_started"}:
                self.surface.publish("assistant", {"state": "transcribing"})
            elif name == "masking_started":
                self.surface.publish("assistant", {"state": "masking"})
            elif name == "onkith_command_queued":
                intent = str(data.get("intent") or "command").lower()
                self.surface.publish(
                    "status", {"message": f"Local {intent} command queued."}
                )
        except Exception:
            # Mirroring state can never break capture or STT.
            return

    def handle_output(self, result: LiveOutput) -> None:
        """Keep original output first, then route only safe post-mask text."""

        # Preserve the exact pre-existing terminal/UI callback and its failure
        # semantics. Everything additive below is guarded so it cannot prevent
        # the database completion that follows this callback.
        self.terminal_output(result)
        try:
            self._route(result)
        except Exception as error:
            self.log(
                "Post-mask assistant ignored an internal error: "
                f"{type(error).__name__}: {error}"
            )

    def _route(self, result: LiveOutput) -> None:
        pii = result.pii
        if not pii.safe_to_release:
            self._gate("blocked", "blocked by privacy")
            self.surface.publish(
                "status", {"message": "Privacy masking failed; nothing was sent."}
            )
            return

        raw = pii.original_text or result.buffer.text
        routing_copy, _ = normalize_wake_word(raw)
        addressed = route_onkith_request(routing_copy)
        if not addressed.command_detected:
            self._gate("held", "held locally")
            self.surface.publish(
                "status", {"message": "Ordinary speech remained on this device."}
            )
            return

        if addressed.intent != "ASK":
            self._gate("held", "local command")
            self.assistant.note_command(pii.masked_text, pii.entities)
            self.queue_command(str(addressed.intent), raw)
            return

        self._gate("sent", "masked text sent")
        if not self.configured:
            self.surface.publish(
                "status",
                {
                    "message": (
                        "Question was privacy-cleared, but GROQ_API_KEY is not set; "
                        "no network request was made."
                    )
                },
            )
        else:
            self.surface.publish(
                "status",
                {"message": "Only the masked question was sent to Groq."},
            )

        # The entity records stay with the local worker for reply rehydration.
        # GroqClient receives only the first argument, which is already masked.
        masked_question = without_wake_word(pii.masked_text).strip()
        self.assistant.submit(
            masked_question or pii.masked_text,
            pii.entities,
            on_command=lambda intent: self.queue_command(intent, raw),
        )

    def _gate(self, state: str, label: str) -> None:
        self.surface.publish("privacy_gate", {"state": state, "label": label})

    def _on_reply(self, seconds: float, failed: bool) -> None:
        if failed:
            message = f"Groq request failed after {seconds:.2f}s."
        else:
            message = f"Masked Groq reply received in {seconds:.2f}s; rehydrated locally."
        self.surface.publish("status", {"message": message})

    def handle_command(self, intent: str, raw: str) -> None:
        """Execute a named action using arguments read only from local raw text."""

        name = (intent or "").strip().upper()
        if name == "PLAY":
            query = play_query_from(raw)
            self.music.submit_play(query)
            message = f"Local music search queued: {query}" if query else "Local music search queued."
        elif name == "CLOSE":
            self.music.submit_close()
            message = "Local music close command queued."
        elif name == "SHOW":
            self.surface.publish("panel", {"open": True})
            message = "Player panel shown locally."
        elif name == "HIDE":
            self.surface.publish("panel", {"open": False})
            message = "Player panel hidden locally."
        elif name == "SHOW_DEVICES":
            self.surface.publish("devices", {"open": True})
            message = "Device panel shown locally."
        elif name == "HIDE_DEVICES":
            self.surface.publish("devices", {"open": False})
            message = "Device panel hidden locally."
        elif name == "LOUDER":
            self._step_volume("5%+")
            message = "Local output volume raised."
        elif name == "QUIETER":
            self._step_volume("5%-")
            message = "Local output volume lowered."
        elif name in {"LIGHT_ON", "LIGHT_OFF", "BRIGHTER", "DIMMER", "LIGHT_LEVEL"}:
            level = requested_level([word.casefold() for word in raw.split()])
            if name == "LIGHT_ON":
                self.light.on()
            elif name == "LIGHT_OFF":
                self.light.off()
            elif name == "BRIGHTER":
                self.light.step(self.light.step_size)
            elif name == "DIMMER":
                self.light.step(-self.light.step_size)
            elif level is not None:
                self.light.set_brightness(level)
            message = "Local light command applied."
        else:
            self.log(f"Unsupported assistant command ignored: {name}")
            return
        self.surface.publish("status", {"message": message})

    def queue_command(self, intent: str, raw: str) -> None:
        """Keep GPIO, mixer and music actions off the masking/LLM workers."""

        if not self._shutdown:
            self._actions.put((str(intent), str(raw)))

    def _action_loop(self) -> None:
        while True:
            item = self._actions.get()
            try:
                if item is None:
                    return
                self.handle_command(*item)
            except Exception as error:
                self.log(
                    "Local command failed: "
                    f"{type(error).__name__}: {error}"
                )
            finally:
                self._actions.task_done()

    def _step_volume(self, change: str) -> None:
        amixer = shutil.which("amixer")
        if amixer is None:
            self.log("amixer is unavailable; local volume command was ignored.")
            return
        subprocess.run(
            [amixer, "-q", "sset", "Master", change],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        # The assistant may queue a final model-decided command while it
        # drains, so close it first and drain the action queue second.
        try:
            self.assistant.shutdown()
        except Exception as error:
            self.log(f"assistant shutdown warning: {type(error).__name__}: {error}")
        self._shutdown = True
        if self._action_thread is not None:
            self._actions.join()
            self._actions.put(None)
            self._action_thread.join(timeout=10)
        for name, closer in (
            ("speech", self.speech.shutdown),
            ("light", self.light.close),
            ("surface", self.surface.shutdown),
        ):
            try:
                closer()
            except Exception as error:
                self.log(f"{name} shutdown warning: {type(error).__name__}: {error}")
