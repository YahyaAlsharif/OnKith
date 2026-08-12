"""Turn masked speech into a spoken reply.

This worker sits after the masker and before text to speech. It only ever
receives text the masker has already cleared for release, so the private
values are placeholders by the time anything leaves the device.

It runs on its own thread for the same reason the music worker does: a
network round trip must never stall the masking lane or the microphone.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

from app.llm import LLMUnavailable
from app.rehydrate import RehydrationMap

LogFunction = Callable[[str], None]

MAX_RETAINED_TURNS = 20
UNREACHABLE_REPLY = "I could not reach the network just then."
EMPTY_REPLY = "I heard you, but nothing came back."


class AssistantWorker:
    """Serialise replies away from the masking and capture lanes."""

    def __init__(
        self,
        *,
        llm: Any,
        speech: Any,
        bus: Any,
        log: LogFunction = lambda message: None,
    ) -> None:
        self.llm = llm
        self.speech = speech
        self.bus = bus
        self.log = log
        self.queue: queue.Queue[tuple[str, list] | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.errors: list[str] = []
        self._turns: list[dict[str, str]] = []
        self._rehydration = RehydrationMap()
        self._consecutive_failures = 0
        self._shutdown = False

    @property
    def configured(self) -> bool:
        return bool(getattr(self.llm, "configured", False))

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._run,
            name="onkith-assistant-worker",
            daemon=False,
        )
        self.thread.start()

    def submit(self, masked_text: str, entities: list | None = None) -> None:
        """Queue one already-masked utterance, with its span map.

        ``entities`` is the masker's record of what it replaced. It stays
        on this device and is used only to put the values back into the
        reply before it is spoken.
        """

        text = (masked_text or "").strip()
        if not text or self._shutdown:
            return
        self.queue.put((text, list(entities or ())))

    def _record(self, role: str, text: str) -> None:
        # The screen is in the room with the person who spoke, so it shows
        # what they actually said. The placeholders exist to keep values away
        # from a network model, and a surface on this device is not one: it
        # is served from localhost to a browser three inches from the mouth
        # that said the words. Reading "[GIVENNAME_1]" back to somebody about
        # their own name is theatre rather than privacy.
        #
        # The transcript database is a different question and keeps the
        # placeholders. It is the only artifact that outlives the process, so
        # it is the only one that can be carried off on a card.
        # The spans travel with it so the screen can mark exactly which words
        # were held back from the model. That marking is the demonstration:
        # the room sees the private values sitting in the sentence and knows
        # the model never did.
        shown, revealed = self._rehydration.apply_with_spans(text)
        self._turns.append({"role": role, "text": shown, "revealed": revealed})
        excess = len(self._turns) - MAX_RETAINED_TURNS
        if excess > 0:
            del self._turns[:excess]
        # Retained so a surface that connects or reloads mid-conversation
        # still has the exchange to render.
        try:
            self.bus.publish("conversation", {"turns": list(self._turns)})
        except Exception:
            pass

    def _state(self, state: str) -> None:
        try:
            self.bus.publish("assistant", {"state": state})
        except Exception:
            pass

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                masked_text, entities = item
                # The map is added before the call so a reply can refer to
                # a placeholder from any earlier turn in the conversation.
                self._rehydration.add(entities)
                self._record("user", masked_text)
                self._state("thinking")
                try:
                    reply = self.llm.reply_to(masked_text)
                except LLMUnavailable as error:
                    self.log(str(error))
                    self._state("idle")
                    continue
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    self.errors.append(message)
                    self.log(f"Assistant reply failed: {message}")
                    self._consecutive_failures += 1
                    # Written to the screen every time, even though it is
                    # only spoken once. Speaking the same sentence after
                    # every utterance is unbearable, but leaving the screen
                    # blank is worse: the question sits there with nothing
                    # under it and the device looks like it ignored
                    # somebody, which is exactly what it looks like when
                    # the wake word was never heard. Those two failures are
                    # nothing alike and have to look different.
                    self._record("onkith", UNREACHABLE_REPLY)
                    self._state("idle")
                    if self._consecutive_failures == 1:
                        self.speech.speak(UNREACHABLE_REPLY)
                    continue

                self._consecutive_failures = 0
                if not reply:
                    # An empty reply used to be dropped in silence here, and
                    # a compound model that answers entirely inside its tool
                    # results returns exactly that.
                    self.log("The model returned an empty reply.")
                    self._record("onkith", EMPTY_REPLY)
                    self._state("idle")
                    continue
                # Both the screen and the speakers put the values back. What
                # the model was given is still the masked copy: nothing about
                # this changes what left the device, only what is shown to
                # the person standing in front of it.
                self._record("onkith", reply)
                missing = self._rehydration.missing(reply)
                if missing:
                    self.log(
                        "The reply used a placeholder with no recorded "
                        f"value: {missing[0][0]}. It will be spoken as is."
                    )
                self.speech.speak(self._rehydration.apply(reply))
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                self._state("idle")
            finally:
                self.queue.task_done()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self.thread is None:
            return
        self.queue.join()
        self.queue.put(None)
        self.thread.join(timeout=30)
