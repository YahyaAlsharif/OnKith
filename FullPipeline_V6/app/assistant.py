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
import time
from typing import Any, Callable

from app.llm import LLMUnavailable, ModelGone, parse_command
from app.rehydrate import ConversationPlaceholders

LogFunction = Callable[[str], None]

MAX_RETAINED_TURNS = 20
UNREACHABLE_REPLY = "I could not reach the network just then."
EMPTY_REPLY = "I heard you, but nothing came back."
# A withdrawn model name reaches the network perfectly well and is told
# no such thing exists. Saying "I could not reach the network" sends
# somebody to check the wifi for a fault that is in a config file, which
# is where an evening went.
MISCONFIGURED_REPLY = "My language model is not set up correctly."
# Said, and shown, when the question cleared the privacy gate and then had
# nowhere to go. Distinct from the network failure above: nothing was sent, so
# telling somebody the network is down would send them to check the wifi for a
# key that was never set.
UNCONFIGURED_REPLY = (
    "I have no language model key configured, so I kept that one to myself."
)


class AssistantWorker:
    """Serialise replies away from the masking and capture lanes."""

    def __init__(
        self,
        *,
        llm: Any,
        speech: Any,
        bus: Any,
        log: LogFunction = lambda message: None,
        on_reply: Callable[[float, bool], None] = lambda seconds, failed: None,
        on_exchange: Callable[[str, str, str | None], None] = (
            lambda direction, text, note: None
        ),
        on_restored: Callable[[str], None] = lambda text: None,
    ) -> None:
        self.llm = llm
        self.speech = speech
        self.bus = bus
        self.log = log
        # How long the network round trip took, reported whether or not it
        # produced an answer. This is the only latency in the run that is
        # not the board's fault, and separating it from the rest is the
        # difference between "the Pi is too slow" and "the wifi is".
        self.on_reply = on_reply
        # The two legs of the round trip, verbatim and still masked, and then
        # the restored answer. A screen in the room is the only place the
        # boundary can actually be demonstrated rather than asserted, so the
        # worker reports both sides rather than only the half a person hears.
        # Neither hook is ever given the entity map or the raw transcript.
        self.on_exchange = on_exchange
        self.on_restored = on_restored
        self.queue: queue.Queue[
            tuple[str, list, Callable[[str], None] | None, bool] | None
        ] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.errors: list[str] = []
        self._turns: list[dict[str, str]] = []
        self._rehydration = ConversationPlaceholders()
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

    def submit(
        self,
        masked_text: str,
        entities: list | None = None,
        on_command: Callable[[str], None] | None = None,
    ) -> None:
        """Queue one already-masked utterance, with its span map.

        ``entities`` is the masker's record of what it replaced. It stays
        on this device and is used only to put the values back into the
        reply before it is spoken.

        ``on_command`` is how an utterance the device failed to recognise
        as an instruction still becomes one. It is given per submission
        rather than held on the worker because it closes over the raw
        transcript of this particular utterance, which is the only copy
        with the numbers and titles still in it, and which never leaves
        this process.
        """

        text = (masked_text or "").strip()
        if not text or self._shutdown:
            return
        self.queue.put((text, list(entities or ()), on_command, True))

    def note_command(self, masked_text: str, entities: list | None = None) -> None:
        """Put a command on the screen without asking the model anything.

        A command the device recognises never reaches the network, which
        is the point of recognising it. Until now that also meant it
        never reached the screen: somebody said "turn the light on", the
        lamp changed, and the conversation showed nothing at all, which
        reads as the device not having heard.

        Queued rather than recorded here, so the worker thread stays the
        only owner of the turns and the rehydration map. Nothing is sent.
        """

        text = (masked_text or "").strip()
        if not text or self._shutdown:
            return
        self.queue.put((text, list(entities or ()), None, False))

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

    def _note_reply(self, seconds: float, failed: bool = False) -> None:
        try:
            self.on_reply(seconds, failed)
        except Exception:
            # A meter that cannot be written to must not cost somebody
            # their answer.
            pass

    def _boundary(self, direction: str, text: str, note: str | None = None) -> None:
        try:
            self.on_exchange(direction, text, note)
        except Exception:
            # A display that cannot be drawn must not cost somebody an answer.
            pass

    def _restored(self, text: str) -> None:
        try:
            self.on_restored(text)
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
                masked_text, entities, on_command, answer = item
                # Renumbered into the conversation's own placeholder space
                # before anything else sees the text, so every later stage -
                # the screen, the request, the model's history and the
                # rehydration below - is working with one numbering in which a
                # placeholder means exactly one value. The masker numbers per
                # utterance, so without this two different people can both be
                # [GIVENNAME_1] and a reply about the first is restored as the
                # second. The map is grown before the call because a reply may
                # refer to a placeholder from any earlier turn.
                masked_text, _ = self._rehydration.adopt(masked_text, entities)
                self._record("user", masked_text)
                if not answer:
                    # A command. Shown, carried out elsewhere, not asked
                    # about.
                    continue
                self._state("thinking")
                if self.configured:
                    # Printed immediately before the request and taken from the
                    # same string the request is built from, so what the screen
                    # shows is what the network was given rather than a
                    # reconstruction of it.
                    self._boundary("out", masked_text)
                else:
                    self._boundary(
                        "unsent",
                        masked_text,
                        "GROQ_API_KEY is not set; no request was made",
                    )
                asked_at = time.perf_counter()
                try:
                    reply = self.llm.reply_to(masked_text)
                except LLMUnavailable as error:
                    self.log(str(error))
                    self._record("onkith", UNCONFIGURED_REPLY)
                    self._restored(UNCONFIGURED_REPLY)
                    self._state("idle")
                    continue
                except Exception as error:
                    self._note_reply(time.perf_counter() - asked_at, failed=True)
                    message = f"{type(error).__name__}: {error}"
                    self.errors.append(message)
                    self.log(f"Assistant reply failed: {message}")
                    self._consecutive_failures += 1
                    spoken = (
                        MISCONFIGURED_REPLY
                        if isinstance(error, ModelGone)
                        else UNREACHABLE_REPLY
                    )
                    # Written to the screen every time, even though it is
                    # only spoken once. Speaking the same sentence after
                    # every utterance is unbearable, but leaving the screen
                    # blank is worse: the question sits there with nothing
                    # under it and the device looks like it ignored
                    # somebody, which is exactly what it looks like when
                    # the wake word was never heard. Those two failures are
                    # nothing alike and have to look different.
                    self._boundary("unsent", masked_text, "request failed")
                    self._record("onkith", spoken)
                    self._restored(spoken)
                    self._state("idle")
                    if self._consecutive_failures == 1:
                        self.speech.speak(spoken)
                    continue

                self._consecutive_failures = 0
                self._note_reply(time.perf_counter() - asked_at)

                # An instruction the device failed to recognise itself.
                # Carried out rather than answered, and never spoken:
                # the line is a wire format and reading "command colon
                # light on" out loud would be a fault, not a reply.
                intent = parse_command(reply)
                if intent is not None:
                    self._state("idle")
                    if on_command is None:
                        # Nothing wired to act on it. Saying so beats
                        # sitting in silence, which looks like the wake
                        # word was never heard.
                        self.log(f"The assistant asked for {intent} with nothing to do it.")
                        continue
                    try:
                        on_command(intent)
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        self.errors.append(message)
                        self.log(f"{intent} from the assistant failed: {message}")
                    continue

                if not reply:
                    # An empty reply used to be dropped in silence here, and
                    # a compound model that answers entirely inside its tool
                    # results returns exactly that.
                    self.log("The model returned an empty reply.")
                    self._record("onkith", EMPTY_REPLY)
                    self._restored(EMPTY_REPLY)
                    self._state("idle")
                    continue
                # What came back, before anything local touches it. Together
                # with the "out" line above this is the entire cloud side of
                # the exchange, and a reader can check both of them for a
                # private value rather than take the claim on trust.
                self._boundary("in", reply)
                # Below this line the screen and the speakers put the values
                # back. What the model was given is still the masked copy:
                # nothing here changes what left the device, only what is shown
                # to the person standing in front of it.
                self._record("onkith", reply)
                missing = self._rehydration.missing(reply)
                if missing:
                    self.log(
                        "The reply used a placeholder with no recorded "
                        f"value: {missing[0][0]}. It will be spoken as is."
                    )
                restored = self._rehydration.apply(reply)
                self._restored(restored)
                self.speech.speak(restored)
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
