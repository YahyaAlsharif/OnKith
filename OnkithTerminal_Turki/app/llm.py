"""Groq chat completions over the standard library.

The device sends masked text only. By the time anything reaches this
module the masker has already run and reported the text safe to release,
so private values have been replaced by placeholders such as
``[GIVENNAME_1]``. The system prompt explains those placeholders so the
reply does not try to guess what was behind them.

No SDK is used: one HTTPS request keeps the dependency list as short as
the rest of the runtime.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_TOKENS = 160
MAX_HISTORY_TURNS = 6

# A compound model answers by searching the web on the server, which costs
# about 3400 tokens a question against a free tier of 8000 a minute. That
# is two questions a minute before the device starts refusing people, and
# it was refusing them roughly every other call while this was measured.
# The plain model costs a few hundred tokens against a limit of 12000, so
# it is the thing to fall back to. It cannot search, so it answers from
# what it knows and from the clock this device gives it.
DEFAULT_FALLBACK_MODEL = "llama-3.3-70b-versatile"

# After a refusal, later questions go straight to the fallback rather than
# spending a round trip discovering the limit is still there. A minute is
# longer than any window Groq has asked us to wait, and short enough that
# a quiet spell puts the better model back.
RATE_LIMIT_COOLDOWN_SECONDS = 60.0

# The last line of this used to read "if a request needs a detail you
# cannot see, ask for just that detail". Sitting next to three worked
# examples of placeholder syntax, that taught the model to treat anything
# it did not know as a placeholder: asked the time, it answered
# "[TIME_1]", which no map could ever fill because nobody had said it. A
# placeholder may now only be echoed, never authored.
SYSTEM_PROMPT = (
    "You are Onkith, a voice assistant on a small home device. "
    "You are heard, never read, so reply in one or two short spoken "
    "sentences with no lists, no markdown and no emoji. "
    "Private details are removed before you see them and arrive as "
    "placeholders like [GIVENNAME_1], [TELEPHONENUM_1] or [EMAIL_1]. "
    "Treat each placeholder as a real value you simply cannot see: refer "
    "to it naturally if you must, never guess what it was, and never "
    "mention masking or placeholders unless you are asked about privacy. "
    "Never write a placeholder that was not already in the message you "
    "were given. If you do not know something, say so in plain words: a "
    "bracketed token is not a way of saying that you do not know. "
    # An instruction is recognised by the first word after the name, so
    # "Onkith can you play bones" and anything the recogniser mishears
    # arrive here instead of being carried out. Answering those with "I do
    # not understand" is the worst of both: the person learns nothing, and
    # the wording that would have worked is never said out loud.
    "This device also plays music, can show or hide a player on its "
    "screen, and has a lamp. Those few instructions are recognised by "
    "the first word after the name, so one that was misheard or worded "
    "another way reaches you instead of being carried out. Only when a "
    "message is plainly asking for music, for the player or for the "
    "light, and there is no way to answer it as a question, say in one "
    "sentence the wording that works: play and then what to play, or "
    "stop, show the player, close the player, louder, quieter, turn the "
    "light on, turn the light off, brighter, dimmer, or set the light to "
    "a percentage. Those are the whole "
    "list by voice, so do not offer skipping or pausing: by voice both "
    "of those stop the music instead, and the buttons for them are on "
    "the recorder on screen. Never say you have started or stopped "
    "anything, because that is not something you can do. "
    # The first version of this said to treat anything that read as an
    # instruction as a misrouted command. Given "my ID is 1234, how are
    # you doing" it recited the list of commands, which makes a thing
    # that answers almost anything look like it answers almost nothing.
    "Everything else is ordinary conversation and gets an ordinary "
    "answer. Most of what reaches you is not a command at all, so never "
    "answer a remark, a piece of news or a question with the list of "
    "commands. "
    # The name is stripped before a message gets here, but it can still
    # survive in the middle of a sentence, and once it did the model
    # closed with "users like you, Onkith". Your own name is not the
    # name of the person in the room.
    "Your name is yours. Whoever is speaking to you has a different one "
    "you have not been told, so never address them by yours."
)


def ambient_context(now: datetime | None = None) -> str:
    """What the device knows about the moment it is in.

    A model has no clock, so without this it cannot answer the most
    ordinary question anybody asks a thing sitting in their kitchen. This
    is context it should always have rather than something it should go and
    look up: the machine running it already knows.
    """

    moment = now if now is not None else datetime.now().astimezone()
    offset = moment.strftime("%z")
    return (
        "Right now it is "
        f"{moment.strftime('%H:%M on %A %d %B %Y')} "
        f"where this device is (UTC{offset[:3]}:{offset[3:]}). "
        "Use that for anything about the time or the date, and work out "
        "other places from it."
    )

LogFunction = Callable[[str], None]


class LLMUnavailable(RuntimeError):
    """No API key is configured, so the assistant cannot answer."""


class RateLimited(RuntimeError):
    """Groq refused this model for now. Another one may still answer."""


@dataclass
class GroqSettings:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.6
    system_prompt: str = SYSTEM_PROMPT

    @classmethod
    def from_environment(cls) -> "GroqSettings":
        return cls(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
            fallback_model=(
                os.environ.get("GROQ_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
                or DEFAULT_FALLBACK_MODEL
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def has_fallback(self) -> bool:
        """Falling back to the model that just refused is not a fallback."""

        return bool(self.fallback_model) and self.fallback_model != self.model


@dataclass
class GroqClient:
    """One conversation with Groq, with a bounded rolling history."""

    settings: GroqSettings = field(default_factory=GroqSettings.from_environment)
    log: LogFunction = lambda message: None
    _history: list[dict[str, str]] = field(default_factory=list, init=False)
    _rate_limited_until: float = field(default=0.0, init=False)

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def reset(self) -> None:
        self._history.clear()
        self._rate_limited_until = 0.0

    def _messages(self, masked_text: str) -> list[dict[str, str]]:
        # The moment is stamped on every request rather than once at start
        # up, because this device is left running for days.
        return [
            {
                "role": "system",
                "content": f"{self.settings.system_prompt} {ambient_context()}",
            },
            *self._history,
            {"role": "user", "content": masked_text},
        ]

    def _remember(self, masked_text: str, reply: str) -> None:
        self._history.append({"role": "user", "content": masked_text})
        self._history.append({"role": "assistant", "content": reply})
        excess = len(self._history) - MAX_HISTORY_TURNS * 2
        if excess > 0:
            del self._history[:excess]

    def _model_for_now(self) -> str:
        """The better model, unless it refused us recently."""

        if time.monotonic() < self._rate_limited_until:
            return self.settings.fallback_model
        return self.settings.model

    def reply_to(self, masked_text: str) -> str:
        """Answer one masked utterance, or raise on a transport failure.

        A refusal from one model is not a failure to answer. The device
        drops to the cheaper model and replies, because somebody standing
        in front of it asked a question and a rate limit is not an answer.
        """

        text = (masked_text or "").strip()
        if not text:
            return ""
        if not self.settings.configured:
            raise LLMUnavailable(
                "GROQ_API_KEY is not set, so Onkith cannot answer."
            )

        model = self._model_for_now()
        try:
            reply = self._ask(model, text)
        except RateLimited:
            if not self.settings.has_fallback or model == self.settings.fallback_model:
                raise
            self._rate_limited_until = (
                time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS
            )
            self.log(
                f"{model} is rate limited; answering with "
                f"{self.settings.fallback_model} for the next "
                f"{int(RATE_LIMIT_COOLDOWN_SECONDS)}s"
            )
            reply = self._ask(self.settings.fallback_model, text)

        if reply:
            self._remember(text, reply)
        return reply

    def _ask(self, model: str, text: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": self._messages(text),
                "max_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Onkith/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds
            ) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = (
                f"Groq returned HTTP {error.code}: "
                f"{redact(detail, self.settings.api_key)}"
            )
            # Told apart from every other failure because it is the one
            # that another model can still answer.
            if error.code == 429:
                raise RateLimited(message) from error
            raise RuntimeError(message) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Groq is unreachable: {error.reason}") from error
        except (TimeoutError, OSError) as error:
            raise RuntimeError(f"Groq request failed: {error}") from error

        return extract_reply(body)


def extract_reply(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def redact(text: str, secret: str) -> str:
    """Keep an API key out of any message that might be displayed."""

    if secret and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text
