"""
onkith_ui.py  |  live terminal UI for the Onkith on-device pipeline

Zero dependencies. Pure ANSI. Safe on a Raspberry Pi 5 over SSH.

    from onkith_ui import quiet_boot
    quiet_boot()                      # call BEFORE importing onnxruntime / transformers

    from onkith_ui import LiveUI
    with LiveUI() as ui:
        ui.banner(system={...}, config={...})
        with ui.task("moonshine medium streaming"):
            model = load_model()
        ui.ready(threshold=0.01707)
        ui.set_state("listening")
        ui.set_level(rms)                       # call from the audio callback
        ui.utterance(masked_text, entities={"GIVENNAME": 1}, latency={...})

Palette and rules
-----------------
Locked Onkith tokens only: --void #080706, --surface #131110, --hairline
#252220, --ink-muted #8D8781, --hold #F5F2EE, --lift #FF5A36, --lift-dim
#A85337. One derived value, ink-dim, is a blend of --ink-muted over --void
because a terminal needs a third text level; swap in your real --ink-3 if
the web brand already defines one.

The brand rule is enforced literally: --lift appears on masked spans and on
masking failures, and nowhere else. Everything else on screen is greyscale,
so every spot of colour marks a place where data would have left the device.

The bottom 3 rows are a pinned status bar drawn by a background thread at
14 fps. Everything else scrolls above it inside a DECSTBM scroll region, so
ordinary print() from the rest of the pipeline still works and still scrolls
in the right place. On exit the region is released and the cursor restored.

If stdout is not a tty (piped to a file, run under nohup, CI), the whole
thing degrades to plain unstyled lines with no threads and no escape codes.
"""

from __future__ import annotations

import atexit
import itertools
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager

__all__ = ["LiveUI", "quiet_boot", "muted_stderr", "THEMES"]


# ---------------------------------------------------------------------------
# boot noise suppression
# ---------------------------------------------------------------------------

def quiet_boot(ort_severity: int = 3) -> None:
    """Silence the third-party startup spam.

    Kills, in order: the onnxruntime GPU device-discovery warnings, the
    "PyTorch was not found" advisory from transformers, TF chatter, and
    generic Python warnings.

    Must run BEFORE onnxruntime / transformers are imported, because some of
    these are read at import time.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ORT_LOGGING_LEVEL", str(ort_severity))
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    import logging
    import warnings

    warnings.filterwarnings("ignore")
    for name in ("transformers", "onnxruntime", "numba", "urllib3", "moonshine_onnx"):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:  # only if already importable; harmless if not installed yet
        import onnxruntime as ort
        ort.set_default_logger_severity(ort_severity)  # 0 verbose .. 4 fatal
    except Exception:
        pass


@contextmanager
def muted_stderr():
    """Swallow C-level stderr for the duration of the block.

    ALSA, JACK and onnxruntime write straight to fd 2 from native code, so
    Python-level logging filters never see them. Wrap PyAudio init and ORT
    session creation in this.
    """
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(devnull)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# capability detection
# ---------------------------------------------------------------------------

def _truecolor() -> bool:
    if os.name == "nt":
        return True
    if os.environ.get("COLORTERM", "") in ("truecolor", "24bit"):
        return True
    term = os.environ.get("TERM", "")
    return "256" in term or "kitty" in term or "alacritty" in term or term.startswith("xterm")


def _unicode_ok() -> bool:
    if os.environ.get("OMIT_UI_ASCII"):
        return False
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


ESC = "\x1b"
CSI = ESC + "["
RESET = CSI + "0m"
BOLD = CSI + "1m"
DIM = CSI + "2m"
ITALIC = CSI + "3m"


def _fg(rgb, truecolor=True) -> str:
    r, g, b = rgb
    if truecolor:
        return f"{CSI}38;2;{r};{g};{b}m"
    return f"{CSI}38;5;{_cube(r, g, b)}m"


def _bg(rgb, truecolor=True) -> str:
    r, g, b = rgb
    if truecolor:
        return f"{CSI}48;2;{r};{g};{b}m"
    return f"{CSI}48;5;{_cube(r, g, b)}m"


def _cube(r, g, b) -> int:
    if abs(r - g) < 12 and abs(g - b) < 12:
        return 232 + min(23, int(r / 255 * 23))
    q = lambda v: int(round(v / 255 * 5))
    return 16 + 36 * q(r) + 6 * q(g) + q(b)


def _blend(fg, bg, alpha):
    return tuple(int(round(f * alpha + b * (1 - alpha))) for f, b in zip(fg, bg))


# ---------------------------------------------------------------------------
# themes
# ---------------------------------------------------------------------------

class Theme:
    def __init__(self, name, canvas, accent, accent_dim, ink, muted, faint,
                 ok, warn, bad, entities, uniform_entity=False):
        self.name = name
        self.canvas = canvas
        self.accent = accent
        self.accent_dim = accent_dim
        self.ink = ink
        self.muted = muted
        self.faint = faint
        self.ok = ok
        self.warn = warn
        self.bad = bad
        self.entities = entities
        self.uniform_entity = uniform_entity

    def entity_color(self, label: str):
        if self.uniform_entity:
            return self.accent
        return self.entities.get(_family(label), self.entities["other"])


_FAMILY = {
    "person": {"GIVENNAME", "SURNAME", "TITLE", "USERNAME", "GENDER", "SEX", "AGE", "WEIGHT", "ALLERGIES"},
    "place": {"STREET", "CITY", "COUNTRY", "ZIPCODE", "BUILDINGNUM", "TIMEZONE"},
    "time": {"DATE", "TIME", "DOB"},
    "contact": {"TELEPHONENUM", "EMAIL", "URL", "IPV4"},
    "id": {"SOCIALNUM", "IDCARDNUM", "PASSPORTNUM", "DRIVERLICENSENUM", "TAXNUM",
           "ACCOUNTNUM", "CREDITCARDNUMBER"},
    "money": {"AMOUNT", "CURRENCY", "SALARY", "BANKNAME"},
    "org": {"ORGANISATION"},
}


def _family(label: str) -> str:
    label = label.upper()
    for fam, members in _FAMILY.items():
        if label in members:
            return fam
    return "other"


VOID      = (8, 7, 6)         # --void      #080706
SURFACE   = (19, 17, 16)      # --surface   #131110
HAIRLINE  = (37, 34, 32)      # --hairline  #252220
INK_MUTED = (141, 135, 129)   # --ink-muted #8D8781
HOLD      = (245, 242, 238)   # --hold      #F5F2EE  data that stayed home
LIFT      = (255, 90, 54)     # --lift      #FF5A36  data that would have left
LIFT_DIM  = (168, 83, 55)     # --lift-dim  #A85337
INK_DIM   = _blend(INK_MUTED, VOID, 0.42)   # derived: third text level

THEMES = {
    # Default. Greyscale chrome, vermillion reserved for masked spans.
    "onkith": Theme(
        name="onkith",
        canvas=VOID,
        accent=LIFT,
        accent_dim=LIFT_DIM,
        ink=HOLD,
        muted=INK_MUTED,
        faint=INK_DIM,
        ok=HOLD,
        warn=LIFT_DIM,
        bad=LIFT,
        entities={"other": LIFT},
        uniform_entity=True,
    ),
    # Same tokens, lifted two steps for a washed-out projector or a bright
    # room on demo day. Nothing new is invented: faint climbs to --ink-muted
    # and rules climb off --hairline, which disappears under projector gamma.
    "bright": Theme(
        name="bright",
        canvas=VOID,
        accent=LIFT,
        accent_dim=LIFT_DIM,
        ink=(255, 255, 255),
        muted=(190, 185, 179),
        faint=INK_MUTED,
        ok=(255, 255, 255),
        warn=LIFT_DIM,
        bad=LIFT,
        entities={"other": LIFT},
        uniform_entity=True,
    ),
}


# ---------------------------------------------------------------------------
# the pet
# ---------------------------------------------------------------------------
# Kith covers what would have left. Sitr, in five characters. Each time spans
# are masked the block sweeps across its face and the counter goes up.

PET_UNI = {
    "boot":      ["(-.-)", "(-.-)", "(-.-)", "(o.o)"],
    "idle":      ["(-.-)", "(-.-)", "(-.-)", "(-.-)", "(-.-)", "(-.-)", "(-.-)", "(－.－)"],
    "listening": ["(o.o)", "(o.o)", "(o.o)", "(o.o)", "(o.o)", "(-.-)"],
    "speech":    ["(O.O)", "(O.O)", "(O.O)", "(O‿O)"],
    "thinking":  ["(o.o)", "(.oo)", "(oo.)", "(o.o)"],
    "masking":   ["(█.o)", "(█▌o)", "(██▌)", "(███)"],
    "buffer":    ["(o.o)", "(o.o)", "(o.o)", "(-.-)"],
    "cover":     ["(█.o)", "(██o)", "(███)", "(███)", "(-█-)", "(-.-)"],
    "error":     ["(×_×)"],
}

PET_ASCII = {
    "boot":      ["(-.-)", "(-.-)", "(-.-)", "(o.o)"],
    "idle":      ["(-.-)", "(-.-)", "(-.-)", "(-.-)", "(-.-)", "(-.-)", "(-.-)", "(_._)"],
    "listening": ["(o.o)", "(o.o)", "(o.o)", "(o.o)", "(o.o)", "(-.-)"],
    "speech":    ["(O.O)", "(O.O)", "(O.O)", "(O_O)"],
    "thinking":  ["(o.o)", "(.oo)", "(oo.)", "(o.o)"],
    "masking":   ["(#.o)", "(#|o)", "(##|)", "(###)"],
    "buffer":    ["(o.o)", "(o.o)", "(o.o)", "(-.-)"],
    "cover":     ["(#.o)", "(##o)", "(###)", "(###)", "(-#-)", "(-.-)"],
    "error":     ["(x_x)"],
}

WORKING_WORDS = ["transcribing", "reading", "parsing", "listening in"]
MASKING_WORDS = ["covering", "masking", "redacting", "scrubbing"]


class _Glyphs:
    def __init__(self, unicode_ok: bool):
        u = self.unicode = unicode_ok
        self.sigil = "▌" if u else "|"
        self.rule = "─" if u else "-"
        self.dot = "·" if u else "."
        self.check = "✓" if u else "+"
        self.cross = "✕" if u else "x"
        self.warn = "▲" if u else "!"
        self.arrow = "›" if u else ">"
        self.bar_l = "▏" if u else "|"
        self.blocks = "▁▁▂▃▄▅▆▇█" if u else "..::-=+*#"
        self.spin = "▏▎▍▌▋▊▉█▉▊▋▌▍▎" if u else ".oOOOOo."
        self.prog_full = "━" if u else "="
        self.prog_empty = "━" if u else "-"


# ---------------------------------------------------------------------------
# line builder (tracks visible width while accumulating escape codes)
# ---------------------------------------------------------------------------

class _Line:
    """Holds (style, text) pairs so visible width stays knowable and the line
    can be truncated without slicing through an escape sequence."""

    __slots__ = ("items", "w")

    def __init__(self):
        self.items = []
        self.w = 0

    def add(self, text, style=""):
        if text:
            self.items.append((style, text))
            self.w += len(text)
        return self

    def extend(self, other):
        self.items.extend(other.items)
        self.w += other.w
        return self

    def pad_to(self, n):
        if n > self.w:
            self.items.append(("", " " * (n - self.w)))
            self.w = n
        return self

    def render(self, width=None):
        out, used = [], 0
        for style, text in self.items:
            if width is not None:
                if used >= width:
                    break
                if used + len(text) > width:
                    text = text[: width - used]
            out.append(f"{style}{text}{RESET}" if style else text)
            used += len(text)
        return "".join(out)


# ---------------------------------------------------------------------------
# the UI
# ---------------------------------------------------------------------------

CHIP_RE = re.compile(r"\[([A-Z][A-Z0-9]*)_(\d+)\]")

STATE_LABEL = {
    "boot": "starting up",
    "idle": "idle",
    "listening": "listening",
    "speech": "voice detected",
    "buffer": "holding for context",
    "thinking": None,      # cycles WORKING_WORDS
    "masking": None,       # cycles MASKING_WORDS
    "cover": "masked",
    "error": "error",
}


class LiveUI:
    STATUS_H = 3

    def __init__(self, theme="onkith", motion=True, pet=True, fps=14,
                 stream=None, log_file=None, verbose=False, name="kith"):
        self.stream = stream or sys.stdout
        self.theme = THEMES[theme] if isinstance(theme, str) else theme
        self.pet_name = name
        self.verbose = verbose

        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if os.environ.get("OMIT_UI") == "plain" or os.environ.get("NO_COLOR"):
            self.tty = False
        self.color = self.tty
        self.truecolor = _truecolor()
        self.g = _Glyphs(_unicode_ok() and self.tty)
        self.motion = motion and self.tty and os.environ.get("OMIT_UI_MOTION") != "0"
        self.pet_on = pet and self.tty

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = 0
        self._t0 = time.time()

        self._state = "boot"
        self._state_since = time.time()
        self._sticky = None            # (state, until_ts)
        self._detail = ""
        self._progress = None          # (label, frac) or None
        self._levels = deque(maxlen=160)
        self._threshold = 0.0
        self._masked = 0
        self._utterances = 0
        self._last = {}                # last latency dict
        self._word_i = 0
        self._word_t = 0.0

        self._rows, self._cols = self._size()
        self._active = False
        self._logf = open(log_file, "a", buffering=1) if log_file else None

    # -- terminal plumbing --------------------------------------------------

    def _size(self):
        try:
            sz = shutil.get_terminal_size(fallback=(100, 30))
            return max(sz.lines, 10), max(sz.columns, 40)
        except Exception:
            return 30, 100

    def _w(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass

    def start(self):
        if self._active:
            return self
        self._active = True
        if not self.tty:
            return self
        self._rows, self._cols = self._size()
        self._w(f"{CSI}?25l")                                   # hide cursor
        self._w(f"{CSI}2J{CSI}H")                               # clear
        self._w(f"{CSI}1;{self._rows - self.STATUS_H}r")        # scroll region
        self._w(f"{CSI}{self._rows - self.STATUS_H};1H")        # park cursor in it
        atexit.register(self.stop)
        try:
            signal.signal(signal.SIGWINCH, self._on_resize)
        except Exception:
            pass
        if self.motion:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            self._paint()
        return self

    def _on_resize(self, *_):
        with self._lock:
            self._rows, self._cols = self._size()
            self._w(f"{CSI}1;{self._rows - self.STATUS_H}r")
            self._paint()

    def stop(self):
        if not self._active:
            return
        self._active = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self.tty:
            with self._lock:
                self._w(f"{CSI}r")                              # release region
                top = self._rows - self.STATUS_H + 1
                for i in range(self.STATUS_H):
                    self._w(f"{CSI}{top + i};1H{CSI}2K")
                self._w(f"{CSI}{top};1H{CSI}?25h{RESET}")
                self._flush()
        if self._logf:
            self._logf.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        if exc and exc[0] is KeyboardInterrupt:
            self.set_state("idle")
            self.info(f"stopped {self.g.dot} {self._masked} spans masked "
                      f"across {self._utterances} utterances")
        self.stop()
        return False

    def _flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

    # -- styling helpers ----------------------------------------------------

    def c(self, rgb):
        return _fg(rgb, self.truecolor) if self.color else ""

    def cbg(self, rgb):
        return _bg(rgb, self.truecolor) if self.color else ""

    def _r(self):
        return RESET if self.color else ""

    @property
    def masked_spans(self):
        """Spans masked this session."""
        return self._masked

    @property
    def utterances(self):
        return self._utterances

    # -- writing above the status bar --------------------------------------

    def write(self, s=""):
        """Emit one scrolling line above the status bar."""
        with self._lock:
            self._w(s + "\n")
            self._flush()

    def _tofile(self, s):
        if self._logf:
            self._logf.write(f"{time.strftime('%H:%M:%S')} {s}\n")

    # -- boot output --------------------------------------------------------

    def banner(self, system=None, config=None, title="onkith",
               subtitle="on-device privacy pipeline"):
        t, g = self.theme, self.g
        self.write()
        line = _Line()
        line.add(f" {g.sigil} ", self.c(t.accent))
        line.add(title, BOLD + self.c(t.ink) if self.color else "")
        line.add(f"  {g.dot}  ", self.c(t.faint))
        line.add(subtitle, self.c(t.muted))
        self.write(line.render())
        if system:
            bits = f"   {(' ' + g.dot + ' ').join(str(v) for v in system.values())}"
            self.write(self.c(t.faint) + bits + self._r())
        self.write()
        if config:
            klen = max(len(k) for k in config)
            for k, v in config.items():
                l = _Line()
                l.add("   " + k.ljust(klen + 2), self.c(t.muted))
                l.add(str(v), self.c(t.ink))
                self.write(l.render())
            self.write()
        self._tofile(f"boot {system} {config}")

    def rule(self, label=""):
        t, g = self.theme, self.g
        width = self._cols - 4
        if label:
            head = f" {label} "
            bar = g.rule * 3 + head + g.rule * max(0, width - len(head) - 3)
        else:
            bar = g.rule * width
        self.write("  " + self.c(t.faint) + bar[:width] + self._r())

    # -- log lines ----------------------------------------------------------

    def _tagline(self, glyph, gcolor, text, detail=None, tcolor=None):
        t = self.theme
        l = _Line()
        l.add(f"  {glyph} ", self.c(gcolor))
        l.add(text, self.c(tcolor or t.ink))
        if detail:
            l.add(f"  {self.g.dot} {detail}", self.c(t.faint))
        self.write(l.render())

    def ok(self, text, detail=None):
        self._tagline(self.g.check, self.theme.ok, text, detail)
        self._tofile(f"ok {text} {detail or ''}")

    def info(self, text, detail=None):
        self._tagline(self.g.dot, self.theme.muted, text, detail, tcolor=self.theme.muted)
        self._tofile(f"info {text}")

    def step(self, text, detail=None):
        self._tagline(self.g.sigil, self.theme.accent, text, detail)
        self._tofile(f"step {text}")

    def warn(self, text, detail=None):
        self._tagline(self.g.warn, self.theme.warn, text, detail)
        self._tofile(f"warn {text}")

    def error(self, text, detail=None):
        self._tagline(self.g.cross, self.theme.bad, text, detail)
        self.set_state("error", sticky=2.0)
        self._tofile(f"error {text}")

    # -- long operations ----------------------------------------------------

    @contextmanager
    def task(self, label, done_label=None):
        """Spinner in the status bar while the block runs, tick line after."""
        prev = self._state
        self.set_state("thinking", detail=label)
        t0 = time.time()
        try:
            yield
        except Exception:
            self.error(f"{label} failed")
            raise
        else:
            self.ok(done_label or label, f"{time.time() - t0:.1f}s")
        finally:
            self.set_state(prev)

    def progress(self, label, frac):
        with self._lock:
            self._progress = (label, max(0.0, min(1.0, frac)))
        if not self.motion:
            self._paint()

    def clear_progress(self):
        with self._lock:
            self._progress = None

    # -- live state ---------------------------------------------------------

    def set_state(self, state, detail="", sticky=None):
        with self._lock:
            if sticky:
                self._sticky = (state, time.time() + sticky)
            self._state = state
            self._state_since = time.time()
            self._detail = detail
        if not self.motion:
            self._paint()

    def set_level(self, rms):
        """Feed the mic meter. Safe to call from the audio thread."""
        try:
            self._levels.append(float(rms))
        except (TypeError, ValueError):
            pass

    def ready(self, threshold=None):
        self._threshold = threshold or 0.0
        d = f"threshold {threshold:.5f}" if threshold else None
        self._tagline(self.g.arrow, self.theme.accent, "listening continuously", d)
        self.set_state("listening")
        self.rule()

    # -- the main event: a masked utterance --------------------------------

    def utterance(self, text, entities=None, latency=None, ts=None):
        """Render one finalized, masked user turn."""
        t = self.theme
        entities = entities or {}
        latency = latency or {}

        self.write()
        line = _Line()
        line.add("  You      ", BOLD + self.c(t.muted) if self.color else "")
        for seg, label in self._chips(text):
            if label is None:
                line.add(seg, self.c(t.ink))
            elif self.color:
                col = t.entity_color(label)
                line.add(seg, self.cbg(_blend(col, t.canvas, 0.22)) + self.c(col) + BOLD)
            else:
                line.add(seg)
        self.write(line.render())

        n = sum(entities.values()) if entities else 0
        self._masked += n
        self._utterances += 1
        self._last = latency
        if n:
            self.set_state("cover", sticky=1.4)
        self._tofile(f"utterance {text!r} entities={entities} latency={latency}")

    def onkith(self, text):
        """Render one user-facing Onkith turn in the theme accent."""
        t = self.theme
        message = str(text).strip()
        if not message:
            return
        self.write()
        line = _Line()
        accent = self.c(t.accent)
        line.add("  Onkith   ", BOLD + accent if self.color else "")
        line.add(message, accent)
        self.write(line.render())
        self._tofile(f"onkith {message!r}")

    def _chips(self, text):
        """Split into (segment, label|None) runs so placeholders can be styled."""
        out, i = [], 0
        for m in CHIP_RE.finditer(text):
            if m.start() > i:
                out.append((text[i:m.start()], None))
            out.append((m.group(0), m.group(1)))
            i = m.end()
        if i < len(text):
            out.append((text[i:], None))
        return out or [(text, None)]

    # -- status bar ---------------------------------------------------------

    def _loop(self):
        interval = 1.0 / 14
        while not self._stop.is_set():
            self._frame += 1
            try:
                self._paint()
            except Exception:
                pass
            self._stop.wait(interval)

    def _effective_state(self):
        if self._sticky:
            state, until = self._sticky
            if time.time() < until:
                return state
            self._sticky = None
        return self._state

    def _pet(self, state):
        table = PET_UNI if self.g.unicode else PET_ASCII
        frames = table.get(state, table["idle"])
        speed = 3 if state in ("cover", "thinking", "masking") else 6
        return frames[(self._frame // speed) % len(frames)]

    def _label(self, state):
        if self._detail and state in ("thinking", "masking"):
            return self._detail
        fixed = STATE_LABEL.get(state)
        if fixed:
            return fixed
        pool = MASKING_WORDS if state == "masking" else WORKING_WORDS
        now = time.time()
        if now - self._word_t > 2.4:
            self._word_t = now
            self._word_i += 1
        return pool[self._word_i % len(pool)]

    def _meter(self, width):
        t, g = self.theme, self.g
        blocks = g.blocks
        if not self._levels:
            return _Line().add(blocks[0] * width, self.c(t.faint))
        vals = list(self._levels)[-width:]
        vals = [0.0] * (width - len(vals)) + vals
        peak = max(max(vals), self._threshold * 3, 1e-6)
        line = _Line()
        for v in vals:
            idx = int(min(0.999, v / peak) * (len(blocks) - 1))
            hot = v >= self._threshold and self._threshold > 0
            line.add(blocks[idx], self.c(t.ink if hot else t.faint))
        return line

    def _paint(self):
        if not self.tty:
            return
        t, g = self.theme, self.g
        rows, cols = self._rows, self._cols
        top = rows - self.STATUS_H + 1
        state = self._effective_state()

        # row 1: hairline
        r1 = _Line().add(g.rule * cols, self.c(t.faint))

        # row 2: pet, state, and the existing microphone meter
        r2 = _Line()
        r2.add(" ")
        if self.pet_on:
            r2.add(g.sigil + " ", self.c(t.accent))
            r2.add(self._pet(state) + " ", self.c(t.accent if state == "cover" else t.ink))
        else:
            r2.add(g.sigil + " ", self.c(t.accent))

        if self._progress:
            label, frac = self._progress
            barw = 18
            filled = int(round(frac * barw))
            r2.add(label + " ", self.c(t.muted))
            r2.add(g.prog_full * filled, self.c(t.accent))
            r2.add(g.prog_empty * (barw - filled), self.c(t.faint))
            r2.add(f" {int(frac * 100):3d}%", self.c(t.muted))
        else:
            spinning = state in ("thinking", "masking", "boot")
            if spinning:
                r2.add(g.spin[self._frame % len(g.spin)] + " ", self.c(t.accent))
            else:
                r2.add("  ")
            label = self._label(state)
            lcol = t.accent if state in ("speech", "cover") else t.muted
            r2.add(label, self.c(lcol))
            if state == "listening" and self.motion:
                r2.add("." * (1 + (self._frame // 7) % 3), self.c(t.faint))

        left_w = 32 if cols >= 88 else max(r2.w + 2, 22)
        mid = _Line().extend(r2).pad_to(left_w)
        meter_w = min(30, cols - left_w - 2)
        if meter_w >= 8:
            mid.extend(self._meter(meter_w))
        mid.pad_to(max(mid.w, cols - 1))

        # row 3: one quiet control hint
        r3 = _Line()
        r3.add("  ctrl-c to stop", self.c(t.faint))

        with self._lock:
            out = [f"{ESC}7"]
            for i, ln in enumerate((r1, mid, r3)):
                out.append(f"{CSI}{top + i};1H{CSI}2K")
                out.append(ln.render(cols))
            out.append(f"{ESC}8")
            self._w("".join(out))
            self._flush()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("onkith_ui is a library. Run ui_demo.py to see it.")
