"""OnKith V6 integrated round-trip benchmark: audio in, spoken answer out.

``Full_Pipeline_Audio_Benchmark.py`` stops at the sanitized transcript and
``Full_Pipeline_CPU_Benchmark.py`` never leaves the device. Both measure the
half of V6 that is local. This one measures the half that is the point:

    audio -> Whisper -> DeBERTa masking -> placeholder map (LOCAL)
          -> masked request -> Groq -> masked reply
          -> local rehydration -> restored answer -> local TTS

What it is for
--------------
It is an **integration and latency benchmark**, not a quality benchmark. The
recognition and privacy numbers a report should quote come from the audio
benchmark, which scores 44 clips against gold annotations. This one answers four
questions that benchmark structurally cannot:

1. Does a raw private value ever reach the outbound request? The check reads the
   bytes handed to ``urllib`` inside ``app.llm``, not a reconstruction of them,
   so a masking failure, a debug field or a stray history entry would all be
   caught by the same test.
2. Do the placeholders survive the model? A reply that turns ``[GIVENNAME_1]``
   into ``[Name]`` cannot be rehydrated, and the person hears a bracket read
   aloud.
3. Does local rehydration put the right value back, including when a
   placeholder repeats, is unknown, or is malformed?
4. What does the whole thing cost, with the network round trip reported
   **separately** from the local pipeline, because one of them is this
   project's engineering and the other is somebody's wifi.

Running without an API key
--------------------------
Supported and useful. Everything local runs, the outbound payload is still
built by the real client and still audited for leaks, and the rehydration
probes still run against locally constructed replies. The live sections then
report ``not run`` rather than a number. Nothing here ever invents an API
response: a section with no measurement says so.

    python Full_Pipeline_V6_EndToEnd_Benchmark.py               6 live cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --extended    10 live cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --smoke-test  3 live cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --llm-stub    no key needed
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --no-tts
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --help

Staying inside the free tier
----------------------------
Requests per minute is not the binding limit here. The system prompt is long,
and the client keeps a rolling conversation history, so a handful of rapid
requests can reach the tokens-per-minute ceiling while sitting far below the
request ceiling. Two things follow, and both are defaults rather than options:
the conversation is reset between cases, because these are independent
measurements and carrying five previous exchanges into the sixth request
inflates its token count for no benefit; and requests are spaced, guided by the
rate-limit headers Groq returns rather than by a fixed guess.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import benchmark_common as common
from .benchmark_common import RunDirectory, WarningLog, print_summary, summarise

FIXTURE_DIR = common.EVALUATION_ROOT / "fixtures" / "audio_benchmark"
MANIFEST_PATH = FIXTURE_DIR / "audio_benchmark.manifest.json"

#: Six representative cases, one per privacy family that matters, and they are
#: what a default run sends to Groq. This is not a sample size chosen for
#: statistics - the 44-clip audio benchmark is where accuracy is measured, on
#: gold annotations, without spending a request. What these six establish is
#: that the integration works end to end and what each leg costs, and six is
#: enough for that while leaving the free tier's daily budget alone.
DEFAULT_CASES: tuple[tuple[str, str], ...] = (
    ("privacy_001", "person name"),
    ("privacy_003", "telephone number"),
    ("privacy_004", "email address, spoken form"),
    ("privacy_007", "date / numeric private value"),
    ("privacy_020", "multiple PII types in one utterance"),
    ("benign_003", "no PII, quantities that must not be masked"),
)

#: ``--extended``. Four more, still small: a second speaker, a repeated name
#: beside a telephone number, a long identifier, and the longest clip in the
#: corpus. Ten live requests remains a small fraction of a day's budget.
EXTENDED_CASES: tuple[tuple[str, str], ...] = DEFAULT_CASES + (
    ("privacy_002", "given name + surname, second speaker"),
    ("privacy_012", "name repeated with a telephone number"),
    ("privacy_016", "identity card number"),
    ("privacy_024", "seven entities, longest clip"),
)

#: One clip the device recognises as an instruction to itself. It goes through
#: the whole local path and is then routed locally, so it costs no API quota -
#: which is the property being checked, not a latency number.
LOCAL_COMMAND_CASE = ("benign_001", "no PII, device command - must not call Groq")

SMOKE_CASES = ("privacy_001", "privacy_003", "benign_003")

#: Seconds between live requests when the response headers give no better
#: guidance. Six requests at this spacing finish in about a minute and stay
#: comfortably inside both the request and the token ceiling. The headers are
#: preferred where Groq sends them; this is the fallback, not the policy.
DEFAULT_REQUEST_PAUSE = 13.0

#: The longest a 429's ``Retry-After`` will be honoured. Beyond this the case is
#: recorded as rate limited and the run moves on: waiting out a daily quota is
#: not a benchmark, it is an outage.
MAX_RETRY_AFTER_SECONDS = 60.0

#: How many restored answers get spoken when TTS is measured. Synthesis is
#: real-time by construction, so timing every case would spend minutes proving
#: something five cases already establish.
DEFAULT_TTS_CASES = 5

WARMUP_TEXT = (
    "Hi, my name is Sarah Walusiak, my number is 555 0147 and my email is "
    "sarah.walusiak@example.org. Remind me about the appointment on 12 March."
)

#: Used only when no real key is configured, so the request the client builds
#: can still be inspected. It is never sent anywhere.
OFFLINE_AUDIT_KEY = "OFFLINE-PAYLOAD-AUDIT-NOT-A-KEY"

#: Below this many characters, a recognizer surface is not evidence of
#: anything. The alignment that locates what the recognizer produced for a gold
#: entity occasionally returns a fragment - it returned "at" for a spoken email
#: address - and a two-letter English word is present in almost any sentence,
#: so testing for it reports a leak where there is none. Such a surface is
#: counted and named in its own field rather than quietly passed or quietly
#: failed: the alignment was unreliable, which is a third thing and should read
#: as one.
MIN_TESTABLE_SURFACE = 4

PLACEHOLDER = re.compile(r"\[([A-Z][A-Z0-9_]*)_(\d+)\]")
#: Anything bracket-shaped, so a *corrupted* placeholder is visible too. The
#: model turning "[GIVENNAME_1]" into "[Name]" is the failure this catches, and
#: a pattern that only matched well-formed tokens would score it as silence.
ANY_BRACKET = re.compile(r"\[[^\[\]\n]{1,64}\]")


class FixtureRejected(RuntimeError):
    """A committed clip is missing or does not match its manifest hash."""


class PayloadAudited(Exception):
    """Stop before the socket, having recorded the request that was built.

    Deliberately not an ``OSError``/``URLError``/``HTTPError``: those are the
    three families ``GroqClient._ask`` catches and rewrites, and this has to
    reach the benchmark unchanged so an audited case is never mistaken for a
    network failure.
    """


# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------


def normalise_for_leak_check(text: str) -> str:
    """Lowercase, punctuation-free, spoken numbers folded to numerals.

    The same normalisation the engine-comparable WER uses, and for the same
    reason. A gold entity spoken "five five five, zero one four seven" arrives
    from Whisper as "555-0147"; a substring search for the reference surface
    would find neither in the other and would report a clean payload for a
    number sitting in plain sight. Folding both sides is what makes the leak
    check mean what it says.
    """

    from .stt_comparison import normalise_numbers

    return normalise_numbers(text or "")


def contains_value(haystack_folded: str, value: str) -> bool:
    """Is one private value present in already-folded text?

    Word-bounded on both ends, so "142" inside "1425" is not reported as the
    building number and a two-letter surname does not match half a verb.
    """

    needle = normalise_for_leak_check(value).strip()
    if not needle:
        return False
    return re.search(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])",
                     haystack_folded) is not None


def bracket_tokens(text: str) -> list[str]:
    return ANY_BRACKET.findall(text or "")


# --------------------------------------------------------------------------
# Stubbed transport
# --------------------------------------------------------------------------


class _StubResponse:
    """Enough of an HTTP response for ``json.load`` and a ``with`` block."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, *_args: Any) -> bytes:
        body, self._body = self._body, b""
        return body

    def __enter__(self) -> "_StubResponse":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def stub_reply_for(masked_text: str) -> str:
    """A cooperative answer that echoes every placeholder it was given.

    Deliberately the *best* case for the round trip, and labelled as such
    everywhere it is reported. What it proves is that the local half works: the
    request is built and sent, the response is parsed, the placeholders come
    back through ``GroqClient``'s own history, and rehydration restores the
    values. What it cannot prove is whether a real model would have preserved
    them, which is why placeholder preservation is reported as not measured
    unless a live key was used.
    """

    tokens = sorted({f"[{a}_{b}]" for a, b in PLACEHOLDER.findall(masked_text)})
    if not tokens:
        return "Sure, I have made a note of that."
    if len(tokens) == 1:
        return f"Sure, I will remember {tokens[0]} and remind you about {tokens[0]}."
    listed = ", ".join(tokens[:-1]) + f" and {tokens[-1]}"
    return f"Sure, I have noted {listed}."


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FixtureRejected(f"audio benchmark manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_cases(manifest: Mapping[str, Any], wanted: Sequence[str]
                 ) -> list[dict[str, Any]]:
    by_id = {clip["clip_id"]: clip for clip in manifest["clips"]}
    missing = [clip_id for clip_id in wanted if clip_id not in by_id]
    if missing:
        raise FixtureRejected(f"unknown clip id(s): {missing}")
    notes = dict(DEFAULT_CASES)
    return [{**by_id[clip_id], "_case_note": notes.get(clip_id, "")}
            for clip_id in wanted]


# --------------------------------------------------------------------------
# Request capture
# --------------------------------------------------------------------------


@dataclass
class CapturedRequest:
    """Exactly what the LLM client handed to the transport."""

    url: str
    method: str
    header_names: list[str]
    body: dict[str, Any]
    raw_bytes: int
    #: Every string anywhere in the JSON body, so the leak check reads the
    #: whole payload rather than only the fields it expected to find.
    all_strings: list[str] = field(default_factory=list)

    @property
    def searchable(self) -> str:
        return " ".join(self.all_strings)


def _walk_strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key)
            yield from _walk_strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_strings(item)


def snapshot_request(request: Any) -> CapturedRequest:
    raw = request.data or b""
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        body = {"_undecodable": raw.decode("utf-8", errors="replace")}
    return CapturedRequest(
        url=request.full_url,
        method=request.get_method(),
        header_names=sorted(request.headers),
        body=body,
        raw_bytes=len(raw),
        all_strings=list(_walk_strings(body)),
    )


def redacted_body(captured: CapturedRequest) -> dict[str, Any]:
    """The payload as written to the run folder.

    Kept verbatim on purpose. If it were summarised, the artifact would prove
    nothing: a reader has to be able to open the file and look for the private
    value themselves. The Authorization header is not part of the body and is
    recorded by name only.
    """

    return captured.body


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------


@dataclass
class Options:
    cases: list[str]
    mode: str = "full"
    device_label: str = ""
    results_root: Path | None = None
    run_tts: bool = True
    tts_cases: int = DEFAULT_TTS_CASES
    live: bool = True
    stub_llm: bool = False
    request_pause: float = DEFAULT_REQUEST_PAUSE


class EndToEndBenchmark:
    def __init__(self, options: Options) -> None:
        self.options = options
        self.warnings = WarningLog()
        self.completed: dict[str, bool] = {}
        self.started_utc = common.utc_now()
        self.run_id = common.make_run_id(
            self.started_utc, options.mode, options.device_label,
            prefix="e2e_v6_whisper",
        )
        self.run = RunDirectory(self.run_id, options.results_root)

        self.manifest: dict[str, Any] = {}
        self.cases: list[dict[str, Any]] = []
        self.system_info: dict[str, Any] = {}
        self.artifact: dict[str, Any] = {}
        self.config: Any = None

        self.whisper = None
        self.masker = None
        self.llm = None
        self.speech = None
        self.llm_module = None

        self.llm_configured = False
        self.llm_mode = "not run"
        self.tts_backend = "not measured"
        self.baseline_rss: int | None = None
        self.stage_rss: dict[str, int | None] = {}
        self.peak_rss = 0
        self.load_seconds: dict[str, float] = {}

        #: What the last response said about the remaining budget, so the
        #: spacing between requests can follow the account real limits
        #: instead of a number guessed in advance.
        self.rate_limit_headers: dict[str, str] = {}
        self.rate_limited_cases: list[str] = []
        self.local_command_check: dict[str, Any] = {}

        self.rows: list[dict[str, Any]] = []
        self.payloads: list[dict[str, Any]] = []
        self.probe_results: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    # -- setup -------------------------------------------------------------

    def announce(self) -> None:
        self.manifest = load_manifest()
        self.cases = select_cases(self.manifest, self.options.cases)
        self.system_info = common.collect_system_info(self.run_id, self.started_utc)
        self.baseline_rss = self.system_info["baseline_process_rss_bytes"]
        self.peak_rss = self.baseline_rss or 0
        self.stage_rss["baseline"] = self.baseline_rss

        from integrated_pipeline.config import PipelineConfig

        self.config = PipelineConfig(root=common.PRODUCTION_ROOT)
        self.artifact = common.verify_privacy_artifacts(
            self.config, self.run_id, self.warnings
        )

        total = sum(case["duration_seconds"] for case in self.cases)
        info = self.system_info
        print()
        print("OnKith V6 Integrated Round-Trip Benchmark")
        print("=========================================")
        print()
        print(f"Device            : {info['board_model'] or info['platform']}")
        print(f"CPU               : {info['cpu_model']}")
        print(f"Architecture      : {info['architecture']} "
              f"({info['logical_cpus']} logical CPUs)")
        print(f"RAM               : {common.gib(info['total_ram_bytes'])}")
        print(f"Pipeline          : {common.relative(common.PRODUCTION_ROOT)}")
        print(f"STT model         : {self.config.whisper['model']} "
              f"({self.config.whisper['runtime']}, "
              f"{self.config.whisper['device']}/"
              f"{self.config.whisper['compute_type']}, "
              f"beam {self.config.whisper['beam_size']})")
        print(f"Privacy model SHA : {self.artifact['measured_sha256']}")
        print(f"Cases             : {len(self.cases)} "
              f"({total:.1f} s of audio, {self.options.mode} mode)")
        print(f"Run id            : {self.run_id}")
        print(f"Output            : {self.run.relative_path}")
        print()

        if not self.artifact["ok"]:
            raise FixtureRejected(
                "the packaged privacy model is not the validated artifact; "
                "refusing to report a privacy result from it"
            )

    def verify_fixtures(self) -> None:
        from .audio_benchmark import verify_clip

        print("Verifying fixtures ...")
        for case in self.cases:
            case["_path"] = verify_clip(case, fixture_dir=FIXTURE_DIR)
        print(f"  {len(self.cases)} clip(s) match their manifest SHA-256")
        self.completed["fixture_verification"] = True

    def load_components(self) -> None:
        print("\nLoading pipeline ...")
        self._load_stt()
        self._load_privacy()
        self._load_llm()
        self._load_tts()
        print()

    def _load_stt(self) -> None:
        from integrated_pipeline.whisper_adapter import (
            WhisperAdapter,
            describe_whisper_backend,
        )

        backend = describe_whisper_backend()
        if not backend.get("available"):
            raise RuntimeError(
                "faster-whisper backend unavailable: "
                + "; ".join(backend.get("errors", ["unknown"]))
            )
        common.quieten_stt()
        started = time.perf_counter()
        self.whisper = WhisperAdapter(self.config)
        self.whisper.load()
        self.load_seconds["stt"] = time.perf_counter() - started
        self.stage_rss["stt_loaded"] = common.rss_bytes()
        self.stt_backend = backend
        print(f"  STT      whisper {self.config.whisper['model']:16} "
              f"loaded in {self.load_seconds['stt']:.2f} s")

    def _load_privacy(self) -> None:
        from integrated_pipeline.pii_adapter import PIIMaskerAdapter
        from integrated_pipeline.pipeline import FailClosedMasker
        from integrated_pipeline.supplemental_pii import SupplementalPIIMasker

        started = time.perf_counter()
        adapter = PIIMaskerAdapter(self.config)
        adapter.load()
        # The exact composition app/omit_ai_live.py builds. Measuring a
        # different arrangement would measure a pipeline nobody runs.
        self.masker = FailClosedMasker(SupplementalPIIMasker(adapter))
        self.load_seconds["privacy"] = time.perf_counter() - started
        self.stage_rss["privacy_loaded"] = common.rss_bytes()
        self.masker.mask_text(WARMUP_TEXT)
        self.stage_rss["all_models_loaded"] = common.rss_bytes()
        print(f"  Privacy  DeBERTa-v3-xsmall INT8    loaded in "
              f"{self.load_seconds['privacy']:.2f} s")

    def _load_llm(self) -> None:
        from app import llm as llm_module
        from app.omit_ai_live import load_local_env

        # The same loader the live pipeline uses, reading the same repository
        # root .env, so the benchmark and the demonstration cannot end up
        # configured differently. It is setdefault throughout: anything already
        # exported wins, which is how a no-key run is produced without touching
        # the file.
        load_local_env()

        self.llm_module = llm_module
        settings = llm_module.GroqSettings.from_environment()
        self.llm_configured = settings.configured and self.options.live
        if self.llm_configured:
            self.llm_mode = "live"
        elif self.options.stub_llm:
            self.llm_mode = "stubbed transport"
        else:
            self.llm_mode = "payload audit only"

        if not settings.configured:
            self.warnings.note(
                "GROQ_API_KEY is not set: the outbound payload is still built "
                "and audited, but no request is sent and no live latency, "
                "placeholder-preservation or API success rate is reported."
            )
        elif not self.options.live:
            self.warnings.note("--offline: a key is configured but was not used.")
        if self.llm_mode == "stubbed transport":
            self.warnings.note(
                "--llm-stub: the request is built and answered by a local "
                "cooperative responder, not by Groq. Latency, placeholder "
                "preservation and API success rate from this run describe the "
                "integration, not the model, and must not be reported as a "
                "measurement of Groq."
            )
        if not self.llm_configured:
            settings.api_key = OFFLINE_AUDIT_KEY
        self.llm = llm_module.GroqClient(settings=settings)
        print(f"  LLM      {settings.model:24} {self.llm_mode}")

    def _load_tts(self) -> None:
        if not self.options.run_tts:
            self.tts_backend = "disabled with --no-tts"
            print("  TTS      skipped (--no-tts)")
            return
        try:
            from app.speech import detect_backend

            backend = detect_backend(self.warnings.note)
            self.speech = backend
            self.tts_backend = backend.name
            print(f"  TTS      {backend.name:24} local")
        except Exception as exc:
            self.tts_backend = f"unavailable: {type(exc).__name__}: {exc}"
            self.warnings.note(f"TTS unavailable: {self.tts_backend}")
            print("  TTS      UNAVAILABLE")

    # -- one case ----------------------------------------------------------

    def _transcribe(self, audio, clip_id: str) -> tuple[str, float]:
        segment = common.make_canary_segment(
            audio,
            session_id="e2e-benchmark",
            utterance_id=f"e2e-{clip_id}",
            segment_id=f"e2e-{clip_id}-0",
        )
        started = time.perf_counter()
        transcript = self.whisper.transcribe_segment(segment)
        return transcript.text, time.perf_counter() - started

    def _ask(self, masked_text: str) -> dict[str, Any]:
        """One request, with the payload captured on the way to the socket.

        The conversation is reset first. Each case is an independent
        measurement, so carrying the previous five exchanges into this request
        would inflate its token count for no benefit - and the free tier runs
        out of tokens per minute long before it runs out of requests.
        """

        captured: list[CapturedRequest] = []
        module = self.llm_module
        real_urlopen = module.urllib.request.urlopen
        mode = self.llm_mode
        self.llm.reset()

        def spy(request, *args, **kwargs):
            snapshot = snapshot_request(request)
            captured.append(snapshot)
            if mode == "live":
                response = real_urlopen(request, *args, **kwargs)
                self._note_rate_limit_headers(getattr(response, "headers", {}))
                return response
            if mode == "stubbed transport":
                # Answer from the payload that was actually built, so the reply
                # can only echo placeholders the request really carried.
                messages = snapshot.body.get("messages") or []
                asked = ""
                for message in reversed(messages):
                    if isinstance(message, Mapping) and message.get("role") == "user":
                        asked = str(message.get("content") or "")
                        break
                return _StubResponse({
                    "choices": [{"message": {"content": stub_reply_for(asked)}}]
                })
            raise PayloadAudited(request.full_url)

        module.urllib.request.urlopen = spy
        started = time.perf_counter()
        reply = ""
        error = ""
        rate_limited = False
        try:
            reply = self.llm.reply_to(masked_text)
        except PayloadAudited:
            error = "not sent: no API key configured"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rate_limited = self._looks_rate_limited(exc)
        finally:
            module.urllib.request.urlopen = real_urlopen

        seconds = time.perf_counter() - started
        # One retry, once, and only for a refusal a wait can actually fix. The
        # client already drops to the fallback model on a 429, so reaching here
        # means both models refused; a second immediate attempt would only add
        # two more requests to a limit that is already full.
        if rate_limited and mode == "live":
            wait = self._retry_after_seconds()
            if wait is not None and wait <= MAX_RETRY_AFTER_SECONDS:
                print(f"        rate limited; waiting {wait:.0f}s and retrying once")
                time.sleep(wait)
                module.urllib.request.urlopen = spy
                retried_at = time.perf_counter()
                try:
                    reply = self.llm.reply_to(masked_text)
                    error = ""
                    rate_limited = False
                except PayloadAudited:
                    error = "not sent: no API key configured"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    rate_limited = self._looks_rate_limited(exc)
                finally:
                    module.urllib.request.urlopen = real_urlopen
                seconds = time.perf_counter() - retried_at
            else:
                print("        rate limited; recorded as such rather than waited out")

        return {
            "captured": captured,
            "reply": reply,
            "seconds": seconds,
            "error": error,
            "rate_limited": rate_limited,
            # Which model actually answered. A latency figure that silently
            # mixed the primary and the fallback would describe neither.
            "model": captured[-1].body.get("model") if captured else None,
            "models_attempted": [item.body.get("model") for item in captured],
        }

    @staticmethod
    def _looks_rate_limited(error: Exception) -> bool:
        return type(error).__name__ == "RateLimited" or "429" in str(error)

    def _note_rate_limit_headers(self, headers: Any) -> None:
        try:
            items = list(headers.items())
        except Exception:
            return
        for key, value in items:
            lowered = str(key).lower()
            if lowered.startswith("x-ratelimit") or lowered == "retry-after":
                self.rate_limit_headers[lowered] = str(value)

    def _retry_after_seconds(self) -> float | None:
        raw = self.rate_limit_headers.get("retry-after")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _pause_before_next_request(self) -> float:
        """How long to wait, preferring what the account actually reports.

        The remaining-request and remaining-token headers say how much budget
        is left. Where they are present and the budget is comfortable there is
        no reason to sit idle for the full fallback interval; where they are
        absent, the fallback is what keeps six requests inside one minute of
        tokens.
        """

        requests_left = self.rate_limit_headers.get("x-ratelimit-remaining-requests")
        tokens_left = self.rate_limit_headers.get("x-ratelimit-remaining-tokens")
        try:
            if requests_left is not None and tokens_left is not None:
                if int(float(requests_left)) > 10 and int(float(tokens_left)) > 20_000:
                    return min(self.options.request_pause, 3.0)
        except ValueError:
            pass
        return self.options.request_pause

    def _speak(self, text: str) -> float | None:
        if self.speech is None or not text:
            return None
        started = time.perf_counter()
        try:
            self.speech.speak(text)
        except Exception as exc:
            self.warnings.note(f"TTS failed: {type(exc).__name__}: {exc}")
            return None
        return time.perf_counter() - started

    def process_case(self, case: Mapping[str, Any], index: int, total: int
                     ) -> dict[str, Any]:
        from app.rehydrate import (
            RehydrationMap,
            map_from_entities,
            rehydrate,
            unresolved_placeholders,
        )
        from .audio_benchmark import locate_entity, read_wav

        clip_id = case["clip_id"]
        audio = read_wav(case["_path"])

        transcript, stt_seconds = self._transcribe(audio, clip_id)
        started = time.perf_counter()
        pii = self.masker.mask_text(transcript)
        mask_seconds = time.perf_counter() - started
        local_seconds = stt_seconds + mask_seconds

        mapping = map_from_entities(pii.entities)
        request_placeholders = sorted(set(PLACEHOLDER.findall(pii.masked_text)))
        request_tokens = sorted({f"[{a}_{b}]" for a, b in request_placeholders})

        outcome = self._ask(pii.masked_text) if pii.safe_to_release else {
            "captured": [], "reply": "", "seconds": 0.0,
            "error": "not sent: masking was blocked",
            "rate_limited": False, "model": None, "models_attempted": [],
        }
        if outcome.get("rate_limited"):
            self.rate_limited_cases.append(clip_id)
        captured = outcome["captured"]
        reply = outcome["reply"]

        # -- leak audit ----------------------------------------------------
        payload_folded = " ".join(
            normalise_for_leak_check(item.searchable) for item in captured
        )
        masked_values = [
            {"placeholder": key, "value": value,
             "in_payload": bool(captured) and contains_value(payload_folded, value)}
            for key, value in mapping.items()
        ]
        # Two readings of every gold entity, because a recognizer changes the
        # surface. ``value`` is what the manifest says was spoken; ``recognized``
        # is what Whisper actually produced for it, located by the same
        # alignment the audio benchmark uses, and it is the string the privacy
        # model was really given. Checking only the first would report a clean
        # payload for a number the recognizer mangled and the masker then
        # missed - which is exactly the case this corpus contains.
        # Whether the masker found this entity at all is the question that
        # separates the two failures a reader has to be able to tell apart: a
        # value the masker replaced and that reached the payload anyway is a
        # boundary failure, and there must be none of those. A value the masker
        # never detected was never behind the boundary in the first place - it
        # is a privacy-model recall miss, which the 44-clip audio benchmark
        # measures properly, and reporting the two as one number would hide
        # both.
        masked_folded = [
            normalise_for_leak_check(value) for value in mapping.values()
        ]

        def was_masked(text: str) -> bool:
            folded = normalise_for_leak_check(text)
            if not folded.strip():
                return False
            return any(
                folded in candidate or candidate in folded
                for candidate in masked_folded
                if candidate.strip()
            )

        gold_values = []
        for span in case.get("expected_private_spans", []):
            location = locate_entity(case["reference_transcript"], span, transcript)
            surface = location.get("hypothesis_surface", "")
            testable = len(surface.strip()) >= MIN_TESTABLE_SURFACE
            detected = was_masked(span["text"]) or (
                bool(surface.strip()) and was_masked(surface)
            )
            gold_values.append({
                "label": span["label"],
                "value": span["text"],
                "recognized": surface,
                "alignment": location.get("alignment", ""),
                "detected_by_masker": detected,
                "surface_testable": testable,
                "in_payload": bool(captured)
                and contains_value(payload_folded, span["text"]),
                "recognized_in_payload": bool(captured) and testable
                and contains_value(payload_folded, surface),
            })
        transcript_in_payload = bool(captured) and bool(transcript.strip()) and (
            normalise_for_leak_check(transcript) in payload_folded
            and normalise_for_leak_check(transcript)
            != normalise_for_leak_check(pii.masked_text)
        )

        # -- placeholder handling -----------------------------------------
        reply_brackets = bracket_tokens(reply)
        reply_known = [token for token in reply_brackets if token in request_tokens]
        reply_unknown = [token for token in reply_brackets if token not in request_tokens]
        restored = rehydrate(reply, mapping)
        unresolved = unresolved_placeholders(reply, mapping)

        # Deterministic rehydration probe. It runs whether or not a live model
        # answered, and it is what makes "rehydration works" a measurement
        # rather than an anecdote: the reply is constructed here, so every
        # placeholder is guaranteed to be exercised, including a repeat.
        probe_reply = ""
        probe_ok: bool | None = None
        if request_tokens:
            echoed = " and again ".join(request_tokens)
            probe_reply = f"Noted: {' , '.join(request_tokens)}. Repeating {echoed}."
            probe_restored = rehydrate(probe_reply, mapping)
            probe_ok = (
                not PLACEHOLDER.search(probe_restored)
                and all(value in probe_restored for value in mapping.values())
            )

        speech_seconds = None
        if self.options.run_tts and index <= self.options.tts_cases:
            speech_seconds = self._speak(restored or reply)

        self.peak_rss = max(self.peak_rss, common.rss_bytes() or 0)

        row = {
            "clip_id": clip_id,
            "case_note": case.get("_case_note", ""),
            "audio_duration_s": case["duration_seconds"],
            "expected_entities": len(case.get("expected_private_spans", [])),
            "reference_transcript": case["reference_transcript"],
            "stt_transcript": transcript,
            "masked_text": pii.masked_text,
            "safe_to_release": bool(pii.safe_to_release),
            "detected_entities": len(pii.entities),
            "placeholder_map_size": len(mapping),
            "request_placeholders": request_tokens,
            "stt_latency_ms": stt_seconds * 1000.0,
            "privacy_latency_ms": mask_seconds * 1000.0,
            "local_latency_ms": local_seconds * 1000.0,
            "llm_latency_ms": (
                outcome["seconds"] * 1000.0
                if self.llm_mode != "payload audit only" else None
            ),
            "llm_error": outcome["error"],
            # Recorded per case so a latency figure can be attributed to one
            # model. The client falls back to the larger model on a refusal,
            # and averaging the two together would describe neither.
            "llm_model": outcome.get("model"),
            "llm_models_attempted": outcome.get("models_attempted") or [],
            "llm_used_fallback": bool(
                outcome.get("model")
                and self.llm is not None
                and outcome.get("model") == self.llm.settings.fallback_model
                and self.llm.settings.fallback_model != self.llm.settings.model
            ),
            "llm_rate_limited": bool(outcome.get("rate_limited")),
            "llm_reply": reply,
            "restored_reply": restored,
            "reply_bracket_tokens": reply_brackets,
            "reply_known_placeholders": reply_known,
            "reply_unknown_placeholders": reply_unknown,
            "unresolved_placeholders": [f"[{a}_{b}]" for a, b in unresolved],
            "rehydration_probe_reply": probe_reply,
            "rehydration_probe_ok": probe_ok,
            "payload_captured": len(captured),
            "masked_values": masked_values,
            "gold_values": gold_values,
            "raw_transcript_in_payload": transcript_in_payload,
            "tts_latency_ms": speech_seconds * 1000.0 if speech_seconds else None,
            "total_latency_ms": (
                (local_seconds + outcome["seconds"]) * 1000.0
                if self.llm_mode != "payload audit only" else None
            ),
            "peak_rss_mb": (self.peak_rss or 0) / 1024 ** 2,
        }
        row["end_to_end_ok"] = self._case_succeeded(row)

        for item in captured:
            self.payloads.append({
                "clip_id": clip_id,
                "url": item.url,
                "method": item.method,
                "header_names": item.header_names,
                "body_bytes": item.raw_bytes,
                "body": redacted_body(item),
            })

        leaked = sum(1 for value in masked_values if value["in_payload"])
        leaked += sum(1 for value in gold_values
                      if value["in_payload"] or value["recognized_in_payload"])
        flag = "ok " if row["end_to_end_ok"] else "FAIL"
        print(f"[{index:02d}/{total:02d}] {clip_id:14s} {case['duration_seconds']:6.2f}s "
              f"local {local_seconds * 1000:6.0f} ms  "
              f"llm {('%6.0f ms' % row['llm_latency_ms']) if row['llm_latency_ms'] else '     -  '}  "
              f"ph {len(request_tokens):2d}  leak {leaked}  {flag}")
        return row

    def _case_succeeded(self, row: Mapping[str, Any]) -> bool:
        """The seven conditions, all of them, or the case did not succeed."""

        if not row["stt_transcript"].strip():
            return False
        if not row["safe_to_release"]:
            return False
        # The boundary condition, and the only leak that is this layer's
        # fault: the masker replaced a value and it left the device anyway.
        if any(value["in_payload"] for value in row["masked_values"]):
            return False
        if any(
            value["detected_by_masker"]
            and (value["in_payload"] or value["recognized_in_payload"])
            for value in row["gold_values"]
        ):
            return False
        if row["raw_transcript_in_payload"]:
            return False
        if row["rehydration_probe_ok"] is False:
            return False
        if self.llm_mode == "payload audit only":
            # Everything measurable without a transport passed. The API leg is
            # reported separately and is not folded into this number.
            return True
        if row["llm_error"]:
            return False
        if row["reply_unknown_placeholders"]:
            return False
        if row["unresolved_placeholders"]:
            return False
        return bool(row["restored_reply"].strip())

    # -- probes ------------------------------------------------------------

    def run_placeholder_probes(self) -> None:
        """Malformed, unknown and repeated placeholders must not crash anything."""

        from app.rehydrate import (
            ConversationPlaceholders,
            RehydrationMap,
            rehydrate,
            unresolved_placeholders,
        )

        mapping = {"[GIVENNAME_1]": "Fahad", "[TELEPHONENUM_1]": "0551234567"}
        probes = [
            ("single", "Okay, reminder to call [GIVENNAME_1] tomorrow.",
             "Okay, reminder to call Fahad tomorrow."),
            ("two types", "I will text [GIVENNAME_1] on [TELEPHONENUM_1].",
             "I will text Fahad on 0551234567."),
            ("repeated", "[GIVENNAME_1] and [GIVENNAME_1] again.",
             "Fahad and Fahad again."),
            ("no placeholder", "It is sunny today.", "It is sunny today."),
            ("unknown placeholder", "I cannot reach [SURNAME_9].",
             "I cannot reach [SURNAME_9]."),
            ("malformed bracket", "Try [GIVENNAME_ or [givenname_1] instead.",
             "Try [GIVENNAME_ or [givenname_1] instead."),
            ("ordinary brackets", "The note said [see attachment].",
             "The note said [see attachment]."),
            ("empty reply", "", ""),
        ]
        for name, reply, expected in probes:
            try:
                got = rehydrate(reply, mapping)
                ok = got == expected
                error = ""
            except Exception as exc:
                got, ok, error = "", False, f"{type(exc).__name__}: {exc}"
            self.probe_results.append({
                "probe": name, "reply": reply, "expected": expected,
                "got": got, "ok": ok, "error": error,
            })

        # A rolling map across turns, over two different placeholder names.
        rolling = RehydrationMap()
        try:
            rolling.add([{"placeholder": "[GIVENNAME_1]", "text": "Fahad"}])
            rolling.add([{"placeholder": "[EMAIL_1]", "text": "f@example.com"}])
            got = rolling.apply("Mail [GIVENNAME_1] at [EMAIL_1].")
            expected = "Mail Fahad at f@example.com."
            missing = rolling.missing("Also [SURNAME_3].")
            self.probe_results.append({
                "probe": "rolling map across turns", "reply": "Mail [GIVENNAME_1] at [EMAIL_1].",
                "expected": expected, "got": got,
                "ok": got == expected and missing == [("SURNAME", "3")],
                "error": "",
            })
        except Exception as exc:
            self.probe_results.append({
                "probe": "rolling map across turns", "reply": "", "expected": "",
                "got": "", "ok": False, "error": f"{type(exc).__name__}: {exc}",
            })

        # The cross-turn collision V5 shipped with, and the property V6 adds.
        # The masker numbers placeholders per utterance, so two different
        # people arrive as [GIVENNAME_1] one turn apart. Under a single rolling
        # map the second overwrote the first and a reply about the first was
        # restored with the second person's name. Nothing leaked - but the
        # device said it would call the wrong person, and that has to be a
        # measured property rather than a claim.
        try:
            conversation = ConversationPlaceholders()
            turn_one, _ = conversation.adopt(
                "Remind me to call [GIVENNAME_1] tomorrow",
                [{"placeholder": "[GIVENNAME_1]", "text": "Fahad"}],
            )
            turn_two, _ = conversation.adopt(
                "Remind me to call [GIVENNAME_1] tomorrow",
                [{"placeholder": "[GIVENNAME_1]", "text": "Ali"}],
            )
            restored_one = conversation.apply("I will call [GIVENNAME_1] tomorrow.")
            restored_two = conversation.apply("I will call [GIVENNAME_2] tomorrow.")
            ok = (
                turn_one == "Remind me to call [GIVENNAME_1] tomorrow"
                and turn_two == "Remind me to call [GIVENNAME_2] tomorrow"
                and restored_one == "I will call Fahad tomorrow."
                and restored_two == "I will call Ali tomorrow."
                and "Ali" not in restored_one
                and "Fahad" not in restored_two
            )
            self.probe_results.append({
                "probe": "two turns, same placeholder, different people",
                "reply": f"{turn_one} | {turn_two}",
                "expected": "I will call Fahad tomorrow. | I will call Ali tomorrow.",
                "got": f"{restored_one} | {restored_two}",
                "ok": ok,
                "error": "",
            })
        except Exception as exc:
            self.probe_results.append({
                "probe": "two turns, same placeholder, different people",
                "reply": "", "expected": "", "got": "", "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        passed = sum(1 for probe in self.probe_results if probe["ok"])
        print(f"\nPlaceholder probes: {passed}/{len(self.probe_results)} passed")
        self.completed["placeholder_probes"] = passed == len(self.probe_results)

    def run_local_command_check(self) -> None:
        """An instruction to the device must not become a network request.

        Recognising a command locally is the point of recognising it, and the
        saving is not only latency: "turn the kitchen lights off" never leaves
        the house. This drives a real clip through the real recognizer and the
        real masker, routes it exactly as ``post_mask_assistant`` does, and
        then asserts the transport was never reached - checked by counting the
        requests the client actually built, not by trusting the router.

        It costs no API quota, which is why it is a default rather than an
        option.
        """

        from app.onkith_music import route_onkith_request
        from app.wake_word_normalizer import normalize_wake_word
        from .audio_benchmark import read_wav, verify_clip

        clip_id, note = LOCAL_COMMAND_CASE
        case = next(
            (item for item in load_manifest()["clips"] if item["clip_id"] == clip_id),
            None,
        )
        if case is None:
            self.warnings.note(f"{clip_id} is not in the manifest; check skipped")
            return

        path = verify_clip(case, fixture_dir=FIXTURE_DIR)
        transcript, _ = self._transcribe(read_wav(path), clip_id)
        pii = self.masker.mask_text(transcript)

        module = self.llm_module
        real_urlopen = module.urllib.request.urlopen
        built: list[Any] = []
        module.urllib.request.urlopen = lambda request, *a, **k: built.append(request)
        try:
            # Unaddressed speech: no wake word, so it is held locally and the
            # assistant is never asked anything at all.
            unaddressed = route_onkith_request(normalize_wake_word(transcript)[0])
            # Addressed speech naming a capability the device has: routed to
            # the local action lane, which also never asks the model.
            addressed = route_onkith_request(
                normalize_wake_word(f"Onkith, {transcript}")[0]
            )
        finally:
            module.urllib.request.urlopen = real_urlopen

        held_locally = not unaddressed.command_detected
        routed_locally = bool(addressed.command_detected) and addressed.intent != "ASK"
        self.local_command_check = {
            "clip_id": clip_id,
            "note": note,
            "transcript": transcript,
            "masked_text": pii.masked_text,
            "unaddressed_held_locally": held_locally,
            "addressed_intent": addressed.intent,
            "addressed_routed_locally": routed_locally,
            "requests_built": len(built),
            "ok": held_locally and routed_locally and not built,
        }
        verdict = "ok" if self.local_command_check["ok"] else "FAIL"
        print(
            f"Local command check: {clip_id} -> "
            f"{addressed.intent or 'held'}, {len(built)} request(s) built  {verdict}"
        )
        self.completed["local_command_check"] = self.local_command_check["ok"]

    # -- aggregation -------------------------------------------------------

    def summarise_run(self) -> None:
        rows = self.rows
        local = [row["local_latency_ms"] / 1000.0 for row in rows]
        stt = [row["stt_latency_ms"] / 1000.0 for row in rows]
        privacy = [row["privacy_latency_ms"] / 1000.0 for row in rows]
        llm = [row["llm_latency_ms"] / 1000.0 for row in rows
               if row["llm_latency_ms"] is not None and not row["llm_error"]]
        total = [row["total_latency_ms"] / 1000.0 for row in rows
                 if row["total_latency_ms"] is not None and not row["llm_error"]]
        tts = [row["tts_latency_ms"] / 1000.0 for row in rows
               if row["tts_latency_ms"] is not None]
        audio_seconds = sum(row["audio_duration_s"] for row in rows)

        masked_values = [value for row in rows for value in row["masked_values"]]
        gold_values = [value for row in rows for value in row["gold_values"]]
        with_payload = [row for row in rows if row["payload_captured"]]

        replies = [row for row in rows if row["llm_reply"].strip()]
        referring = [row for row in replies if row["reply_bracket_tokens"]]
        preserved = [row for row in referring if not row["reply_unknown_placeholders"]]
        probe_rows = [row for row in rows if row["rehydration_probe_ok"] is not None]
        probe_ok = [row for row in probe_rows if row["rehydration_probe_ok"]]
        live_restored = [row for row in referring if not row["unresolved_placeholders"]]

        failures: dict[str, int] = {}
        for row in rows:
            if row["end_to_end_ok"]:
                continue
            if not row["stt_transcript"].strip():
                failures["stt"] = failures.get("stt", 0) + 1
            elif not row["safe_to_release"]:
                failures["masking"] = failures.get("masking", 0) + 1
            elif any(value["in_payload"] for value in row["masked_values"]) or \
                    any(value["in_payload"] or value["recognized_in_payload"]
                        for value in row["gold_values"]) or \
                    row["raw_transcript_in_payload"]:
                failures["pii_leak"] = failures.get("pii_leak", 0) + 1
            elif row["llm_error"]:
                failures["api"] = failures.get("api", 0) + 1
            elif row["reply_unknown_placeholders"]:
                failures["placeholder_corruption"] = (
                    failures.get("placeholder_corruption", 0) + 1
                )
            elif row["unresolved_placeholders"] or row["rehydration_probe_ok"] is False:
                failures["unmasking"] = failures.get("unmasking", 0) + 1
            else:
                failures["other"] = failures.get("other", 0) + 1

        self.summary = {
            "run_id": self.run_id,
            "cases": len(rows),
            "audio_seconds": audio_seconds,
            "llm_configured": self.llm_configured,
            "llm_mode": self.llm_mode,
            "local": {
                "stt_seconds": summarise(stt),
                "privacy_seconds": summarise(privacy),
                "pre_llm_seconds": summarise(local),
                "real_time_factor_mean": (
                    statistics.fmean(local) / (audio_seconds / len(rows))
                    if rows and audio_seconds else None
                ),
                "note": (
                    "Local only: Whisper plus the fail-closed DeBERTa masker, "
                    "measured on this device. Deliberately kept apart from the "
                    "network figure below, because a slow answer caused by wifi "
                    "is not caused by the privacy system."
                ),
            },
            "llm": {
                "measured": bool(llm),
                "requests_attempted": len(with_payload),
                "requests_succeeded": len(llm),
                "requests_failed": sum(1 for row in rows if row["llm_error"]
                                       and row["payload_captured"]),
                "round_trip_seconds": summarise(llm) if llm else {"count": 0},
                "provider": "groq",
                "model": self.llm.settings.model if self.llm else None,
                "fallback_model": self.llm.settings.fallback_model if self.llm else None,
                "timeout_seconds": self.llm.settings.timeout_seconds if self.llm else None,
                "mode": self.llm_mode,
                "note": (
                    "Network round trip only."
                    if self.llm_mode == "live" else
                    "Stubbed transport: the request was built and parsed by the "
                    "real client but answered locally, so this timing is the "
                    "integration's own cost with no network in it, and the "
                    "placeholder-preservation figure describes a cooperative "
                    "responder rather than Groq."
                    if self.llm_mode == "stubbed transport" else
                    "Not run: no API key was configured, so no request left this "
                    "machine. The payload each request would have carried was "
                    "still built by the real client and is in "
                    "e2e_outbound_payloads.jsonl."
                ),
            },
            "end_to_end": {
                "total_seconds": summarise(total) if total else {"count": 0},
                "successful_cases": sum(1 for row in rows if row["end_to_end_ok"]),
                "case_count": len(rows),
                "failures_by_cause": failures,
                "success_definition": (
                    "usable transcript, masking released, no raw private value "
                    "in the outbound payload, deterministic rehydration probe "
                    "restored every value"
                    + (
                        f", the request succeeded ({self.llm_mode}), no "
                        "corrupted placeholder in the reply, every returned "
                        "placeholder resolved, and a non-empty restored answer."
                        if self.llm_mode != "payload audit only" else
                        ". The four transport-dependent conditions are not "
                        "counted in this run because no key was configured."
                    )
                ),
            },
            "privacy_round_trip": {
                "payloads_captured": len(self.payloads),
                "masker_values_tested": len(masked_values),
                "masker_values_in_payload": sum(
                    1 for value in masked_values if value["in_payload"]
                ),
                "gold_entities_tested": len(gold_values),
                "gold_entities_in_payload": sum(
                    1 for value in gold_values if value["in_payload"]
                ),
                "gold_entities_recognized_surface_in_payload": sum(
                    1 for value in gold_values if value["recognized_in_payload"]
                ),
                "gold_entities_altered_by_stt": sum(
                    1 for value in gold_values
                    if value["alignment"] not in {"matched", ""}
                ),
                "raw_transcripts_in_payload": sum(
                    1 for row in rows if row["raw_transcript_in_payload"]
                ),

                # ------------------------------------------------------------
                # The distinction that matters, kept as two numbers.
                # ------------------------------------------------------------
                # A value the masker replaced, which then appeared in the
                # payload anyway. This is a failure of the privacy boundary -
                # the thing this pipeline exists to provide - and the only
                # acceptable value is zero.
                "masked_but_leaked": sum(
                    1 for value in masked_values if value["in_payload"]
                ) + sum(
                    1 for value in gold_values
                    if value["detected_by_masker"]
                    and (value["in_payload"] or value["recognized_in_payload"])
                ),
                # A gold entity the privacy model never detected. It was never
                # behind the boundary, so its surface is still in the text that
                # went out. This is a recall miss, measured properly against
                # gold annotations by the 44-clip audio benchmark, and it is
                # reported here so that a clean boundary is not mistaken for
                # perfect detection.
                "missed_by_detector_and_in_payload": sum(
                    1 for value in gold_values
                    if not value["detected_by_masker"]
                    and (value["in_payload"] or value["recognized_in_payload"])
                ),
                "missed_by_detector_total": sum(
                    1 for value in gold_values if not value["detected_by_masker"]
                ),
                "detector_misses": [
                    {
                        "clip_id": row["clip_id"],
                        "label": value["label"],
                        "spoken": value["value"],
                        "recognized": value["recognized"],
                        "in_payload": bool(
                            value["in_payload"] or value["recognized_in_payload"]
                        ),
                    }
                    for row in rows
                    for value in row["gold_values"]
                    if not value["detected_by_masker"]
                ],
                # Gold entities whose recognizer surface was too short for a
                # substring test to mean anything. Neither passed nor failed.
                "surfaces_too_short_to_test": sum(
                    1 for value in gold_values
                    if not value["surface_testable"] and value["recognized"].strip()
                ),

                "leakage_rate": (
                    (sum(1 for value in masked_values if value["in_payload"])
                     + sum(1 for value in gold_values
                           if value["in_payload"] or value["recognized_in_payload"]))
                    / max(1, len(masked_values) + len(gold_values))
                ),
                "definition": (
                    "Checks against the bytes the LLM client handed to urllib, "
                    "not against the arguments a function was called with. "
                    "Every side is number-folded first, so a telephone number "
                    "spoken as words and written by the recognizer as digits "
                    "still matches. The two headline counts answer different "
                    "questions and must not be added together: "
                    "`masked_but_leaked` is a value the masker replaced that "
                    "left the device anyway, which is a boundary failure and "
                    "must be zero; `missed_by_detector_and_in_payload` is a "
                    "gold entity the privacy model never found, which is a "
                    "recall miss and is why the audio benchmark exists. A "
                    "pipeline can have a perfect boundary and an imperfect "
                    "detector, and this run says which."
                ),
            },
            "placeholder_preservation": {
                "replies_received": len(replies),
                "replies_referring_to_a_placeholder": len(referring),
                "replies_with_only_intact_placeholders": len(preserved),
                "measured_against": self.llm_mode,
                "definition": (
                    "Of the replies that used any bracketed token at all, how "
                    "many used only tokens that were in the request. A reply "
                    "containing [Name], [GIVENNAME] or an invented [TIME_1] "
                    "counts as corrupted, because no local map can fill it. "
                    "Meaningful only against a live model: a stubbed run scores "
                    "the responder this file wrote."
                ),
            },
            "rehydration": {
                "deterministic_probe_cases": len(probe_rows),
                "deterministic_probe_passed": len(probe_ok),
                "live_replies_fully_resolved": len(live_restored),
                "live_replies_referring": len(referring),
                "malformed_probe_passed": sum(
                    1 for probe in self.probe_results if probe["ok"]
                ),
                "malformed_probe_count": len(self.probe_results),
            },
            "tts": {
                "backend": self.tts_backend,
                "measured_cases": len(tts),
                "seconds": summarise(tts) if tts else {"count": 0},
                "speaks": (
                    "the locally restored answer, never the masked reply"
                    if tts else "not measured in this run"
                ),
            },
            "models": {
                # Reported per model rather than pooled. The client falls back
                # to the larger model when the smaller one refuses, and one
                # mean over both would be a latency figure for neither.
                "primary": getattr(self.llm.settings, "model", None),
                "fallback": getattr(self.llm.settings, "fallback_model", None),
                "answered_by": {
                    name: sum(1 for row in rows if row.get("llm_model") == name)
                    for name in sorted(
                        {row.get("llm_model") for row in rows if row.get("llm_model")}
                    )
                },
                "cases_using_fallback": sum(
                    1 for row in rows if row.get("llm_used_fallback")
                ),
            },
            "rate_limiting": {
                "cases_rate_limited": len(self.rate_limited_cases),
                "clips": list(self.rate_limited_cases),
                "observed_headers": dict(self.rate_limit_headers),
                "request_pause_seconds": self.options.request_pause,
                "history_reset_between_cases": True,
                "definition": (
                    "A case is rate limited when both the primary and the "
                    "fallback model refused it with HTTP 429. Retry-After is "
                    "honoured once, capped at "
                    f"{MAX_RETRY_AFTER_SECONDS:.0f}s; past that the case is "
                    "recorded as rate limited and the run continues rather "
                    "than waiting out a quota window."
                ),
            },
            "local_command": dict(self.local_command_check),
        }

    # -- output ------------------------------------------------------------

    def save(self) -> None:
        finished_utc = common.utc_now()
        config_payload = {
            "run_id": self.run_id,
            "benchmark": "v6_end_to_end",
            "pipeline_version": "V6",
            "pipeline_root": common.relative(common.PRODUCTION_ROOT),
            "stt_engine": "whisper",
            # V6's recognizer and privacy core are V4's files unchanged.
            "core_identical_to": "V4",
            "stt_model_id": self.config.whisper["model_id"],
            "stt_compute_type": self.config.whisper["compute_type"],
            "stt_beam_size": self.config.whisper["beam_size"],
            "privacy_model_sha256": self.artifact["measured_sha256"],
            "llm_provider": "groq",
            "llm_model": self.llm.settings.model if self.llm else None,
            "llm_configured": self.llm_configured,
            "llm_mode": self.llm_mode,
            "tts_backend": self.tts_backend,
            "mode": self.options.mode,
            "cases": [case["clip_id"] for case in self.cases],
            "device_label": self.options.device_label or None,
            "entry_point": "Full_Pipeline_V6_EndToEnd_Benchmark.py",
            "started_utc": self.started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "duration_seconds": (finished_utc - self.started_utc).total_seconds(),
            "warnings": list(self.warnings),
            "sections_completed": self.completed,
        }
        memory = common.collect_memory(
            self.run_id, baseline_rss=self.baseline_rss,
            peak_rss=self.peak_rss, stages=self.stage_rss,
        )

        self.run.json("system_info.json", self.system_info)
        self.run.json("benchmark_config.json", config_payload)
        self.run.json("artifact_manifest.json",
                      common.artifact_manifest_for_disk(self.artifact))
        self.run.json("e2e_summary.json", self.summary)
        self.run.json("e2e_memory_metrics.json", memory)
        self.run.csv("e2e_case_results.csv", [
            {name: row.get(name) for name in CASE_CSV_FIELDS} for row in self.rows
        ], CASE_CSV_FIELDS)
        self.run.jsonl("e2e_cases.jsonl", self.rows)
        self.run.jsonl("e2e_outbound_payloads.jsonl", self.payloads)
        self.run.jsonl("e2e_placeholder_probes.jsonl", self.probe_results)
        if self.local_command_check:
            self.run.json("e2e_local_command_check.json", self.local_command_check)
        self.run.markdown("e2e_benchmark_summary.md", self._summary_lines(config_payload))
        self.run.report()

    def _summary_lines(self, config_payload: Mapping[str, Any]) -> list[str]:
        summary = self.summary
        info = self.system_info
        local = summary["local"]
        llm = summary["llm"]
        privacy = summary["privacy_round_trip"]
        preservation = summary["placeholder_preservation"]
        rehydration = summary["rehydration"]
        end = summary["end_to_end"]
        tts = summary["tts"]

        def stat(block: Mapping[str, Any], key: str) -> str:
            value = block.get(key)
            return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"

        lines = [
            f"# V6 integrated round-trip benchmark - {self.run_id}",
            "",
            "Audio in, spoken answer out, through the pipeline the presentation",
            "runs. It is an **integration and latency** benchmark: recognition and",
            "masking accuracy are measured on 44 clips by",
            "`Full_Pipeline_Audio_Benchmark.py`, and the numbers below are not a",
            "substitute for those.",
            "",
            f"- **Cases:** {summary['cases']} clips, "
            f"{summary['audio_seconds']:.1f} s of audio",
            f"- **Pipeline:** {config_payload['pipeline_root']} "
            f"(Whisper {config_payload['stt_model_id']}, DeBERTa-v3-xsmall INT8)",
            f"- **LLM:** groq {config_payload['llm_model']} - "
            + {
                "live": "live",
                "stubbed transport": (
                    "**stubbed transport**: the request was built and parsed by "
                    "the real client and answered locally. Not a Groq "
                    "measurement."
                ),
            }.get(summary["llm_mode"],
                  "**not called** (no API key); payload audit only"),
            f"- **TTS:** {tts['backend']}",
            f"- **Started / finished (UTC):** {config_payload['started_utc'][:19]} -> "
            f"{config_payload['finished_utc'][:19]} "
            f"({config_payload['duration_seconds'] / 60:.1f} min)",
            "",
            "## Device",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| OS | {info['platform']} |",
            f"| Architecture | {info['architecture']} |",
            f"| CPU | {info['cpu_model']} |",
            f"| Logical / physical cores | {info['logical_cpus']} / {info['physical_cpus']} |",
            f"| RAM | {common.gib(info['total_ram_bytes'])} |",
            f"| Privacy model SHA-256 | `{config_payload['privacy_model_sha256']}` |",
            "",
            "## Privacy round trip",
            "",
            "Read off the bytes handed to `urllib` inside `app/llm.py`, not off an",
            "assumption that masking worked. Every payload is in",
            "`e2e_outbound_payloads.jsonl` verbatim, so the claim can be checked",
            "rather than believed.",
            "",
            "| Check | Value |",
            "|---|---:|",
            f"| Outbound payloads captured | {privacy['payloads_captured']} |",
            f"| Values the masker replaced, tested | {privacy['masker_values_tested']} |",
            f"| ...appearing in a payload | **{privacy['masker_values_in_payload']}** |",
            f"| Gold private entities tested | {privacy['gold_entities_tested']} |",
            f"| ...spoken surface appearing in a payload | "
            f"**{privacy['gold_entities_in_payload']}** |",
            f"| ...recognized surface appearing in a payload | "
            f"**{privacy['gold_entities_recognized_surface_in_payload']}** |",
            f"| ...surface altered by the recognizer before masking | "
            f"{privacy['gold_entities_altered_by_stt']} |",
            f"| Raw transcripts appearing in a payload | "
            f"**{privacy['raw_transcripts_in_payload']}** |",
            f"| Leakage rate | {privacy['leakage_rate']:.4f} |",
            "",
            privacy["definition"],
            "",
            "## Placeholder preservation",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Replies received | {preservation['replies_received']} |",
            f"| Replies referring to a placeholder | "
            f"{preservation['replies_referring_to_a_placeholder']} |",
            f"| ...using only intact request placeholders | "
            f"{preservation['replies_with_only_intact_placeholders']} |",
            "",
            preservation["definition"],
            "",
            "## Local rehydration",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Deterministic probe cases | {rehydration['deterministic_probe_cases']} |",
            f"| ...restored every value | {rehydration['deterministic_probe_passed']} |",
            f"| Live replies referring to a placeholder | "
            f"{rehydration['live_replies_referring']} |",
            f"| ...fully resolved locally | {rehydration['live_replies_fully_resolved']} |",
            f"| Malformed/unknown-placeholder probes passed | "
            f"{rehydration['malformed_probe_passed']}/"
            f"{rehydration['malformed_probe_count']} |",
            "",
            "## Latency",
            "",
            "Local and network are reported apart on purpose. The first is this",
            "device; the second is somebody's internet connection.",
            "",
            "| Stage | Mean (ms) | P50 | P95 |",
            "|---|---:|---:|---:|",
            f"| Whisper STT | {stat(local['stt_seconds'], 'mean')} | "
            f"{stat(local['stt_seconds'], 'p50')} | {stat(local['stt_seconds'], 'p95')} |",
            f"| Privacy masking | {stat(local['privacy_seconds'], 'mean')} | "
            f"{stat(local['privacy_seconds'], 'p50')} | "
            f"{stat(local['privacy_seconds'], 'p95')} |",
            f"| Local pre-LLM total | {stat(local['pre_llm_seconds'], 'mean')} | "
            f"{stat(local['pre_llm_seconds'], 'p50')} | "
            f"{stat(local['pre_llm_seconds'], 'p95')} |",
            f"| Groq round trip | {stat(llm['round_trip_seconds'], 'mean')} | "
            f"{stat(llm['round_trip_seconds'], 'p50')} | "
            f"{stat(llm['round_trip_seconds'], 'p95')} |",
            f"| End to end, text answer | {stat(end['total_seconds'], 'mean')} | "
            f"{stat(end['total_seconds'], 'p50')} | {stat(end['total_seconds'], 'p95')} |",
            f"| Local TTS | {stat(tts['seconds'], 'mean')} | "
            f"{stat(tts['seconds'], 'p50')} | {stat(tts['seconds'], 'p95')} |",
            "",
            f"- API requests attempted: {llm['requests_attempted']}, "
            f"succeeded: {llm['requests_succeeded']}, failed: {llm['requests_failed']}",
            f"- {llm['note']}",
            f"- TTS speaks {tts['speaks']}.",
            "",
            "## End-to-end outcome",
            "",
            f"**Successful cases: {end['successful_cases']} / {end['case_count']}**",
            "",
            end["success_definition"],
            "",
        ]
        if end["failures_by_cause"]:
            lines += ["| Failure cause | Cases |", "|---|---:|"]
            lines += [f"| {name} | {count} |"
                      for name, count in sorted(end["failures_by_cause"].items())]
        else:
            lines.append("No case failed.")
        lines += [
            "",
            "## Memory",
            "",
            "| Stage | RSS |",
            "|---|---|",
        ]
        for name, value in self.stage_rss.items():
            lines.append(f"| {name} | {common.mib(value)} |")
        lines += [
            f"| peak during run | {common.mib(self.peak_rss)} |",
            "",
            "## Warnings",
            "",
        ]
        lines += ([f"- {message}" for message in self.warnings]
                  if self.warnings else ["- none"])
        return lines

    # -- driver ------------------------------------------------------------

    def run_all(self) -> int:
        self.announce()
        self.verify_fixtures()
        self.load_components()
        print("Processing cases")
        print("-" * 96)
        total = len(self.cases)
        for index, case in enumerate(self.cases, start=1):
            self.rows.append(self.process_case(case, index, total))
            if self.llm_mode == "live" and index < total:
                # A courtesy gap, not a measurement artifact: the free tier is
                # limited per minute in tokens as well as requests, and a
                # refusal would be scored as an integration failure it is not.
                # The interval follows the response headers where the account
                # sends them, and the fallback constant where it does not.
                time.sleep(self._pause_before_next_request())
        print("-" * 96)
        self.run_local_command_check()
        self.run_placeholder_probes()
        self.completed["cases"] = True
        self.summarise_run()

        print()
        print_summary("local pre-LLM", self.summary["local"]["pre_llm_seconds"])
        if self.summary["llm"]["measured"]:
            print_summary("groq round trip", self.summary["llm"]["round_trip_seconds"])
            print_summary("end to end", self.summary["end_to_end"]["total_seconds"])
        if self.summary["tts"]["measured_cases"]:
            print_summary("local TTS", self.summary["tts"]["seconds"])
        privacy = self.summary["privacy_round_trip"]
        print()
        print(f"masked PII leaked anyway     : {privacy['masked_but_leaked']}"
              f" / {privacy['masker_values_tested']} replaced values"
              "   (boundary failures; must be 0)")
        print(f"PII missed by the detector   : "
              f"{privacy['missed_by_detector_and_in_payload']}"
              f" / {privacy['gold_entities_tested']} gold entities"
              "   (recall misses, never masked)")
        for miss in privacy["detector_misses"]:
            if miss["in_payload"]:
                print(f"    {miss['clip_id']:14s} {miss['label']:14s} "
                      f"{miss['recognized'] or miss['spoken']!r}")
        if privacy["surfaces_too_short_to_test"]:
            print(f"untestable short surfaces    : "
                  f"{privacy['surfaces_too_short_to_test']} (alignment fragment; "
                  "neither passed nor failed)")
        print(f"payloads captured            : {privacy['payloads_captured']}")
        end = self.summary["end_to_end"]
        print(f"end-to-end successful cases  : "
              f"{end['successful_cases']} / {end['case_count']}")
        print()
        self.save()
        if self.llm_mode == "stubbed transport":
            print()
            print("--llm-stub: no request left this machine. The latency and")
            print("placeholder figures above describe the integration, not Groq.")
            return 0
        if not self.llm_configured:
            print()
            print("GROQ_API_KEY was not set, so the live API, placeholder-preservation")
            print("and end-to-end latency sections report 'not run' rather than a")
            print("number. Everything local was measured, and the outbound payloads")
            print("were built and audited. Exit code 2 means exactly this.")
            return 2
        return 0

    def close(self) -> None:
        try:
            if self.whisper is not None:
                self.whisper.close()
        except Exception:
            pass


CASE_CSV_FIELDS = [
    "clip_id", "case_note", "audio_duration_s", "expected_entities",
    "detected_entities", "placeholder_map_size", "stt_latency_ms",
    "privacy_latency_ms", "local_latency_ms", "llm_latency_ms",
    "total_latency_ms", "tts_latency_ms", "rehydration_probe_ok",
    "raw_transcript_in_payload", "end_to_end_ok", "peak_rss_mb",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Full_Pipeline_V6_EndToEnd_Benchmark.py",
        description=(
            "OnKith V6 integrated round-trip benchmark. Drives committed WAV "
            "clips through Whisper, the DeBERTa privacy masker, the Groq "
            "request, local rehydration and local TTS, and reports what left "
            "this machine, whether the placeholders survived, whether the "
            "values came back, and what each leg cost. Six live cases by "
            "default, because the free tier is token-limited and the 44-clip "
            "audio benchmark is where accuracy is actually measured."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes:\n"
            "  0  everything ran, including the live API leg\n"
            "  1  a fixture or the packaged privacy artifact was rejected\n"
            "  2  everything local ran; no GROQ_API_KEY, so the API leg did not\n"
        ),
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="three cases, for checking the wiring")
    parser.add_argument("--extended", action="store_true",
                        help="ten live cases instead of six")
    parser.add_argument("--case", action="append", dest="cases", metavar="CLIP_ID",
                        help="score only this clip; repeatable")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="audit the outbound payload without sending it, "
                             "even when a key is configured")
    parser.add_argument("--llm-stub", action="store_true",
                        help="answer each built request locally instead of "
                             "sending it, to exercise the whole integration "
                             "without a key. Never a measurement of Groq.")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--tts-cases", type=int, default=DEFAULT_TTS_CASES,
                        help=f"speak the first N restored answers "
                             f"(default: {DEFAULT_TTS_CASES})")
    parser.add_argument("--request-pause", type=float,
                        default=DEFAULT_REQUEST_PAUSE,
                        help="fallback seconds between live API requests when "
                             "the response carries no rate-limit headers "
                             f"(default: {DEFAULT_REQUEST_PAUSE:.0f})")
    parser.add_argument("--device-label", default="",
                        help="tag for the run id, e.g. pi5")
    parser.add_argument("--results-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    common.ensure_import_paths()
    args = build_parser().parse_args(argv)

    if args.list_cases:
        print("default (6 live cases):")
        for clip_id, note in DEFAULT_CASES:
            print(f"  {clip_id:14s} {note}")
        print("\n--extended adds:")
        for clip_id, note in EXTENDED_CASES[len(DEFAULT_CASES):]:
            print(f"  {clip_id:14s} {note}")
        print("\nalways included, routed locally and never sent:")
        print(f"  {LOCAL_COMMAND_CASE[0]:14s} {LOCAL_COMMAND_CASE[1]}")
        return 0

    if args.cases:
        wanted, mode = list(args.cases), "selected"
    elif args.smoke_test:
        wanted, mode = list(SMOKE_CASES), "smoke"
    elif args.extended:
        wanted, mode = [clip_id for clip_id, _ in EXTENDED_CASES], "extended"
    else:
        wanted, mode = [clip_id for clip_id, _ in DEFAULT_CASES], "full"

    options = Options(
        cases=wanted,
        mode=mode,
        device_label=args.device_label,
        results_root=args.results_root,
        run_tts=not args.no_tts,
        tts_cases=max(0, args.tts_cases),
        live=not args.offline,
        stub_llm=bool(args.llm_stub),
        request_pause=max(0.0, args.request_pause),
    )
    benchmark = EndToEndBenchmark(options)
    try:
        return benchmark.run_all()
    except FixtureRejected as exc:
        print(f"\nFixture rejected: {exc}", file=sys.stderr)
        return 1
    finally:
        benchmark.close()
