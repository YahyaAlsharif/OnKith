"""Deterministically generate disjoint OOD development and frozen-final sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .schema import EntitySpan, EvalCase, write_cases

GENERATOR_VERSION = "1.0.0"
SEEDS = {"dev": 20260811, "final": 20260812}
CATEGORY_ALLOCATION = {
    "names": 72,
    "email": 60,
    "numeric_structured": 84,
    "contextual_multi_entity": 48,
    "hard_negatives": 72,
    "long_window": 24,
}

NAME_POOLS = {
    "dev": [
        ("Amina", "Al-Hassan"), ("Siobhán", "O'Rourke"),
        ("João", "da Costa"), ("Łukasz", "Nowak"),
        ("Mei-Ling", "Zhou"), ("Kwame", "Mensah"),
        ("Priya", "Nair"), ("Fatou", "Diop"),
        ("Nikolai", "Petrov"), ("Ximena", "Quispe"),
        ("Tariq", "ibn Rashid"), ("Zahra", "El Masri"),
    ],
    "final": [
        ("Nneka", "Okafor"), ("Björk", "Þórsdóttir"),
        ("Santiago", "De la Vega"), ("Ayşe", "Yılmaz"),
        ("Hye-Jin", "Park"), ("Thabo", "Mokoena"),
        ("Ananya", "Iyer"), ("Mariam", "Ba"),
        ("Dmytro", "Shevchenko"), ("Inti", "Huamán"),
        ("Rami", "Abu Khalil"), ("Leïla", "Ben Saïd"),
    ],
}

EMAIL_POOLS = {
    "dev": [
        ("noor.s", "quantum-labs.dev"), ("ops+night", "northstar.coop"),
        ("a_b-test", "mail.riverstone.technology"), ("q.x", "cedar-house.museum"),
        ("alerts-7", "sub.orbit.example.net"), ("first.last", "alpine-solutions.travel"),
        ("x%tag", "secure.harbor.services"), ("team.bot", "deepfield.engineering"),
        ("user-42", "cobalt-research.science"), ("hello.world", "mail.saffron.agency"),
    ],
    "final": [
        ("mira.k", "lighthouse-studio.design"), ("audit+west", "pinecone.community"),
        ("z_y-test", "relay.bluewater.solutions"), ("r.t", "archive.lantern.foundation"),
        ("notify-8", "sub.ember.example.org"), ("given.family", "atlas-ventures.global"),
        ("n%route", "private.sunrise.support"), ("service.bot", "wildwood.consulting"),
        ("member-73", "aurora-research.institute"), ("contact.point", "mail.marigold.network"),
    ],
}

NUMERIC_POOLS = {
    "dev": [
        ("TELEPHONENUM", "phone number", "+971 50 482 7193"),
        ("TELEPHONENUM", "contact number", "+353 86 274 9051"),
        ("CREDITCARDNUMBER", "card number", "4729 1604 8831 7256"),
        ("CREDITCARDNUMBER", "payment card", "5318-4402-9176-3054"),
        ("ACCOUNTNUM", "account number", "GB72 NWBK 6016 1331 9268 19"),
        ("ACCOUNTNUM", "bank account", "0094 7712 6408"),
        ("IDCARDNUM", "national id", "ID-7842-QL-951"),
        ("IDCARDNUM", "patient id", "PT 604 817 293"),
        ("PASSPORTNUM", "passport number", "P7X493821"),
        ("DRIVERLICENSENUM", "driver license", "DVL-82-591-QR"),
        ("TAXNUM", "tax identifier", "TX-940-761-285"),
        ("ZIPCODE", "postal code", "SW1A 7QJ"),
        ("TELEPHONENUM", "mobile", "+81 90 6382 4715"),
        ("ACCOUNTNUM", "routing number", "0210 0048 9"),
    ],
    "final": [
        ("TELEPHONENUM", "phone number", "+254 712 608 935"),
        ("TELEPHONENUM", "contact number", "+64 21 593 804"),
        ("CREDITCARDNUMBER", "card number", "6011 9438 7205 1642"),
        ("CREDITCARDNUMBER", "payment card", "3782-907184-63015"),
        ("ACCOUNTNUM", "account number", "DE58 5001 0517 5407 3249 31"),
        ("ACCOUNTNUM", "bank account", "8841 3506 2974"),
        ("IDCARDNUM", "national id", "ID-6395-ZM-204"),
        ("IDCARDNUM", "patient id", "PT 918 365 470"),
        ("PASSPORTNUM", "passport number", "K8R260519"),
        ("DRIVERLICENSENUM", "driver license", "DVL-47-830-XN"),
        ("TAXNUM", "tax identifier", "TX-315-892-746"),
        ("ZIPCODE", "postal code", "EC2Y 4AB"),
        ("TELEPHONENUM", "mobile", "+55 11 97642 8305"),
        ("ACCOUNTNUM", "routing number", "1100 0091 7"),
    ],
}

CONTEXT_POOLS = {
    "dev": [
        ("18B", "Rue des Lilas", "Lyon", "Orchid Transit", "14 March 2031"),
        ("742", "Maple Quay", "Halifax", "Copper Finch Labs", "6 July 2029"),
        ("5-12", "Sakura Dori", "Kyoto", "Blue Koi Studio", "22 October 2030"),
        ("91", "Kenyatta Avenue", "Nairobi", "Acacia Health Group", "3 May 2028"),
        ("27A", "Al Noor Street", "Muscat", "Desert Bloom Trust", "19 January 2032"),
        ("604", "Avenida del Sol", "Quito", "Andes Signal Works", "11 June 2027"),
        ("33", "Björkgatan", "Uppsala", "Northern Loom AB", "25 August 2033"),
        ("8C", "Lotus Crescent", "Kuala Lumpur", "Straits Archive", "9 April 2030"),
    ],
    "final": [
        ("44D", "Via delle Rose", "Bologna", "Amber Rail Society", "17 February 2034"),
        ("215", "Harbour Lane", "Dunedin", "Silver Fern Analytics", "8 September 2028"),
        ("7-9", "Haneul Gil", "Busan", "Morning Crane Media", "30 November 2031"),
        ("63", "Independence Road", "Accra", "Baobab Learning Centre", "12 April 2029"),
        ("16A", "Cedar Walk", "Amman", "Levant Water Forum", "21 December 2032"),
        ("903", "Calle del Lago", "Arequipa", "Condor Mapping Unit", "4 January 2027"),
        ("52", "Koivukuja", "Turku", "Baltic Thread Oy", "28 June 2033"),
        ("11F", "Jasmine Terrace", "Penang", "Monsoon Records", "15 May 2030"),
    ],
}

LONG_POOLS = {
    "dev": [
        ("GIVENNAME", "Amalthea"),
        ("EMAIL", "boundary.case@heliograph.aero"),
        ("TELEPHONENUM", "+358 45 781 2096"),
        ("ORGANISATION", "Verdant Compass Cooperative"),
    ],
    "final": [
        ("GIVENNAME", "Euphemia"),
        ("EMAIL", "window.test@moonlit-archive.gallery"),
        ("TELEPHONENUM", "+27 82 604 1739"),
        ("ORGANISATION", "Crimson Tern Research Collective"),
    ],
}

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _segments(*parts: str | tuple[str, str]) -> tuple[str, tuple[EntitySpan, ...]]:
    text_parts: list[str] = []
    entities: list[EntitySpan] = []
    cursor = 0
    for part in parts:
        if isinstance(part, tuple):
            entity_type, value = part
            start = cursor
            text_parts.append(value)
            cursor += len(value)
            entities.append(EntitySpan(entity_type, start, cursor, value))
        else:
            text_parts.append(part)
            cursor += len(part)
    return "".join(text_parts), tuple(entities)


def _case(
    partition: str,
    category: str,
    index: int,
    subcategory: str,
    template: str,
    parts: tuple[str | tuple[str, str], ...],
    tags: tuple[str, ...],
    *,
    pair_id: str | None = None,
    notes: str = "",
) -> EvalCase:
    text, entities = _segments(*parts)
    return EvalCase(
        case_id=f"{partition}-{category}-{index:03d}",
        partition=partition,
        category=category,
        subcategory=subcategory,
        template_id=f"{partition}-{template}",
        text=text,
        entities=entities,
        tags=tags,
        pair_id=pair_id,
        notes=notes,
    )


def _name_cases(partition: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    frames = {
        "dev": ("Please send the report to ", " today.", "I was speaking with "),
        "final": ("The next appointment belongs to ", " this afternoon.", "Earlier we heard from "),
    }[partition]
    index = 1
    for pool_index, (given, surname) in enumerate(NAME_POOLS[partition], start=1):
        pair = f"{partition}-name-{pool_index:02d}"
        variants = [
            ("standard", (frames[0], ("GIVENNAME", given), " ", ("SURNAME", surname), frames[1]), ("written", "case_title")),
            ("lowercase", (frames[0].lower(), ("GIVENNAME", given.lower()), " ", ("SURNAME", surname.lower()), frames[1].lower()), ("minimal_pair", "case_lower", "speech_like")),
            ("uppercase", (frames[0].upper(), ("GIVENNAME", given.upper()), " ", ("SURNAME", surname.upper()), frames[1].upper()), ("minimal_pair", "case_upper")),
            ("reordered", ("The family name is ", ("SURNAME", surname), " and the given name is ", ("GIVENNAME", given), "."), ("reordered_name", "contextual")),
            ("initial", ("The contact is ", ("GIVENNAME", given[0]), ". ", ("SURNAME", surname), "; please confirm."), ("initials", "ambiguous")),
            ("conversation", ("uh ", frames[2].lower(), ("GIVENNAME", given.lower()), " ", ("SURNAME", surname.lower()), " yesterday and they asked me to call back"), ("speech_like", "filler_words", "punctuation_free")),
        ]
        for variant, parts, tags in variants:
            cases.append(_case(partition, "names", index, variant, f"names-{pool_index:02d}-{variant}", parts, tags, pair_id=pair))
            index += 1
    return cases


def _spoken_email(local: str, domain: str) -> str:
    value = f"{local} at {domain}"
    return value.replace(".", " dot ").replace("-", " dash ").replace("+", " plus ")


def _email_cases(partition: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    index = 1
    for pool_index, (local, domain) in enumerate(EMAIL_POOLS[partition], start=1):
        address = f"{local}@{domain}"
        subdomain_address = f"{local}@relay.{domain}"
        plus_address = f"{local}+receipt@{domain}"
        pair = f"{partition}-email-{pool_index:02d}"
        variants = [
            ("unseen_domain", ("Email me at ", ("EMAIL", address), "."), ("written", "unseen_domain")),
            ("uppercase", ("EMAIL ME AT ", ("EMAIL", address.upper()), "."), ("minimal_pair", "case_upper", "unseen_domain")),
            ("subdomain", ("Use ", ("EMAIL", subdomain_address), " for the alert."), ("subdomain", "multi_level_domain")),
            ("plus_addressing", ("Send the receipt to ", ("EMAIL", plus_address), "."), ("plus_addressing", "unusual_local_part")),
            ("spoken", ("my email is ", ("EMAIL", _spoken_email(local, domain))), ("speech_like", "spoken_email", "punctuation_free")),
            ("asr_fragment", ("uh contact me ", ("EMAIL", _spoken_email(local, domain)), " please"), ("speech_like", "spoken_email", "filler_words", "partial_utterance")),
        ]
        for variant, parts, tags in variants:
            cases.append(_case(partition, "email", index, variant, f"email-{pool_index:02d}-{variant}", parts, tags, pair_id=pair))
            index += 1
    return cases


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z+]", "", value)


def _spaced(value: str) -> str:
    compact = _compact(value).lstrip("+")
    return " ".join(compact)


def _spoken(value: str, *, filler: bool = False) -> str:
    words: list[str] = []
    inserted = False
    for character in _compact(value).lstrip("+").lower():
        words.append(DIGIT_WORDS.get(character, character))
        if filler and not inserted and len(words) == 3:
            words.append("uh")
            inserted = True
    return " ".join(words)


def _numeric_cases(partition: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    index = 1
    for pool_index, (entity_type, cue, value) in enumerate(NUMERIC_POOLS[partition], start=1):
        pair = f"{partition}-numeric-{pool_index:02d}"
        variants = [
            ("cued_formatted", (f"My {cue} is ", (entity_type, value), "."), ("written", "cued", "regional_format")),
            ("compact", (f"Record the {cue} as ", (entity_type, _compact(value)), "."), ("compact_form", "separator_change")),
            ("spaced", (f"The {cue} reads ", (entity_type, _spaced(value)), "."), ("spaced_form", "separator_change")),
            ("spoken", (f"the {cue} is ", (entity_type, _spoken(value))), ("speech_like", "spoken_digits", "punctuation_free")),
            ("uncued_context", ("For verification use ", (entity_type, value), " when requested."), ("uncued", "context_dependent")),
            ("asr_filler", (f"uh the {cue} is ", (entity_type, _spoken(value, filler=True)), " please note that"), ("speech_like", "spoken_digits", "filler_words", "punctuation_free")),
        ]
        for variant, parts, tags in variants:
            cases.append(_case(partition, "numeric_structured", index, variant, f"numeric-{pool_index:02d}-{variant}", parts, tags, pair_id=pair))
            index += 1
    return cases


def _context_cases(partition: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    index = 1
    for pool_index, (building, street, city, organisation, date) in enumerate(CONTEXT_POOLS[partition], start=1):
        pair = f"{partition}-context-{pool_index:02d}"
        variants = [
            ("address_date", ("Meet me at ", ("BUILDINGNUM", building), " ", ("STREET", street), " in ", ("CITY", city), " on ", ("DATE", date), "."), ("address", "date", "multi_entity")),
            ("lower_no_punctuation", ("meet me at ", ("BUILDINGNUM", building.lower()), " ", ("STREET", street.lower()), " in ", ("CITY", city.lower()), " on ", ("DATE", date.lower())), ("speech_like", "case_lower", "punctuation_free", "multi_entity")),
            ("organisation_address", ("The ", ("ORGANISATION", organisation), " office is at ", ("BUILDINGNUM", building), " ", ("STREET", street), ", ", ("CITY", city), "."), ("organisation", "address", "multi_entity")),
            ("organisation_date", ("I joined ", ("ORGANISATION", organisation), " on ", ("DATE", date), "."), ("organisation", "date", "contextual")),
            ("conversation", ("uh please send this to ", ("ORGANISATION", organisation.lower()), " over in ", ("CITY", city.lower()), " and mention ", ("DATE", date.lower())), ("speech_like", "filler_words", "punctuation_free", "multi_entity")),
            ("reordered", (("CITY", city), " is where ", ("ORGANISATION", organisation), " keeps the office on ", ("STREET", street), "."), ("reordered_context", "ambiguous", "multi_entity")),
        ]
        for variant, parts, tags in variants:
            cases.append(_case(partition, "contextual_multi_entity", index, variant, f"context-{pool_index:02d}-{variant}", parts, tags, pair_id=pair))
            index += 1
    return cases


def _hard_negative_cases(partition: str) -> list[EvalCase]:
    prefix = "D" if partition == "dev" else "F"
    families = [
        ("product_code", lambda i: f"Install filter cartridge {prefix}Q-{410+i}-ZX before startup."),
        ("model_number", lambda i: f"The appliance model is {prefix}M{7300+i} revision B."),
        ("temperature", lambda i: f"The chamber held steady at {18+i} degrees Celsius."),
        ("price", lambda i: f"The replacement part costs {40+i}.75 dollars before tax."),
        ("time", lambda i: f"The public lecture starts at {8+i}:30 in the main hall."),
        ("generic_date", lambda i: f"The annual festival is listed for {i+1} October on every calendar."),
        ("benign_sequence", lambda i: f"Count the samples in order {i} {i+1} {i+2} {i+3} and stop."),
        ("domain_like", lambda i: f"The documentation label says guide-{prefix.lower()}{i}.invalid without an email address."),
        ("fictional_reference", lambda i: f"In the novel the fictional artifact is called REF-{prefix}{220+i}-MYTH."),
        ("name_like_word", lambda i: f"The {['rose','amber','grant','faith','hunter','summer'][i-1]} paint shade is available in matte."),
        ("code_fragment", lambda i: f"Use config_{prefix.lower()}{i} = build(version={i + 2}) in the example code."),
        ("room_version", lambda i: f"Training is in room {200+i}, using software version {i}.4.{i+1}."),
    ]
    cases: list[EvalCase] = []
    index = 1
    for family_index, (family, factory) in enumerate(families, start=1):
        for variant in range(1, 7):
            tags = ("hard_negative", family)
            if variant in {2, 5}:
                tags += ("speech_like", "punctuation_free")
            text = factory(variant)
            if "punctuation_free" in tags:
                text = re.sub(r"[^\w\s.-]", "", text.lower())
            cases.append(_case(partition, "hard_negatives", index, family, f"hard-negative-{family_index:02d}-{variant}", (text,), tags))
            index += 1
    return cases


def _long_cases(partition: str) -> list[EvalCase]:
    filler_words = {
        "dev": "the project update remains available for careful review during each weekly meeting".split(),
        "final": "our planning summary stays accessible for detailed discussion throughout every monthly workshop".split(),
    }[partition]
    positions = [0, 120, 380, 500, 620, 699]
    cases: list[EvalCase] = []
    index = 1
    for pool_index, (entity_type, value) in enumerate(LONG_POOLS[partition], start=1):
        for position in positions:
            words = [filler_words[i % len(filler_words)] for i in range(720)]
            prefix = " ".join(words[:position])
            suffix = " ".join(words[position:])
            before = (prefix + " ") if prefix else ""
            after = (" " + suffix) if suffix else ""
            tags = ["long_input", "multi_window", f"word_position_{position}"]
            if position in {380, 500, 620}:
                tags.append("window_overlap_probe")
            if position == 699:
                tags.append("near_end")
            if position == 0:
                tags.append("near_beginning")
            cases.append(_case(
                partition,
                "long_window",
                index,
                entity_type.lower(),
                f"long-{pool_index:02d}-{position}",
                (before, (entity_type, value), after),
                tuple(tags),
                pair_id=f"{partition}-long-{pool_index:02d}",
                notes=f"Entity inserted at whitespace-token position {position} of 720 filler tokens.",
            ))
            index += 1
    return cases


def generate_partition(partition: str) -> list[EvalCase]:
    if partition not in SEEDS:
        raise ValueError(partition)
    cases = [
        *_name_cases(partition), *_email_cases(partition),
        *_numeric_cases(partition), *_context_cases(partition),
        *_hard_negative_cases(partition), *_long_cases(partition),
    ]
    random.Random(SEEDS[partition]).shuffle(cases)
    counts = Counter(case.category for case in cases)
    if counts != Counter(CATEGORY_ALLOCATION):
        raise AssertionError(f"Unexpected allocation for {partition}: {counts}")
    for case in cases:
        case.validate()
    return cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_partition(output_dir: Path, partition: str, *, refresh_final: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / f"ood_{partition}.jsonl"
    manifest_path = output_dir / f"ood_{partition}.manifest.json"
    if partition == "final" and (fixture_path.exists() or manifest_path.exists()) and not refresh_final:
        raise FileExistsError("Frozen final fixtures already exist; pass --refresh-final deliberately.")
    cases = generate_partition(partition)
    write_cases(fixture_path, cases)
    entity_values = sorted({entity.text.casefold() for case in cases for entity in case.entities if len(entity.text) > 1})
    tag_counts = Counter(tag for case in cases for tag in case.tags)
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "partition": partition,
        "seed": SEEDS[partition],
        "frozen": partition == "final",
        "case_count": len(cases),
        "category_allocation": CATEGORY_ALLOCATION,
        "hard_negative_count": sum(case.is_hard_negative for case in cases),
        "entity_value_count": len(entity_values),
        "entity_value_sha256": hashlib.sha256("\n".join(entity_values).encode("utf-8")).hexdigest(),
        "tag_counts": dict(sorted(tag_counts.items())),
        "fixture_file": fixture_path.name,
        "fixture_sha256": _sha256(fixture_path),
        "policy": (
            "Development set may guide error analysis and data design."
            if partition == "dev"
            else "Frozen final set must not guide training, augmentation, thresholding, or model selection."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "final", "all"), default="all")
    parser.add_argument("--refresh-final", action="store_true")
    args = parser.parse_args()
    partitions = ("dev", "final") if args.partition == "all" else (args.partition,)
    for partition in partitions:
        manifest = write_partition(args.output_dir, partition, refresh_final=args.refresh_final)
        print(f"{partition}: {manifest['case_count']} cases, sha256={manifest['fixture_sha256']}")


if __name__ == "__main__":
    main()
