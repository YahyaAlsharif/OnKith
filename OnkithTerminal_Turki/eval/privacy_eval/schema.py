"""Small, strict schemas shared by fixtures, runners, and tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EntitySpan:
    entity_type: str
    start: int
    end: int
    text: str

    @classmethod
    def from_dict(cls, value: dict) -> "EntitySpan":
        return cls(
            entity_type=str(value["entity_type"]),
            start=int(value["start"]),
            end=int(value["end"]),
            text=str(value["text"]),
        )

    def validate(self, source: str) -> None:
        if not self.entity_type or self.entity_type == "O":
            raise ValueError("Entity type must be a non-O label.")
        if self.start < 0 or self.end <= self.start or self.end > len(source):
            raise ValueError(f"Invalid entity bounds: {self!r}")
        if source[self.start : self.end] != self.text:
            raise ValueError(f"Entity text is not source-aligned: {self!r}")


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    partition: str
    category: str
    subcategory: str
    template_id: str
    text: str
    entities: tuple[EntitySpan, ...]
    tags: tuple[str, ...]
    pair_id: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "EvalCase":
        case = cls(
            case_id=str(value["case_id"]),
            partition=str(value["partition"]),
            category=str(value["category"]),
            subcategory=str(value["subcategory"]),
            template_id=str(value["template_id"]),
            text=str(value["text"]),
            entities=tuple(EntitySpan.from_dict(item) for item in value["entities"]),
            tags=tuple(str(item) for item in value.get("tags", [])),
            pair_id=value.get("pair_id"),
            notes=str(value.get("notes", "")),
        )
        case.validate()
        return case

    @property
    def is_hard_negative(self) -> bool:
        return not self.entities

    def validate(self) -> None:
        if self.partition not in {"dev", "final"}:
            raise ValueError(f"Unexpected partition {self.partition!r}")
        if not self.case_id or not self.text:
            raise ValueError("Cases require a non-empty ID and text.")
        previous_end = -1
        for entity in sorted(self.entities, key=lambda item: (item.start, item.end)):
            entity.validate(self.text)
            if entity.start < previous_end:
                raise ValueError(f"Overlapping gold entities in {self.case_id}")
            previous_end = entity.end

    def to_dict(self) -> dict:
        value = asdict(self)
        value["entities"] = [asdict(item) for item in self.entities]
        value["tags"] = list(self.tags)
        return value


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid fixture line {line_number} in {path}") from exc
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return cases


def write_cases(path: Path, cases: Iterable[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            case.validate()
            stream.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
