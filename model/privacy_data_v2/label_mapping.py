"""Source-label to OnKith mapping, with a written reason for every decision.

Ported from the audited tables in
``model/data/OpenPII_1_5M_Unified_PII_Builder.ipynb`` and narrowed to the
Model V2 ontology. Two rules are absolute:

* nothing is ever guessed - an unrecognised source label is dropped, not
  approximated;
* no new entity label may enter through a source. ``ALLERGIES``, ``USERNAME``
  and ``WEIGHT`` are required deletions, ``HEIGHT`` remains absent, and every other
  out-of-ontology concept is rejected with a reason.

``PASSWORD`` maps at *label* level but is admitted at *span* level only by
``password_rules.classify_password_span``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

try:
    from .ontology import ABSENT_ENTITIES, ENTITIES, REMOVED_ENTITIES
except ImportError:  # Direct notebook/script execution from this directory.
    from ontology import ABSENT_ENTITIES, ENTITIES, REMOVED_ENTITIES

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"

STATUS_MAPPED = "mapped"
STATUS_DROPPED = "dropped"

#: Sources approved for Model V2. AI4Privacy is the base; the rest are
#: gap-fillers selected per row, never concatenated wholesale.
SOURCES = ("ai4privacy", "gretel", "nemotron", "kiji", "saudi_mena")

#: AI4Privacy/OpenPII publishes OnKith's own label names.
AI4PRIVACY_MAP: dict[str, str | None] = {entity: entity for entity in ENTITIES}
AI4PRIVACY_MAP.update({label: None for label in (*REMOVED_ENTITIES, *ABSENT_ENTITIES)})

#: Kiji uses its own vocabulary; ported verbatim from the audited builder table.
KIJI_MAP: dict[str, str | None] = {
    "FIRSTNAME": "GIVENNAME", "SURNAME": "SURNAME", "CITY": "CITY",
    "BUILDINGNUM": "BUILDINGNUM", "STREET": "STREET", "ZIP": "ZIPCODE",
    "STATE": None, "COUNTRY": "COUNTRY", "COMPANYNAME": "ORGANISATION",
    "PHONENUMBER": "TELEPHONENUM", "EMAIL": "EMAIL", "DATEOFBIRTH": "DATE",
    "URL": "URL", "DRIVERLICENSENUM": "DRIVERLICENSENUM", "SSN": "SOCIALNUM",
    "SECURITYTOKEN": None, "PASSPORTID": "PASSPORTNUM", "IBAN": "ACCOUNTNUM",
    "PASSWORD": "PASSWORD", "NATIONALID": "IDCARDNUM", "LICENSEPLATENUM": None,
    "TAXNUM": "TAXNUM", "IDCARDNUM": "IDCARDNUM", "AGE": "AGE",
    "CREDITCARDNUMBER": "CREDITCARDNUMBER", "USERNAME": None,
}

#: Snake-case aliases shared by Gretel, Nemotron and other generic schemas.
SAFE_GENERIC_MAP: dict[str, str] = {
    "first_name": "GIVENNAME", "given_name": "GIVENNAME", "last_name": "SURNAME",
    "surname": "SURNAME", "family_name": "SURNAME",
    "company_name": "ORGANISATION", "organization_name": "ORGANISATION",
    "organisation_name": "ORGANISATION", "organization": "ORGANISATION",
    "organisation": "ORGANISATION",
    "street_address": "STREET", "street": "STREET",
    "phone_number": "TELEPHONENUM", "telephone_number": "TELEPHONENUM",
    "phone": "TELEPHONENUM",
    "national_id": "IDCARDNUM", "national_id_number": "IDCARDNUM",
    "id_card_number": "IDCARDNUM",
    "ssn": "SOCIALNUM", "social_security_number": "SOCIALNUM",
    "postal_code": "ZIPCODE", "postcode": "ZIPCODE", "zip_code": "ZIPCODE",
    "account_number": "ACCOUNTNUM", "bank_account_number": "ACCOUNTNUM",
    "iban": "ACCOUNTNUM",
    "credit_card_number": "CREDITCARDNUMBER", "credit_debit_card": "CREDITCARDNUMBER",
    "date_of_birth": "DATE", "birth_date": "DATE", "date": "DATE",
    "time": "TIME",
    "city": "CITY", "country": "COUNTRY",
    "tax_id": "TAXNUM", "tax_number": "TAXNUM",
    "taxpayer_identification_number": "TAXNUM",
    "url": "URL", "email": "EMAIL", "email_address": "EMAIL", "ipv4": "IPV4",
    "age": "AGE", "gender": "GENDER", "sex": "SEX", "title": "TITLE",
    "password": "PASSWORD",
    "salary": "SALARY", "amount": "AMOUNT", "currency": "CURRENCY",
    "timezone": "TIMEZONE", "time_zone": "TIMEZONE", "bank_name": "BANKNAME",
    "building_number": "BUILDINGNUM", "building_num": "BUILDINGNUM",
    "passport_number": "PASSPORTNUM", "passport_id": "PASSPORTNUM",
    "driver_license_number": "DRIVERLICENSENUM",
    "drivers_license_number": "DRIVERLICENSENUM",
    "driving_license_number": "DRIVERLICENSENUM",
}

#: Mappings that are defensible but not exact. Selection samples and reports
#: these so a reviewer can confirm them against real rows.
MEDIUM_CONFIDENCE_ALIASES: frozenset[str] = frozenset(
    {
        "iban", "bank_account_number", "account_number",
        "national_id", "national_id_number", "id_card_number",
        "date_of_birth", "birth_date",
        "password",
    }
)

#: Every rejection carries the reason it was rejected.
DROP_REASONS: dict[str, str] = {
    "username": "required deletion",
    "user_name": "required deletion",
    "allergies": "required deletion",
    "height": "required absence (never part of the TinyBERT head)",
    "weight": "required deletion (removed from the Model V2 ontology)",
    "state": "no safe universal state/province label",
    "securitytoken": "credential/token has no universal label",
    "security_token": "credential/token has no universal label",
    "licenseplatenum": "vehicle plate is not a driver licence",
    "license_plate": "vehicle plate is not a driver licence",
    "medical_record_number": "medical identifier is not a generic ID card",
    "customer_id": "customer ID is ambiguous",
    "employee_id": "employee ID is ambiguous",
    "name": "full/generic name cannot be split into GIVENNAME/SURNAME safely",
    "full_name": "full name cannot be split into GIVENNAME/SURNAME safely",
    "address": "full/generic address cannot be narrowed safely",
    "device_identifier": "unsupported identifier",
    "bank_routing_number": "routing number is not an account number",
    "routing_number": "routing number is not an account number",
    "date_time": "composite date-time cannot map to one label",
    "unique_identifier": "identifier type is ambiguous",
    "biometric_identifier": "unsupported biometric identifier",
    "certificate_license_number": "licence type is ambiguous",
    "vehicle_identifier": "unsupported vehicle identifier",
    "coordinate": "coordinates have no universal label",
    "api_key": "credential has no universal label",
    "ipv6": "taxonomy contains IPV4 only",
    "health_plan_beneficiary_number": "health-plan ID is not safely generic",
    "swift_bic": "BIC is not a bank name",
    "cvv": "CVV is not a credit-card number",
    "pin": "generic PIN field is not a disclosed credential; see password_rules",
    "nationality": "nationality is not necessarily a country mention",
    "ip_address": "IP version is ambiguous",
}

SOURCE_MAPS: dict[str, dict[str, str | None]] = {
    "ai4privacy": AI4PRIVACY_MAP,
    "kiji": KIJI_MAP,
    "gretel": {},
    "nemotron": {},
    "saudi_mena": {entity: entity for entity in ENTITIES},
}


@dataclass(frozen=True)
class MappingDecision:
    target: str | None
    status: str
    reason: str
    confidence: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target or "",
            "status": self.status,
            "reason": self.reason,
            "confidence": self.confidence or "",
        }


def native_key(label: Any) -> str:
    value = re.sub(r"^[BI]-", "", str(label or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"[\s\-]+", "_", value).lower()


def _confidence(native_label: Any, target: str) -> str:
    """PASSWORD is always medium: it still has to clear the span-level gate."""

    if target == "PASSWORD" or native_key(native_label) in MEDIUM_CONFIDENCE_ALIASES:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_HIGH


def mapping_decision(source: str, native_label: Any) -> MappingDecision:
    """Map one native source label, never guessing."""

    raw = str(native_label or "").strip()
    supported = set(ENTITIES)
    source_map = SOURCE_MAPS.get(str(source).lower(), {})

    if raw in source_map:
        target = source_map[raw]
        if target and target in supported:
            return MappingDecision(
                target, STATUS_MAPPED, "explicit source mapping", _confidence(raw, target)
            )
        reason = DROP_REASONS.get(native_key(raw), "explicit unsafe/unsupported source label")
        return MappingDecision(None, STATUS_DROPPED, reason, None)

    upper = re.sub(r"^[BI]-", "", raw.upper())
    if upper in supported:
        return MappingDecision(upper, STATUS_MAPPED, "identity mapping", _confidence(raw, upper))
    if upper in REMOVED_ENTITIES:
        return MappingDecision(
            None, STATUS_DROPPED, f"required deletion: {REMOVED_ENTITIES[upper]}", None
        )
    if upper in ABSENT_ENTITIES:
        return MappingDecision(
            None, STATUS_DROPPED, f"required absence: {ABSENT_ENTITIES[upper]}", None
        )

    key = native_key(raw)
    if key in SAFE_GENERIC_MAP:
        target = SAFE_GENERIC_MAP[key]
        if target not in supported:
            return MappingDecision(
                None, STATUS_DROPPED, f"alias targets {target}, which Model V2 removed", None
            )
        return MappingDecision(
            target, STATUS_MAPPED, "explicit safe alias", _confidence(raw, target)
        )
    if key in DROP_REASONS:
        return MappingDecision(None, STATUS_DROPPED, DROP_REASONS[key], None)
    return MappingDecision(
        None, STATUS_DROPPED, "unmapped; no safe mapping (never guessed)", None
    )


def mapping_table(sources: Iterable[str] = SOURCES) -> list[dict[str, Any]]:
    """The full static mapping specification, for the committed audit CSV."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        natives = set(SOURCE_MAPS.get(source, {}))
        if source in {"gretel", "nemotron"}:
            natives |= set(SAFE_GENERIC_MAP) | set(DROP_REASONS)
        for native in sorted(natives):
            if (source, native) in seen:
                continue
            seen.add((source, native))
            decision = mapping_decision(source, native)
            rows.append(
                {
                    "source_dataset": source,
                    "native_label": native,
                    "onkith_label": decision.target or "",
                    "status": decision.status,
                    "confidence": decision.confidence or "",
                    "reason": decision.reason,
                    "requires_span_gate": "password_rules" if decision.target == "PASSWORD" else "",
                }
            )
    return rows


def write_mapping_csv(path: str | Any) -> dict[str, Any]:
    """Write the committed mapping audit and return a compact summary."""

    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = mapping_table()
    fields = [
        "source_dataset", "native_label", "onkith_label", "status",
        "confidence", "reason", "requires_span_gate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mapped = [row for row in rows if row["status"] == STATUS_MAPPED]
    return {
        "path": str(path),
        "rows": len(rows),
        "mapped": len(mapped),
        "dropped": len(rows) - len(mapped),
        "medium_confidence": sum(1 for row in mapped if row["confidence"] == CONFIDENCE_MEDIUM),
        "distinct_targets": len({row["onkith_label"] for row in mapped}),
    }
