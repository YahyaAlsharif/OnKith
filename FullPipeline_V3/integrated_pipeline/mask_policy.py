"""Configurable per-label masking policy.

OnKith separates two questions that used to be one:

* **what the model detects** - decided by the trained ontology;
* **what OnKith masks** - decided here, by configuration, at runtime.

Keeping them apart means an ordinary request such as "what time is it in
Riyadh" still works while the model correctly recognises ``Riyadh`` as a
``CITY``. The detection is preserved and reported; only the text substitution is
suppressed. Nothing is filtered earlier in the inference path, so entity
telemetry, span traces and evaluation stay complete.

Safety rules, in priority order:

1. **Default to masking.** A missing policy, a missing label, an empty policy or
   an unreadable policy all mask exactly as before. An existing deployment that
   has never heard of ``mask_policy`` is unaffected.
2. **Only an explicit boolean ``false`` disables masking.** Strings, numbers and
   ``null`` are configuration errors, not permission to release data.
3. **Errors are recorded, never silently applied.** Every rejected entry is kept
   in :attr:`MaskPolicy.errors` so a typo surfaces in the span trace instead of
   quietly becoming a privacy hole.

The policy is entity-type keyed and contains no label-specific logic, which is
why it survived production moving from the 67-label Model V1 contract to the
63-label Model V2 (DeBERTa-v3-xsmall) ontology without a single change here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

CONFIG_KEY = "mask_policy"


class MaskPolicy:
    """An immutable, entity-type keyed masking decision table."""

    __slots__ = ("_disabled", "_errors", "_configured")

    def __init__(
        self,
        disabled: Iterable[str] = (),
        *,
        errors: Iterable[str] = (),
        configured: bool = False,
    ) -> None:
        self._disabled = frozenset(str(label) for label in disabled)
        self._errors = tuple(errors)
        self._configured = bool(configured)

    # -- construction -------------------------------------------------------

    @classmethod
    def mask_everything(cls) -> "MaskPolicy":
        """The default and the safe fallback."""

        return cls()

    @classmethod
    def from_config(cls, raw: Any) -> "MaskPolicy":
        """Build a policy from the ``mask_policy`` configuration value.

        Never raises. A malformed policy degrades to masking everything and
        explains why in :attr:`errors`, because refusing to mask is the only
        failure mode this component must not have.
        """

        if raw is None:
            return cls.mask_everything()
        if not isinstance(raw, Mapping):
            return cls(
                errors=(
                    f"{CONFIG_KEY} must be an object mapping entity types to booleans; "
                    f"got {type(raw).__name__}. Masking everything.",
                ),
            )

        disabled: list[str] = []
        errors: list[str] = []
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{CONFIG_KEY} key {key!r} is not an entity type; ignored.")
                continue
            label = key.strip()
            # `isinstance(True, int)` is True in Python, so test the type exactly.
            if value is True:
                continue
            if value is False:
                disabled.append(label)
                continue
            errors.append(
                f"{CONFIG_KEY}[{label!r}] must be true or false; got {value!r}. "
                "Masking this label."
            )
        return cls(disabled, errors=errors, configured=True)

    @classmethod
    def from_pipeline_config(cls, config: Any) -> "MaskPolicy":
        """Read the policy from a :class:`PipelineConfig`-shaped object."""

        try:
            raw = config.pii.get(CONFIG_KEY)
        except Exception:
            return cls.mask_everything()
        return cls.from_config(raw)

    # -- decisions ----------------------------------------------------------

    def should_mask(self, entity_type: Any) -> bool:
        """Whether a detected entity type is replaced in ``masked_text``."""

        if not isinstance(entity_type, str):
            return True
        return entity_type.strip() not in self._disabled

    # -- introspection ------------------------------------------------------

    @property
    def configured(self) -> bool:
        """Whether a policy was supplied at all."""

        return self._configured

    @property
    def disabled_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._disabled))

    @property
    def errors(self) -> tuple[str, ...]:
        return self._errors

    @property
    def masks_everything(self) -> bool:
        return not self._disabled

    def as_dict(self) -> dict[str, Any]:
        """A compact description for span traces and result metadata."""

        return {
            "configured": self._configured,
            "masks_everything": self.masks_everything,
            "disabled_labels": list(self.disabled_labels),
            "errors": list(self._errors),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.masks_everything:
            return "MaskPolicy(mask everything)"
        return f"MaskPolicy(disabled={list(self.disabled_labels)})"


#: Shared default so callers can avoid constructing one per request.
MASK_EVERYTHING = MaskPolicy.mask_everything()


def resolve(policy: Any) -> MaskPolicy:
    """Coerce ``None``/mapping/:class:`MaskPolicy` into a usable policy."""

    if isinstance(policy, MaskPolicy):
        return policy
    if policy is None:
        return MASK_EVERYTHING
    return MaskPolicy.from_config(policy)
