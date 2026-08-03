"""Shared QA1.1 structural-integrity metrics.

This module implements the QA1.1 indicators described in the audit table.

Evaluation unit
---------------
One RDF entry ``e`` and one target language ``ell``.

The caller supplies the audit's existing triple-set well-formedness function
so that the original-resource audit and AuditStress use exactly the same
rules. This check is an audit-specific sanity test for the dataset's
pipe-delimited triple representation; it is not a complete RDF parser.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import re
from typing import Any


def _is_missing_scalar(value: Any) -> bool:
    """Return ``True`` for common scalar missing-value representations."""
    if value is None:
        return True

    # Python and NumPy floating-point NaN values.
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass

    # Other NaN-like scalar values generally compare unequal to themselves.
    try:
        unequal_to_self = value != value
        if isinstance(unequal_to_self, bool) and unequal_to_self:
            return True
    except Exception:
        pass

    # pandas.NA is a singleton scalar whose Boolean value is undefined.
    value_type = type(value)
    if (
        value_type.__name__ == "NAType"
        and value_type.__module__.startswith("pandas")
    ):
        return True

    return False


def normalize_space(value: Any) -> str:
    """Apply the whitespace-normalisation function ``norm(.)``.

    Missing scalar values are mapped to the empty string. Other values are
    converted to strings, consecutive whitespace is collapsed, and leading
    and trailing whitespace is removed.
    """
    if _is_missing_scalar(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_identifier(value: Any) -> str:
    """Normalise identifier whitespace without changing case or punctuation."""
    return normalize_space(value)


def _materialize_sequence(values: Sequence[Any], *, name: str) -> list[Any]:
    """Materialise a sequence while rejecting an accidental scalar string."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of values, not a string")
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable sequence") from exc



def assert_well_formedness_contract(
    triple_well_formed_fn: Callable[[Sequence[Any]], bool],
) -> None:
    """Verify that ``WF(.)`` matches the QA1.1 table's documented rules.

    The shared core is versioned separately from this module. These small
    contract cases make any future change to its well-formedness semantics
    fail visibly instead of silently desynchronising the paper and code.
    """
    cases: list[tuple[str, list[str], bool]] = [
        ("non-empty triple set", [], False),
        ("valid three-component triple", ["A | p | B"], True),
        ("missing object component", ["A | p"], False),
        ("empty object component", ["A | p | "], False),
        ("predefined placeholder", ["A | none | B"], False),
        ("residual English language tag", ['A | p | "x"@en'], False),
        ("residual datatype marker", ['A | p | "1"^^xsd:integer'], False),
        ("valid HTTP URI component", ["https://example.org/A | p | B"], True),
    ]

    failures = []
    for label, triples, expected in cases:
        observed = bool(triple_well_formed_fn(triples))
        if observed != expected:
            failures.append(
                f"{label}: expected {expected}, observed {observed}"
            )

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(
            "The supplied triple_well_formed function does not match "
            f"the QA1.1 table contract: {joined}"
        )

def qa11_entry_metrics(
    *,
    expected_triple_count: int,
    english_lids: Sequence[Any],
    target_lids: Sequence[Any],
    target_triples: Sequence[Any],
    target_texts: Sequence[Any],
    triple_well_formed_fn: Callable[[Sequence[Any]], bool],
) -> dict[str, bool]:
    """Calculate QA1.1 for one RDF entry and one target language.

    Returned fields
    ---------------
    ``target_triple_parity``
        ``P_t,ell(e) = I[|T_e^ell| = n_exp(e)]``.
    ``target_text_present``
        Passes when at least one target lexicalisation exists and every
        target lexicalisation is non-empty after whitespace normalisation.
    ``lexicalisation_id_alignment``
        Passes when a non-empty English identifier set exists and equals the
        target-language identifier set after whitespace normalisation.
    ``lexicalisation_id_unique``
        Passes when the target identifier collection is non-empty, contains
        no empty identifier, and contains no duplicate identifier.
    ``target_rdf_well_formed``
        Result of the shared audit-specific ``WF(T_e^ell)`` function.
    ``record_integrity``
        Conjunction of the five preceding conditions.
    """
    english_id_values = _materialize_sequence(
        english_lids,
        name="english_lids",
    )
    target_id_values = _materialize_sequence(
        target_lids,
        name="target_lids",
    )
    target_triple_values = _materialize_sequence(
        target_triples,
        name="target_triples",
    )
    target_text_values = _materialize_sequence(
        target_texts,
        name="target_texts",
    )

    # Every parsed target lexicalisation has exactly one identifier and one
    # text value. A mismatch indicates an invalid caller representation rather
    # than one of the five paper metrics, so fail explicitly.
    if len(target_id_values) != len(target_text_values):
        raise ValueError(
            "target_lids and target_texts must represent the same "
            "target lexicalisations and therefore have equal lengths"
        )

    english_ids = [
        normalize_identifier(value)
        for value in english_id_values
    ]
    target_ids = [
        normalize_identifier(value)
        for value in target_id_values
    ]

    # n_exp(e) is prepared by the caller from the valid declared entry size,
    # with the English modified-triple count as the documented fallback.
    try:
        if isinstance(expected_triple_count, bool):
            raise ValueError
        expected_count = int(expected_triple_count)
        expected_count_valid = expected_count >= 0
    except (TypeError, ValueError, OverflowError):
        expected_count = -1
        expected_count_valid = False

    # 1. Target triple parity:
    # P_t,ell(e) = I[|T_e^ell| = n_exp(e)].
    target_triple_parity = (
        expected_count_valid
        and len(target_triple_values) == expected_count
    )

    # 2. Target-text presence:
    # P_v,ell(e) = I[|V_e^ell| > 0 and all norm(v) != epsilon].
    target_text_present = (
        len(target_text_values) > 0
        and all(
            normalize_space(text) != ""
            for text in target_text_values
        )
    )

    # 3. Lexicalisation-ID alignment:
    # P_a,ell(e) = I[|L_e^en| > 0 and L_e^en = L_e^ell], where L denotes
    # the set of normalised identifiers. Requiring an English anchor prevents
    # two absent identifier collections from being reported as aligned.
    english_id_set = set(english_ids)
    target_id_set = set(target_ids)
    lexicalisation_id_alignment = (
        len(english_id_set) > 0
        and english_id_set == target_id_set
    )

    # 4. Lexicalisation-ID uniqueness:
    # P_u,ell(e) = I[|L-vector_e^ell| > 0, epsilon not in L-vector_e^ell,
    #                 and |L-vector_e^ell| = |L-set_e^ell|].
    lexicalisation_id_unique = (
        len(target_ids) > 0
        and all(identifier != "" for identifier in target_ids)
        and len(target_ids) == len(target_id_set)
    )

    # 5. Target triple-set well-formedness:
    # P_w,ell(e) = I[WF(T_e^ell)]. The supplied function implements the exact
    # audit rules used elsewhere; this module does not silently redefine them.
    target_rdf_well_formed = bool(
        triple_well_formed_fn(target_triple_values)
    )

    # 6. Record integrity:
    # RI_ell(e) is the conjunction/product of the five binary conditions.
    record_integrity = all(
        (
            target_triple_parity,
            target_text_present,
            lexicalisation_id_alignment,
            lexicalisation_id_unique,
            target_rdf_well_formed,
        )
    )

    return {
        "target_triple_parity": bool(target_triple_parity),
        "target_text_present": bool(target_text_present),
        "lexicalisation_id_alignment": bool(
            lexicalisation_id_alignment
        ),
        "lexicalisation_id_unique": bool(
            lexicalisation_id_unique
        ),
        "target_rdf_well_formed": bool(
            target_rdf_well_formed
        ),
        "record_integrity": bool(record_integrity),
    }
