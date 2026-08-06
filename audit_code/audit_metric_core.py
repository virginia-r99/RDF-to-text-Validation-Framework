"""Shared metrics for the original-resource audit and AuditStress runner.

The purpose of this module is to prevent metric drift. The corrected original
audit notebooks and the aligned AuditStress runner import the same functions.

Compatibility version: audit-text-aligned-2.7
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Sequence
import html
import json
import math
import re
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

try:
    import joblib
except ImportError:
    joblib = None

CORE_COMPATIBILITY = "audit-text-aligned-2.7"
RANDOM_SEED = 42
LANG_LABELS = {"en": "English", "es": "Spanish", "ca": "Catalan"}


def normalize_space(value: object) -> str:
    """Collapse whitespace and map missing scalar values to the empty string."""
    if value is None:
        return ""

    # Handle NumPy/pandas missing scalar values without applying a Boolean
    # conversion to array-like objects.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(value: object) -> str:
    value = html.unescape(normalize_space(value)).strip()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"@[A-Za-z-]+\s*$", "", value)
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\n\r\"'").casefold()


# ---------------------------------------------------------------------------
# QA1.2 — recurring-component consistency
# ---------------------------------------------------------------------------
#
# This section implements the three indicators reported in
# Table~\ref{tab:audit-framework-qa12}:
#
#   1. D_{ell,r}(x): dominant mapping rate;
#   2. D_{w,ell,r}: weighted dominant mapping rate; and
#   3. M_{ell,r}(x): derived mapping-review flag.
#
# Evaluation unit
# ---------------
# A component-level mapping observation is defined by:
#
#   * target language ell;
#   * reported component class r;
#   * normalised English source component x; and
#   * normalised target form y.
#
# The original-resource notebook constructs one mapping row for every
# positionally aligned English–target triple component. Subject is aligned
# with subject, predicate with predicate, and object with object. Only entries
# whose source and target triple sets are structurally aligned are eligible.
#
# Empty normalised source or target forms are excluded here. Missing,
# malformed, or structurally invalid components are handled by QA1.1 rather
# than being interpreted as alternative mappings in QA1.2.
#
# Recurrence threshold
# --------------------
# QA1.2 evaluates only recurring source components:
#
#     n_{ell,r}(x) >= rho
#
# where rho is ``minimum_occurrences``. The default and reported value is
# rho = 2.
# ---------------------------------------------------------------------------


# Stable output schema for one recurring source-component profile.
#
# Table correspondence:
#
#   language                  -> ell
#   component_type            -> r
#   source_normalized         -> x
#   occurrences               -> n_{ell,r}(x)
#   dominant_mapping_rate     -> D_{ell,r}(x)
#   mapping_review_signal     -> M_{ell,r}(x)
#
# The remaining columns provide traceability and interpretable diagnostics;
# they are not additional independent metrics.
QA12_PROFILE_COLUMNS = [
    "language",
    "language_label",
    "component_type",
    "source_normalized",
    "source_example",
    "minimum_occurrences",
    "occurrences",
    "target_form_count",
    "dominant_target_form",
    "dominant_target_count",
    "dominant_mapping_rate",
    "mapping_review_signal",
    "target_forms",
]


def qa12_profile_normalized_forms(
    target_forms: Sequence[Any],
    *,
    minimum_occurrences: int = 2,
) -> dict[str, Any] | None:
    """Calculate D and M for one recurring source component.

    Evaluation unit
    ---------------
    This function receives all normalised target forms ``y`` observed for
    one fixed combination of:

    * target language ``ell``;
    * component class ``r``; and
    * normalised English source component ``x``.

    The caller performs that grouping. This function calculates the
    component-specific statistics inside the group.

    Table correspondence
    --------------------
    Mapping frequency:

        c_{ell,r}(x,y)
            = number of times target form y is observed for source x

    Total occurrence count:

        n_{ell,r}(x)
            = sum_y c_{ell,r}(x,y)

    Dominant mapping rate:

        D_{ell,r}(x)
            = max_y c_{ell,r}(x,y) / n_{ell,r}(x)

    Derived review flag:

        M_{ell,r}(x)
            = I[D_{ell,r}(x) < 1]

    The flag is equivalent to observing more than one non-empty normalised
    target form. It identifies items for contextual inspection; it is not an
    automatic error label or an independent metric.

    Recurrence rule
    ---------------
    The function returns ``None`` when:

        n_{ell,r}(x) < rho

    where ``rho = minimum_occurrences``. QA1.2 requires rho >= 2 because a
    source component observed once cannot exhibit recurring consistency.
    """

    # QA1.2 is explicitly restricted to recurring source components.
    # A threshold below two would contradict the table definition.
    if minimum_occurrences < 2:
        raise ValueError(
            "minimum_occurrences must be at least 2 because QA1.2 "
            "evaluates recurring source components."
        )

    # Remove missing or empty target forms after superficial whitespace
    # normalisation. These are not valid mapping alternatives and are handled
    # by the structural QA1.1 checks.
    #
    # Each remaining list element is one observed target form y.
    forms = []
    for form in target_forms:
        normalized_form = normalize_space(form)
        if normalized_form:
            forms.append(normalized_form)

    # Implements:
    #
    #     n_{ell,r}(x) = sum_y c_{ell,r}(x,y)
    #
    # Since ``forms`` contains one item per observed mapping occurrence, its
    # length is the total mapping count for source component x.
    occurrence_count = len(forms)

    # Apply the recurrence condition n_{ell,r}(x) >= rho.
    if occurrence_count < minimum_occurrences:
        return None

    # ``Counter`` constructs c_{ell,r}(x,y) for every distinct target form y.
    #
    # Example:
    #     forms = ["forma", "forma", "variant"]
    #     counts = {"forma": 2, "variant": 1}
    counts = Counter(forms)

    # Select max_y c_{ell,r}(x,y). Frequency is the primary ordering key.
    # Lexical ordering is used only to break ties deterministically so that
    # repeated executions produce the same diagnostic ``dominant_target_form``.
    # Tie-breaking does not change D_{ell,r}(x).
    dominant_target_form, dominant_target_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]

    # Implements:
    #
    #     D_{ell,r}(x)
    #       = max_y c_{ell,r}(x,y) / n_{ell,r}(x)
    #
    # Interpretation:
    #     D = 1    -> every occurrence maps to the same normalised form;
    #     D < 1    -> at least two normalised target forms were observed.
    dominant_mapping_rate = (
        dominant_target_count / occurrence_count
    )

    # Implements:
    #
    #     M_{ell,r}(x) = I[D_{ell,r}(x) < 1]
    #
    # Counting distinct target forms is mathematically equivalent to testing
    # D < 1, but avoids relying on a floating-point comparison.
    mapping_review_signal = len(counts) > 1

    # Return the two table indicators together with their supporting counts.
    return {
        # n_{ell,r}(x)
        "occurrences": int(occurrence_count),

        # Number of target forms with c_{ell,r}(x,y) > 0.
        "target_form_count": int(len(counts)),

        # Diagnostic argmax_y c_{ell,r}(x,y).
        "dominant_target_form": dominant_target_form,

        # max_y c_{ell,r}(x,y)
        "dominant_target_count": int(
            dominant_target_count
        ),

        # D_{ell,r}(x)
        "dominant_mapping_rate": float(
            dominant_mapping_rate
        ),

        # M_{ell,r}(x)
        "mapping_review_signal": bool(
            mapping_review_signal
        ),

        # Full c_{ell,r}(x,y) distribution for contextual review.
        "target_forms": dict(counts),
    }


def qa12_build_mapping_profiles(
    mappings: pd.DataFrame,
    *,
    minimum_occurrences: int = 2,
) -> pd.DataFrame:
    """Build one QA1.2 profile for every recurring source component.

    Input representation
    --------------------
    ``mappings`` must contain one row per positionally aligned source–target
    component occurrence. The original-resource notebook creates these rows
    by:

    1. retaining structurally aligned source and target triple sets;
    2. aligning triples by their position in the entry;
    3. aligning subject, predicate, and object by role; and
    4. normalising each source and target component.

    Required columns
    ----------------
    language
        Target language ``ell``.

    language_label
        Human-readable label used only for reporting.

    component_type
        Reported component class ``r``.

    source_normalized
        Normalised English source component ``x``.

    target_normalized
        Normalised target form ``y``.

    Output
    ------
    One row is returned for each member of:

        X_{ell,r}^{(rho)}
            = {x : n_{ell,r}(x) >= rho}

    Each row contains n_{ell,r}(x), D_{ell,r}(x), M_{ell,r}(x), and
    supporting diagnostic information.
    """

    # Enforce the recurrence definition used in the table.
    if minimum_occurrences < 2:
        raise ValueError(
            "minimum_occurrences must be at least 2."
        )

    # Verify that the DataFrame can represent ell, r, x, and y.
    required_columns = {
        "language",
        "language_label",
        "component_type",
        "source_normalized",
        "target_normalized",
    }
    missing = sorted(
        required_columns - set(mappings.columns)
    )
    if missing:
        raise ValueError(
            "QA1.2 mapping data is missing required columns: "
            f"{missing}"
        )

    usable = mappings.copy()

    # Apply the same superficial whitespace normalisation before grouping.
    # The upstream notebook is responsible for the semantic label
    # normalisation that produces x and y; this step prevents whitespace-only
    # variants or missing values from becoming artificial mapping forms.
    usable["source_normalized"] = usable[
        "source_normalized"
    ].map(normalize_space)
    usable["target_normalized"] = usable[
        "target_normalized"
    ].map(normalize_space)

    # Empty source or target components do not define valid mappings.
    usable = usable[
        usable["source_normalized"].ne("")
        & usable["target_normalized"].ne("")
    ].copy()

    profile_rows: list[dict[str, Any]] = []

    # Group all observations belonging to one fixed (ell, r, x).
    grouping_columns = [
        "language",
        "language_label",
        "component_type",
        "source_normalized",
    ]

    for keys, group in usable.groupby(
        grouping_columns,
        sort=True,
        dropna=False,
    ):
        (
            language,
            language_label,
            component_type,
            source_normalized,
        ) = keys

        # Calculate n_{ell,r}(x), D_{ell,r}(x), and M_{ell,r}(x) from
        # the complete list of target forms y observed inside this group.
        profile = qa12_profile_normalized_forms(
            group["target_normalized"].tolist(),
            minimum_occurrences=minimum_occurrences,
        )

        # Exclude source components outside X_{ell,r}^{(rho)}.
        if profile is None:
            continue

        # Retain the most frequent raw source spelling as an interpretable
        # example. This is a diagnostic field and does not affect any metric.
        if (
            "source_raw" in group.columns
            and group["source_raw"].map(
                normalize_space
            ).ne("").any()
        ):
            source_examples = (
                group["source_raw"]
                .map(normalize_space)
                .loc[lambda values: values.ne("")]
                .value_counts()
            )
            source_example = str(
                source_examples.index[0]
            )
        else:
            source_example = str(source_normalized)

        profile_rows.append({
            # ell
            "language": language,

            # Reporting label for ell.
            "language_label": language_label,

            # r
            "component_type": component_type,

            # x
            "source_normalized": source_normalized,

            # Diagnostic raw spelling of x.
            "source_example": source_example,

            # rho
            "minimum_occurrences": int(
                minimum_occurrences
            ),

            # n_{ell,r}(x), D_{ell,r}(x), M_{ell,r}(x), and diagnostics.
            **profile,

            # Store c_{ell,r}(x,y) as a deterministic JSON object for export.
            "target_forms": json.dumps(
                profile["target_forms"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        })

    # Preserve a stable schema even when no source component meets rho.
    if not profile_rows:
        return pd.DataFrame(
            columns=QA12_PROFILE_COLUMNS
        )

    return pd.DataFrame(profile_rows)[
        QA12_PROFILE_COLUMNS
    ]


def qa12_summarize_mapping_profiles(
    profiles: pd.DataFrame,
    *,
    additional_group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Calculate the weighted dominant mapping rate D_w.

    Default reporting strata
    -------------------------
    The table defines one aggregate for every target-language/component-class
    pair ``(ell, r)``. AuditStress may add an extra reporting stratum, such as
    ``group_kind``, without changing the metric.

    Table correspondence
    --------------------
    The recurrent source-component set is:

        X_{ell,r}^{(rho)}
            = {x : n_{ell,r}(x) >= rho}

    The weighted dominant mapping rate is:

        D_{w,ell,r}
            = [
                sum_{x in X_{ell,r}^{(rho)}}
                    n_{ell,r}(x) D_{ell,r}(x)
              ]
              /
              [
                sum_{x in X_{ell,r}^{(rho)}}
                    n_{ell,r}(x)
              ]

    The output also reports:

    ``mapping_review_rate``
        Mean of M_{ell,r}(x) across recurring source components.

    ``items_selected_for_review``
        Number of recurring source components for which M_{ell,r}(x) = 1.

    These are descriptive summaries of the derived review flags and are not
    additional independent metrics in the QA1.2 table.
    """

    # The table's default strata are target language ell and component
    # class r. Additional columns only partition the same calculation for
    # benchmark reporting.
    base_group_columns = [
        "language",
        "language_label",
        "component_type",
    ]
    group_columns = [
        *additional_group_columns,
        *base_group_columns,
    ]

    # Stable output schema.
    output_columns = [
        *group_columns,
        "recurring_source_items",
        "mapping_observations",
        "weighted_dominant_mapping_rate",
        "mapping_review_rate",
        "items_selected_for_review",
    ]

    if profiles.empty:
        return pd.DataFrame(columns=output_columns)

    # Verify that each profile supplies n, D, and M for a recurring source
    # component, together with the requested reporting strata.
    required_columns = {
        *group_columns,
        "occurrences",
        "dominant_mapping_rate",
        "mapping_review_signal",
    }
    missing = sorted(
        required_columns - set(profiles.columns)
    )
    if missing:
        raise ValueError(
            "QA1.2 profile data is missing required columns: "
            f"{missing}"
        )

    summary_rows: list[dict[str, Any]] = []

    # Calculate D_w separately for each requested reporting stratum.
    for keys, group in profiles.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))

        # n_{ell,r}(x): weights assigned to recurring source components.
        weights = pd.to_numeric(
            group["occurrences"],
            errors="coerce",
        ).to_numpy(dtype=float)

        # D_{ell,r}(x): component-specific dominant mapping rates.
        rates = pd.to_numeric(
            group["dominant_mapping_rate"],
            errors="coerce",
        ).to_numpy(dtype=float)

        # Only finite profiles with a positive occurrence count can
        # contribute to the weighted aggregate.
        valid = (
            np.isfinite(weights)
            & np.isfinite(rates)
            & (weights > 0)
        )
        total_weight = float(weights[valid].sum())

        # Implements D_{w,ell,r}.
        weighted_rate = (
            float(
                np.sum(
                    weights[valid] * rates[valid]
                )
                / total_weight
            )
            if total_weight > 0
            else np.nan
        )

        # M_{ell,r}(x) values used only for descriptive review summaries.
        review_values = (
            group["mapping_review_signal"]
            .astype(bool)
        )

        summary_rows.append({
            **key_values,

            # |X_{ell,r}^{(rho)}|
            "recurring_source_items": int(len(group)),

            # sum_x n_{ell,r}(x)
            "mapping_observations": int(
                pd.to_numeric(
                    group["occurrences"],
                    errors="coerce",
                ).fillna(0).sum()
            ),

            # D_{w,ell,r}
            "weighted_dominant_mapping_rate": (
                weighted_rate
            ),

            # Descriptive mean of M_{ell,r}(x).
            "mapping_review_rate": float(
                review_values.mean()
            ),

            # Descriptive count of source components with M_{ell,r}(x) = 1.
            "items_selected_for_review": int(
                review_values.sum()
            ),
        })

    return pd.DataFrame(summary_rows)[
        output_columns
    ]


def assert_qa12_metric_contract() -> None:
    """Verify that the implementation matches the QA1.2 table.

    The contract checks the defining cases for all three reported
    indicators:

    1. A recurring source component with one target form has
       D_{ell,r}(x) = 1 and M_{ell,r}(x) = 0.

    2. A recurring source component with target-form counts {2, 1} has
       D_{ell,r}(x) = 2/3 and M_{ell,r}(x) = 1.

    3. A source component observed fewer than rho times is excluded from
       X_{ell,r}^{(rho)}.

    4. The weighted summary uses n_{ell,r}(x) as the weight:
       one profile with n=3 and D=2/3 plus one profile with n=2 and D=1
       must produce D_w = 4/5.
    """

    # Consistent recurring mapping:
    # c(x, "forma") = 2, n(x) = 2, D(x) = 1, M(x) = 0.
    consistent = qa12_profile_normalized_forms(
        ["forma", "forma"],
        minimum_occurrences=2,
    )

    # Variable recurring mapping:
    # c(x, "forma") = 2, c(x, "variant") = 1,
    # n(x) = 3, D(x) = 2/3, M(x) = 1.
    variable = qa12_profile_normalized_forms(
        ["forma", "forma", "variant"],
        minimum_occurrences=2,
    )

    # Non-recurring mapping:
    # n(x) = 1 < rho, so x is not in X_{ell,r}^{(rho)}.
    excluded = qa12_profile_normalized_forms(
        ["single"],
        minimum_occurrences=2,
    )

    if consistent is None:
        raise AssertionError(
            "A recurring consistent mapping was excluded."
        )
    if not math.isclose(
        consistent["dominant_mapping_rate"],
        1.0,
    ):
        raise AssertionError(
            "Consistent mappings must have D = 1."
        )
    if consistent["mapping_review_signal"]:
        raise AssertionError(
            "Consistent mappings must not trigger M."
        )

    if variable is None:
        raise AssertionError(
            "A recurring variable mapping was excluded."
        )
    if not math.isclose(
        variable["dominant_mapping_rate"],
        2.0 / 3.0,
    ):
        raise AssertionError(
            "Variable mapping D does not match max(c)/n."
        )
    if not variable["mapping_review_signal"]:
        raise AssertionError(
            "More than one target form must trigger M."
        )

    if excluded is not None:
        raise AssertionError(
            "Non-recurring mappings must be excluded."
        )

    # Verify the weighted dominant mapping rate:
    #
    #     D_w = [3(2/3) + 2(1)] / (3 + 2) = 4/5.
    weighted_profiles = pd.DataFrame([
        {
            "language": "es",
            "language_label": "Spanish",
            "component_type": "predicate",
            "occurrences": variable["occurrences"],
            "dominant_mapping_rate": variable[
                "dominant_mapping_rate"
            ],
            "mapping_review_signal": variable[
                "mapping_review_signal"
            ],
        },
        {
            "language": "es",
            "language_label": "Spanish",
            "component_type": "predicate",
            "occurrences": consistent["occurrences"],
            "dominant_mapping_rate": consistent[
                "dominant_mapping_rate"
            ],
            "mapping_review_signal": consistent[
                "mapping_review_signal"
            ],
        },
    ])
    weighted_summary = qa12_summarize_mapping_profiles(
        weighted_profiles
    )
    if len(weighted_summary) != 1:
        raise AssertionError(
            "The weighted QA1.2 contract must produce one summary row."
        )

    weighted_rate = float(
        weighted_summary.loc[
            0,
            "weighted_dominant_mapping_rate",
        ]
    )
    if not math.isclose(weighted_rate, 4.0 / 5.0):
        raise AssertionError(
            "Weighted D_w does not use occurrence counts as weights."
        )

def split_triple(value: object) -> tuple[str, str, str]:
    parts = [part.strip() for part in str(value or "").split("|")]
    if len(parts) < 3:
        return "", "", ""
    return parts[0], parts[1], "|".join(parts[2:]).strip()


def clean_rdf_label(value: object) -> str:
    value = html.unescape(str(value or "")).strip()
    value = re.sub(r"\^\^xsd:[A-Za-z]+\s*$", "", value)
    value = re.sub(r"@[A-Za-z-]+\s*$", "", value)
    value = value.strip("\"' ")
    value = value.replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def serialize_triple(triple: str) -> str:
    subject, predicate, obj = map(clean_rdf_label, split_triple(triple))
    return f"[S] {subject} [P] {predicate} [O] {obj}"


def serialize_triple_set(triples: list[str]) -> str:
    # This deliberately matches the final AuditStress runner.
    return " ".join(serialize_triple(triple) for triple in triples)


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def textual_units(text: object) -> list[str]:
    """Segment a verbalisation into non-empty sentence-level units."""
    normalized = normalize_space(text)
    if not normalized:
        return []
    units = [
        unit.strip()
        for unit in SENTENCE_BOUNDARY_RE.split(normalized)
        if unit.strip()
    ]
    return units or [normalized]


class SemanticScorer:
    def __init__(
        self,
        backend: str,
        model_name: str,
        batch_size: int,
    ):
        self.backend = backend
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.vectorizer = None

        if backend == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers to run the final QA2 audit."
                ) from exc
            self.model = SentenceTransformer(model_name)
        elif backend == "tfidf":
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=200_000,
                norm="l2",
                dtype=np.float32,
            )
        else:
            raise ValueError(
                "semantic backend must be sentence_transformer or tfidf"
            )

    @property
    def description(self) -> str:
        if self.backend == "sentence_transformer":
            return self.model_name
        return "character TF-IDF smoke-test backend"

    def fit(self, texts: list[str]) -> None:
        if self.backend == "tfidf":
            self.vectorizer.fit([str(text or "") for text in texts])

    def encode(self, texts: list[str]):
        texts = [str(text or "") for text in texts]
        if self.backend == "sentence_transformer":
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype(np.float32)
        return self.vectorizer.transform(texts)

    def pairwise(self, left: pd.Series, right: pd.Series) -> np.ndarray:
        left = pd.Series(left, dtype="string").fillna("").astype(str)
        right = pd.Series(right, dtype="string").fillna("").astype(str)
        if len(left) != len(right):
            raise ValueError("left and right must contain the same number of rows")
        unique = pd.Index(pd.unique(pd.concat([left, right], ignore_index=True)))
        embeddings = self.encode(unique.tolist())
        positions = pd.Series(np.arange(len(unique)), index=unique)
        left_index = positions.loc[left].to_numpy()
        right_index = positions.loc[right].to_numpy()
        if sparse.issparse(embeddings):
            scores = np.asarray(
                embeddings[left_index]
                .multiply(embeddings[right_index])
                .sum(axis=1)
            ).ravel()
        else:
            scores = np.einsum(
                "ij,ij->i",
                embeddings[left_index],
                embeddings[right_index],
            )

        # The supported backends return L2-normalised vectors, so this dot
        # product is cosine similarity and is mathematically bounded by
        # [-1, 1]. Float32 accumulation can still produce tiny overshoots
        # such as 1.000000119. Clipping removes only that round-off.
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(
                "Semantic similarity produced a non-finite value."
            )
        return np.clip(scores, -1.0, 1.0)


def assert_semantic_cosine_range_contract() -> None:
    """Verify that harmless float32 cosine overshoots are clipped."""

    scorer = object.__new__(SemanticScorer)
    scorer.backend = "sentence_transformer"
    scorer.model_name = "synthetic-contract"
    scorer.batch_size = 1
    scorer.model = None
    scorer.vectorizer = None

    def synthetic_encode(texts: list[str]) -> np.ndarray:
        return np.tile(
            np.array([[1.0000001, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    scorer.encode = synthetic_encode
    scores = scorer.pairwise(
        pd.Series(["same"]),
        pd.Series(["same"]),
    )

    if len(scores) != 1 or float(scores[0]) != 1.0:
        raise AssertionError(
            "Cosine similarities must be clipped to [-1, 1]."
        )


MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
    "gener": 1, "febrer": 2, "marc": 3, "març": 3, "abril": 4,
    "maig": 5, "juny": 6, "juliol": 7, "agost": 8, "setembre": 9,
    "octubre": 10, "novembre": 11, "desembre": 12,
}


def canonical_numeric_atoms(
    value: object,
    language: str = "en",
) -> set[str]:
    normalized = normalize_label(value)
    atoms: set[str] = set()

    for compound in re.findall(
        r"(?<!\w)\d+(?:[-/:]\d+)+(?!\w)",
        normalized,
    ):
        atoms.update(str(int(part)) for part in re.findall(r"\d+", compound))
    normalized = re.sub(
        r"(?<!\w)\d+(?:[-/:]\d+)+(?!\w)",
        " ",
        normalized,
    )

    for token in re.findall(
        r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)",
        normalized,
    ):
        token = token.strip()
        if not token:
            continue
        if "." in token and "," in token:
            decimal = "." if token.rfind(".") > token.rfind(",") else ","
            thousands = "," if decimal == "." else "."
            canonical = token.replace(thousands, "").replace(decimal, ".")
        elif "." in token or "," in token:
            separator = "." if "." in token else ","
            pieces = token.split(separator)
            final_len = len(pieces[-1])
            looks_thousands = (
                len(pieces) > 2
                or (
                    final_len == 3
                    and (
                        (
                            language in {"es", "ca"}
                            and separator == "."
                        )
                        or (
                            language == "en"
                            and separator == ","
                        )
                    )
                )
            )
            canonical = (
                "".join(pieces)
                if looks_thousands
                else token.replace(separator, ".")
            )
        else:
            canonical = token
        try:
            parsed = float(canonical)
            atoms.add(
                str(int(parsed))
                if parsed.is_integer()
                else str(parsed).rstrip("0").rstrip(".")
            )
        except ValueError:
            atoms.add(canonical)

    for token in normalized.split():
        if token in MONTH_NAMES:
            atoms.add(str(MONTH_NAMES[token]))
    return atoms


def literal_retention(
    triples: list[str],
    text: str,
    language: str = "en",
) -> tuple[float, int]:
    """Calculate QA2.2 target triple-to-text literal retention.

    The atom extractor A(T) is applied to every subject, predicate, and object
    component in the complete triple set. A(v) is applied to the complete
    verbalisation. The score is undefined when A(T) is empty.
    """
    triple_atoms = literal_atoms_from_triples(
        triples,
        language,
    )
    if not triple_atoms:
        return np.nan, 0

    text_atoms = canonical_numeric_atoms(text, language)
    return (
        len(triple_atoms & text_atoms) / len(triple_atoms),
        len(triple_atoms),
    )

def literal_atoms_from_triples(
    triples: list[str],
    language: str = "en",
) -> set[str]:
    """Extract canonical numeric/date atoms from a complete triple set.

    QA2.1 defines A(T) over the complete triple set, so subject, predicate,
    and object components are all inspected. In the WebNLG data, eligible
    atoms normally occur in subjects or objects, but including predicates
    keeps the implementation faithful to the table definition.
    """
    atoms: set[str] = set()
    for triple in triples:
        for component in split_triple(triple):
            atoms |= canonical_numeric_atoms(component, language)
    return atoms


def source_target_literal_preservation(
    source_triples: list[str],
    target_triples: list[str],
    target_language: str,
) -> tuple[float, int]:
    source_atoms = literal_atoms_from_triples(source_triples, "en")
    if not source_atoms:
        return np.nan, 0
    target_atoms = literal_atoms_from_triples(
        target_triples,
        target_language,
    )
    return (
        len(source_atoms & target_atoms) / len(source_atoms),
        len(source_atoms),
    )


def predicate_component_diagnostics(
    source_triples_column: pd.Series,
    target_triples_column: pd.Series,
    scorer: SemanticScorer,
) -> pd.DataFrame:
    """Calculate the QA2.1 predicate diagnostics for aligned instances.

    ``PredSim`` is defined only when both triple sets are non-empty, contain
    the same number of triples, and every aligned position supplies a
    non-empty normalised predicate. This prevents Python ``zip`` truncation
    from silently changing the denominator n_e in the table.

    The structured-predicate review flag U is one when at least one aligned
    English predicate is camelCase and remains identical to the target
    predicate after applying the shared label normalisation N(.).
    """
    source_series = pd.Series(source_triples_column).reset_index(drop=True)
    target_series = pd.Series(target_triples_column).reset_index(drop=True)
    if len(source_series) != len(target_series):
        raise ValueError(
            "source and target triple columns must contain the same number "
            "of rows"
        )

    row_count = len(source_series)
    similarity = np.full(row_count, np.nan, dtype=float)
    review_flag = np.full(row_count, np.nan, dtype=float)
    pair_count = np.zeros(row_count, dtype=int)
    alignment_valid = np.zeros(row_count, dtype=bool)
    expanded: list[dict[str, Any]] = []

    for row_index, (source_triples, target_triples) in enumerate(
        zip(source_series, target_series)
    ):
        source_values = list(source_triples or [])
        target_values = list(target_triples or [])

        if (
            not source_values
            or len(source_values) != len(target_values)
        ):
            continue

        normalized_pairs: list[tuple[str, str]] = []
        structured_unchanged = False
        valid = True

        for source_triple, target_triple in zip(
            source_values, target_values
        ):
            source_raw = split_triple(source_triple)[1]
            target_raw = split_triple(target_triple)[1]
            source_normalized = normalize_label(source_raw)
            target_normalized = normalize_label(target_raw)

            if not source_normalized or not target_normalized:
                valid = False
                break

            normalized_pairs.append(
                (source_normalized, target_normalized)
            )
            if (
                source_normalized == target_normalized
                and bool(CAMEL_CASE_RE.search(source_raw))
            ):
                structured_unchanged = True

        if not valid or len(normalized_pairs) != len(source_values):
            continue

        alignment_valid[row_index] = True
        pair_count[row_index] = len(normalized_pairs)
        review_flag[row_index] = float(structured_unchanged)

        for source_predicate, target_predicate in normalized_pairs:
            expanded.append({
                "row_index": row_index,
                "source_predicate": source_predicate,
                "target_predicate": target_predicate,
            })

    if expanded:
        frame = pd.DataFrame(expanded)
        frame["similarity"] = scorer.pairwise(
            frame["source_predicate"],
            frame["target_predicate"],
        )
        means = frame.groupby("row_index")["similarity"].mean()
        for row_index, value in means.items():
            similarity[int(row_index)] = float(value)

    return pd.DataFrame({
        "predicate_similarity": similarity,
        "structured_predicate_review_flag": review_flag,
        "predicate_pair_count": pair_count,
        "predicate_alignment_valid": alignment_valid,
    })


def predicate_component_similarity(
    source_triples_column: pd.Series,
    target_triples_column: pd.Series,
    scorer: SemanticScorer,
) -> np.ndarray:
    """Backward-compatible accessor for QA2.1 PredSim."""
    return predicate_component_diagnostics(
        source_triples_column,
        target_triples_column,
        scorer,
    )["predicate_similarity"].to_numpy(dtype=float)


def minimum_triple_coverage(
    triples_column: pd.Series,
    text_column: pd.Series,
    scorer: SemanticScorer,
) -> np.ndarray:
    """Calculate C_min for each aligned RDF-to-text instance.

    Each target triple is serialised independently with explicit [S], [P],
    and [O] markers and compared with the complete target verbalisation. The
    least-supported triple score is retained. Rows without a non-empty triple
    set or a non-empty target verbalisation receive NaN.
    """
    triples_series = pd.Series(triples_column).reset_index(drop=True)
    text_series = pd.Series(text_column).reset_index(drop=True)
    if len(triples_series) != len(text_series):
        raise ValueError(
            "triples and text columns must contain the same number of rows"
        )

    rows: list[dict[str, Any]] = []
    result = np.full(len(triples_series), np.nan, dtype=float)

    for row_index, (triples, text) in enumerate(
        zip(triples_series, text_series)
    ):
        triple_values = list(triples or [])
        normalized_text = normalize_space(text)
        if not triple_values or not normalized_text:
            continue

        serialised = [
            serialize_triple(triple)
            for triple in triple_values
        ]
        if any(not normalize_space(value) for value in serialised):
            continue

        for triple in serialised:
            rows.append({
                "row_index": row_index,
                "triple": triple,
                "text": normalized_text,
            })

    if not rows:
        return result

    expanded = pd.DataFrame(rows)
    expanded["support"] = scorer.pairwise(
        expanded["triple"],
        expanded["text"],
    )
    minima = expanded.groupby("row_index")["support"].min()
    for row_index, value in minima.items():
        result[int(row_index)] = float(value)
    return result


def min_support(
    triples_column: pd.Series,
    text_column: pd.Series,
    scorer: SemanticScorer,
) -> np.ndarray:
    """Backward-compatible alias for minimum_triple_coverage."""
    return minimum_triple_coverage(
        triples_column,
        text_column,
        scorer,
    )


def minimum_text_groundedness(
    triples_column: pd.Series,
    text_column: pd.Series,
    scorer: SemanticScorer,
) -> np.ndarray:
    """Calculate G_min for each aligned RDF-to-text instance.

    The target verbalisation is segmented into sentence-level units. Each unit
    receives the similarity of its best-matching target triple, and the least
    supported unit is retained. Rows without target triples or non-empty
    textual units receive NaN.
    """
    triples_series = pd.Series(triples_column).reset_index(drop=True)
    text_series = pd.Series(text_column).reset_index(drop=True)
    if len(triples_series) != len(text_series):
        raise ValueError(
            "triples and text columns must contain the same number of rows"
        )

    pairs: list[dict[str, Any]] = []
    result = np.full(len(triples_series), np.nan, dtype=float)

    for row_index, (triples, text) in enumerate(
        zip(triples_series, text_series)
    ):
        units = textual_units(text)
        serialised = [
            serialize_triple(triple)
            for triple in list(triples or [])
        ]
        if not units or not serialised:
            continue

        for unit_index, unit in enumerate(units):
            for triple in serialised:
                pairs.append({
                    "row_index": row_index,
                    "unit_index": unit_index,
                    "triple": triple,
                    "unit": unit,
                })

    if not pairs:
        return result

    expanded = pd.DataFrame(pairs)
    expanded["support"] = scorer.pairwise(
        expanded["triple"],
        expanded["unit"],
    )
    best_per_unit = (
        expanded.groupby(["row_index", "unit_index"])["support"]
        .max()
    )
    minima = best_per_unit.groupby("row_index").min()
    for row_index, value in minima.items():
        result[int(row_index)] = float(value)
    return result


def qa22_compute_metrics(
    frame: pd.DataFrame,
    scorer: SemanticScorer,
    *,
    source_triples_column: str,
    source_text_column: str,
    target_triples_column: str,
    target_text_column: str,
    target_language_column: str,
) -> pd.DataFrame:
    """Calculate all instance-level QA2.2 indicators.

    The returned columns implement C_min, G_min, L_{T->V}, and the row-level
    eligibility/success indicators E and Z used to calculate R_lit.
    """
    required = {
        source_triples_column,
        source_text_column,
        target_triples_column,
        target_text_column,
        target_language_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"QA2.2 input is missing required columns: {missing}"
        )

    data = frame.reset_index(drop=True)
    target_counts = data[target_triples_column].map(
        lambda values: len(list(values or []))
    )

    coverage = minimum_triple_coverage(
        data[target_triples_column],
        data[target_text_column],
        scorer,
    )
    groundedness = minimum_text_groundedness(
        data[target_triples_column],
        data[target_text_column],
        scorer,
    )

    literal_scores: list[float] = []
    literal_counts: list[int] = []
    english_eligible: list[bool] = []
    target_success: list[bool] = []

    for row in data.itertuples(index=False):
        source_triples = list(
            getattr(row, source_triples_column) or []
        )
        source_text = getattr(row, source_text_column)
        target_triples = list(
            getattr(row, target_triples_column) or []
        )
        target_text = getattr(row, target_text_column)
        target_language = str(
            getattr(row, target_language_column)
        )

        score, count = literal_retention(
            target_triples,
            target_text,
            target_language,
        )
        literal_scores.append(float(score) if pd.notna(score) else np.nan)
        literal_counts.append(int(count))

        source_atoms = literal_atoms_from_triples(
            source_triples,
            "en",
        )
        source_text_atoms = canonical_numeric_atoms(
            source_text,
            "en",
        )
        english_eligible.append(
            bool(source_atoms)
            and source_atoms.issubset(source_text_atoms)
        )

        target_atoms = literal_atoms_from_triples(
            target_triples,
            target_language,
        )
        target_text_atoms = canonical_numeric_atoms(
            target_text,
            target_language,
        )
        target_success.append(
            bool(target_atoms)
            and target_atoms.issubset(target_text_atoms)
        )

    return pd.DataFrame({
        "target_triple_count": target_counts.astype(int),
        "minimum_triple_coverage": coverage,
        "minimum_text_groundedness": groundedness,
        "textual_unit_count": data[target_text_column].map(
            lambda text: len(textual_units(text))
        ).astype(int),
        "target_text_literal_retention": literal_scores,
        "target_literal_count": literal_counts,
        "english_literal_eligible": pd.Series(
            english_eligible,
            dtype=bool,
        ),
        "target_literal_full_success": pd.Series(
            target_success,
            dtype=bool,
        ),
    })


def qa22_build_english_calibration(
    english_instances: pd.DataFrame,
    scorer: SemanticScorer,
    *,
    triples_column: str,
    text_column: str,
    percentile: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact-size English calibration thresholds for C_min and G_min.

    One row should represent one unique English gold RDF-to-text instance.
    Thresholds are produced only for triple-set sizes observed in that English
    calibration set. No global fallback is applied because the table defines
    tau_k from English instances containing exactly k triples.
    """
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must lie in [0, 1]")
    for column in [triples_column, text_column]:
        if column not in english_instances.columns:
            raise ValueError(
                f"English calibration is missing column {column!r}."
            )

    english = english_instances.copy().reset_index(drop=True)
    english["english_triple_count"] = english[triples_column].map(
        lambda values: len(list(values or []))
    ).astype(int)
    english["english_minimum_triple_coverage"] = (
        minimum_triple_coverage(
            english[triples_column],
            english[text_column],
            scorer,
        )
    )
    english["english_minimum_text_groundedness"] = (
        minimum_text_groundedness(
            english[triples_column],
            english[text_column],
            scorer,
        )
    )

    threshold_rows: list[dict[str, Any]] = []
    for triple_count, group in english.groupby(
        "english_triple_count",
        sort=True,
    ):
        coverage_values = group[
            "english_minimum_triple_coverage"
        ].dropna()
        groundedness_values = group[
            "english_minimum_text_groundedness"
        ].dropna()

        threshold_rows.append({
            "triple_count": int(triple_count),
            "coverage_threshold": (
                float(coverage_values.quantile(percentile))
                if len(coverage_values)
                else np.nan
            ),
            "groundedness_threshold": (
                float(groundedness_values.quantile(percentile))
                if len(groundedness_values)
                else np.nan
            ),
            "english_coverage_instances": int(len(coverage_values)),
            "english_groundedness_instances": int(
                len(groundedness_values)
            ),
            "calibration_percentile": float(percentile),
        })

    thresholds = pd.DataFrame(threshold_rows)
    return english, thresholds


def qa22_apply_english_calibration(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    triple_count_column: str = "target_triple_count",
    coverage_column: str = "minimum_triple_coverage",
    groundedness_column: str = "minimum_text_groundedness",
) -> pd.DataFrame:
    """Attach tau_k^C, tau_k^G, P_C, and P_G to target instances.

    A pass indicator is undefined when the target score is undefined or when
    no English threshold exists for the target's exact triple-set size.
    """
    required_metrics = {
        triple_count_column,
        coverage_column,
        groundedness_column,
    }
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise ValueError(
            f"QA2.2 metrics are missing columns: {missing_metrics}"
        )
    required_thresholds = {
        "triple_count",
        "coverage_threshold",
        "groundedness_threshold",
    }
    missing_thresholds = sorted(
        required_thresholds - set(thresholds.columns)
    )
    if missing_thresholds:
        raise ValueError(
            "QA2.2 thresholds are missing columns: "
            f"{missing_thresholds}"
        )

    output = metrics.copy().reset_index(drop=True)
    threshold_view = thresholds[[
        "triple_count",
        "coverage_threshold",
        "groundedness_threshold",
    ]].copy()
    output = output.merge(
        threshold_view,
        left_on=triple_count_column,
        right_on="triple_count",
        how="left",
        validate="many_to_one",
    )
    output = output.drop(columns=["triple_count"])

    coverage_pass = pd.Series(
        pd.NA,
        index=output.index,
        dtype="boolean",
    )
    coverage_valid = (
        output[coverage_column].notna()
        & output["coverage_threshold"].notna()
    )
    coverage_pass.loc[coverage_valid] = (
        output.loc[coverage_valid, coverage_column]
        >= output.loc[coverage_valid, "coverage_threshold"]
    )

    groundedness_pass = pd.Series(
        pd.NA,
        index=output.index,
        dtype="boolean",
    )
    groundedness_valid = (
        output[groundedness_column].notna()
        & output["groundedness_threshold"].notna()
    )
    groundedness_pass.loc[groundedness_valid] = (
        output.loc[groundedness_valid, groundedness_column]
        >= output.loc[
            groundedness_valid,
            "groundedness_threshold",
        ]
    )

    output["coverage_margin"] = (
        output[coverage_column]
        - output["coverage_threshold"]
    )
    output["groundedness_margin"] = (
        output[groundedness_column]
        - output["groundedness_threshold"]
    )
    output["english_calibrated_coverage_pass"] = coverage_pass
    output["english_calibrated_groundedness_pass"] = groundedness_pass
    return output


def qa22_relative_literal_retention_summary(
    metrics: pd.DataFrame,
    *,
    language_column: str = "language",
    language_label_column: str = "language_label",
    english_eligible_column: str = "english_literal_eligible",
    target_success_column: str = "target_literal_full_success",
    additional_group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Calculate R_lit for each target language and optional stratum."""
    group_columns = [
        *additional_group_columns,
        language_column,
        language_label_column,
    ]
    required = {
        *group_columns,
        english_eligible_column,
        target_success_column,
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(
            f"QA2.2 literal summary is missing columns: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))

        eligible = group[english_eligible_column].fillna(False).astype(bool)
        success = group[target_success_column].fillna(False).astype(bool)
        denominator = int(eligible.sum())
        numerator = int((eligible & success).sum())

        rows.append({
            **key_values,
            "audited_instances": int(len(group)),
            "english_eligible_instances": denominator,
            "target_successes_among_english_eligible": numerator,
            "relative_literal_retention": (
                float(numerator / denominator)
                if denominator > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def assert_qa22_metric_contract() -> None:
    """Verify the QA2.2 implementation against the table definitions."""

    class ContractScorer:
        backend = "sentence_transformer"

        @staticmethod
        def pairwise(left: pd.Series, right: pd.Series) -> np.ndarray:
            values: list[float] = []
            for left_value, right_value in zip(left, right):
                left_text = str(left_value)
                right_text = str(right_value)
                if right_text == "First fact. Second fact.":
                    values.append(0.9 if "Alpha" in left_text else 0.4)
                elif right_text == "First fact.":
                    values.append(0.8 if "Alpha" in left_text else 0.2)
                elif right_text == "Second fact.":
                    values.append(0.1 if "Alpha" in left_text else 0.7)
                else:
                    values.append(0.5)
            return np.asarray(values, dtype=float)

    scorer = ContractScorer()
    triples = pd.Series([[
        "Alpha | relation | 2020",
        "Beta | relation | 12",
    ]])
    text = pd.Series(["First fact. Second fact."])

    coverage = minimum_triple_coverage(
        triples,
        text,
        scorer,
    )
    if not math.isclose(float(coverage[0]), 0.4):
        raise AssertionError(
            "C_min must retain the least-supported triple score."
        )

    groundedness = minimum_text_groundedness(
        triples,
        text,
        scorer,
    )
    if not math.isclose(float(groundedness[0]), 0.7):
        raise AssertionError(
            "G_min must retain the least-supported best-matched unit."
        )

    literal_score, literal_count = literal_retention(
        ["Alpha | relation 2020 | value"],
        "The relation occurred in 2020.",
        "en",
    )
    if literal_count != 1 or not math.isclose(literal_score, 1.0):
        raise AssertionError(
            "L_T->V must inspect all triple components and retain 2020."
        )

    english = pd.DataFrame({
        "source_triples": [[
            "Alpha | relation | 2020",
            "Beta | relation | 12",
        ]],
        "source_text": ["First fact. Second fact."],
    })
    _, thresholds = qa22_build_english_calibration(
        english,
        scorer,
        triples_column="source_triples",
        text_column="source_text",
        percentile=0.05,
    )
    if thresholds["triple_count"].tolist() != [2]:
        raise AssertionError(
            "English calibration must be stratified by exact triple count."
        )

    target = pd.DataFrame({
        "target_triple_count": [2, 3],
        "minimum_triple_coverage": [0.4, 0.5],
        "minimum_text_groundedness": [0.7, 0.5],
    })
    calibrated = qa22_apply_english_calibration(
        target,
        thresholds,
    )
    if not bool(calibrated.loc[0, "english_calibrated_coverage_pass"]):
        raise AssertionError("P_C must pass at its exact threshold.")
    if pd.notna(calibrated.loc[1, "english_calibrated_coverage_pass"]):
        raise AssertionError(
            "An unavailable exact-size threshold must not use a global fallback."
        )

    relative_input = pd.DataFrame({
        "language": ["es", "es", "es"],
        "language_label": ["Spanish"] * 3,
        "english_literal_eligible": [True, True, False],
        "target_literal_full_success": [True, False, True],
    })
    relative = qa22_relative_literal_retention_summary(
        relative_input
    )
    if not math.isclose(
        float(relative.loc[0, "relative_literal_retention"]),
        0.5,
    ):
        raise AssertionError(
            "R_lit must divide target successes by English-eligible instances."
        )

def aligned_reference_advantage(
    target_text: pd.Series,
    aligned_source_text: pd.Series,
    comparison_source_text: pd.Series,
    scorer: SemanticScorer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare an alternative target with aligned and comparison sources."""
    target = pd.Series(target_text, dtype="string").fillna("").astype(str)
    aligned = (
        pd.Series(aligned_source_text, dtype="string")
        .fillna("").astype(str)
    )
    comparison = (
        pd.Series(comparison_source_text, dtype="string")
        .fillna("").astype(str)
    )
    aligned_similarity = scorer.pairwise(aligned, target)
    comparison_similarity = scorer.pairwise(comparison, target)
    valid = (
        target.map(normalize_space).ne("")
        & aligned.map(normalize_space).ne("")
        & comparison.map(normalize_space).ne("")
    ).to_numpy()
    advantage = aligned_similarity - comparison_similarity
    aligned_similarity = aligned_similarity.astype(float)
    comparison_similarity = comparison_similarity.astype(float)
    advantage = advantage.astype(float)
    aligned_similarity[~valid] = np.nan
    comparison_similarity[~valid] = np.nan
    advantage[~valid] = np.nan
    return aligned_similarity, comparison_similarity, advantage


def cluster_bootstrap_mean_ci(
    group: pd.DataFrame,
    value_column: str,
    cluster_column: str = "record_key",
    replicates: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    usable = group[[cluster_column, value_column]].dropna()
    clusters = usable[cluster_column].unique()
    if len(clusters) < 2 or replicates <= 0:
        return np.nan, np.nan
    grouped = {
        cluster: usable.loc[
            usable[cluster_column] == cluster,
            value_column,
        ].to_numpy()
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )
        values.append(
            float(
                np.concatenate(
                    [grouped[item] for item in sampled]
                ).mean()
            )
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


PLACEHOLDER_RE = re.compile(
    r"^(?:none|null|nan|n/?a|unknown|undefined)$",
    re.IGNORECASE,
)
SOURCE_MARKUP_RE = re.compile(r"(?:@[Ee][Nn]\b|\^\^)")
VALID_URI_RE = re.compile(
    r"^(?:<https?://[^>]+>|https?://\S+)$",
    re.IGNORECASE,
)
CAMEL_CASE_RE = re.compile(r"[a-zà-ÿ][A-Z]")


def malformed_component(value: str) -> bool:
    text = normalize_space(value)
    if VALID_URI_RE.fullmatch(text):
        return False
    return (
        not text
        or bool(PLACEHOLDER_RE.fullmatch(text))
        or bool(SOURCE_MARKUP_RE.search(text))
    )


def triple_well_formed(triples: list[str]) -> bool:
    if not triples:
        return False
    for triple in triples:
        if any(
            malformed_component(component)
            for component in split_triple(triple)
        ):
            return False
    return True


def source_identical_predicate(
    source_triples: list[str],
    target_triples: list[str],
) -> bool:
    return any(
        bool(normalize_label(split_triple(source)[1]))
        and normalize_label(split_triple(source)[1])
        == normalize_label(split_triple(target)[1])
        for source, target in zip(source_triples, target_triples)
    )


def structured_untranslated_predicate(
    source_triples: list[str],
    target_triples: list[str],
) -> bool:
    """Return the QA2.1 structured-predicate review flag U.

    This backward-compatible scalar helper follows the same alignment rule as
    ``predicate_component_diagnostics``. Unequal or empty triple sets are not
    treated as aligned predicate evidence and therefore return ``False``.
    """
    if (
        not source_triples
        or len(source_triples) != len(target_triples)
    ):
        return False

    for source_triple, target_triple in zip(
        source_triples,
        target_triples,
    ):
        source_predicate = split_triple(source_triple)[1]
        target_predicate = split_triple(target_triple)[1]
        source_normalized = normalize_label(source_predicate)
        target_normalized = normalize_label(target_predicate)
        if not source_normalized or not target_normalized:
            return False
        if (
            source_normalized == target_normalized
            and bool(CAMEL_CASE_RE.search(source_predicate))
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# QA2.1 — source-to-target knowledge preservation
# ---------------------------------------------------------------------------
#
# The reporting unit is one aligned source-target RDF-to-text instance. In
# WebNLG this is identified by an RDF entry, lexicalisation ID, and target
# language. The table uses e as shorthand for this aligned instance.
#
# The shared implementation returns:
#   TextSim_l(e)       cross-lingual verbalisation similarity;
#   TripleSim_l(e)     complete triple-set similarity;
#   PredSim_l(e)       mean similarity across all aligned predicate pairs;
#   U_l(e)             structured unchanged-predicate review flag; and
#   L_{S->T,l}(e)      source-to-target numeric/date atom preservation.
# ---------------------------------------------------------------------------


def _require_qa21_sentence_embeddings(scorer: SemanticScorer) -> None:
    """Require the multilingual sentence-embedding backend in the table."""
    if getattr(scorer, "backend", None) != "sentence_transformer":
        raise ValueError(
            "QA2.1 reportable metrics require the sentence_transformer "
            "backend with L2-normalised multilingual embeddings."
        )


def qa21_pairwise_similarity(
    left: Sequence[Any],
    right: Sequence[Any],
    scorer: SemanticScorer,
) -> np.ndarray:
    """Calculate cosine similarity for non-empty aligned text pairs."""
    _require_qa21_sentence_embeddings(scorer)
    left_series = pd.Series(left).reset_index(drop=True).map(normalize_space)
    right_series = pd.Series(right).reset_index(drop=True).map(normalize_space)
    if len(left_series) != len(right_series):
        raise ValueError("left and right must contain the same number of rows")

    result = np.full(len(left_series), np.nan, dtype=float)
    valid = left_series.ne("") & right_series.ne("")
    if valid.any():
        result[valid.to_numpy()] = scorer.pairwise(
            left_series.loc[valid].reset_index(drop=True),
            right_series.loc[valid].reset_index(drop=True),
        )
    return result


def qa21_compute_metrics(
    frame: pd.DataFrame,
    scorer: SemanticScorer,
    *,
    source_text_column: str,
    target_text_column: str,
    source_triples_column: str,
    target_triples_column: str,
    target_language_column: str,
) -> pd.DataFrame:
    """Calculate all QA2.1 indicators through one shared implementation.

    ``frame`` must contain one aligned RDF-to-text instance per row. Predicate
    diagnostics are defined only for complete one-to-one triple alignment;
    other semantic indicators remain available when their own inputs exist.
    """
    _require_qa21_sentence_embeddings(scorer)

    required = {
        source_text_column,
        target_text_column,
        source_triples_column,
        target_triples_column,
        target_language_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"QA2.1 input is missing required columns: {missing}"
        )

    working = frame.reset_index(drop=True)
    source_triples = working[source_triples_column].map(
        lambda value: list(value or [])
    )
    target_triples = working[target_triples_column].map(
        lambda value: list(value or [])
    )

    source_serialized = source_triples.map(serialize_triple_set)
    target_serialized = target_triples.map(serialize_triple_set)

    output = pd.DataFrame(index=working.index)

    # TextSim_l(e)
    output["text_similarity"] = qa21_pairwise_similarity(
        working[source_text_column],
        working[target_text_column],
        scorer,
    )

    # TripleSim_l(e)
    output["triple_similarity"] = qa21_pairwise_similarity(
        source_serialized,
        target_serialized,
        scorer,
    )

    # PredSim_l(e), U_l(e), n_e, and the positional-alignment diagnostic.
    predicate = predicate_component_diagnostics(
        source_triples,
        target_triples,
        scorer,
    )
    output = pd.concat([output, predicate], axis=1)

    # L_{S->T,l}(e). The score is NaN when A(T_en) is empty.
    literal_results = [
        source_target_literal_preservation(
            source_values,
            target_values,
            str(language),
        )
        for source_values, target_values, language in zip(
            source_triples,
            target_triples,
            working[target_language_column],
        )
    ]
    output["source_target_literal_preservation"] = [
        value for value, _ in literal_results
    ]
    output["source_literal_count"] = [
        count for _, count in literal_results
    ]

    return output


def assert_qa21_metric_contract() -> None:
    """Fail early if QA2.1 drifts from Table QA2.1."""

    class _ContractScorer:
        backend = "sentence_transformer"

        @staticmethod
        def pairwise(left: pd.Series, right: pd.Series) -> np.ndarray:
            values = []
            for left_value, right_value in zip(left, right):
                pair = (str(left_value), str(right_value))
                if pair == ("English text", "Texto español"):
                    values.append(0.8)
                elif "[S]" in pair[0] and "[S]" in pair[1]:
                    values.append(0.7)
                elif pair[0] == pair[1]:
                    values.append(1.0)
                else:
                    values.append(0.5)
            return np.asarray(values, dtype=float)

    scorer = _ContractScorer()
    frame = pd.DataFrame({
        "source_text": ["English text"],
        "target_text": ["Texto español"],
        "source_triples": [[
            "A | birthPlace | 2012",
            "A | leaderName | Bob",
        ]],
        "target_triples": [[
            "A | birthPlace | 2012",
            "A | nombre del líder | Bob",
        ]],
        "language": ["es"],
    })
    result = qa21_compute_metrics(
        frame,
        scorer,
        source_text_column="source_text",
        target_text_column="target_text",
        source_triples_column="source_triples",
        target_triples_column="target_triples",
        target_language_column="language",
    )

    if not math.isclose(float(result.loc[0, "text_similarity"]), 0.8):
        raise AssertionError("TextSim does not use aligned text embeddings.")
    if not math.isclose(float(result.loc[0, "triple_similarity"]), 0.7):
        raise AssertionError("TripleSim does not use complete serialisations.")
    if not math.isclose(float(result.loc[0, "predicate_similarity"]), 0.75):
        raise AssertionError("PredSim is not the mean over all predicate pairs.")
    if float(result.loc[0, "structured_predicate_review_flag"]) != 1.0:
        raise AssertionError("Structured unchanged predicates must trigger U.")
    if int(result.loc[0, "predicate_pair_count"]) != 2:
        raise AssertionError("The predicate denominator n_e is incorrect.")
    if not math.isclose(
        float(result.loc[0, "source_target_literal_preservation"]),
        1.0,
    ):
        raise AssertionError("Source-to-target literal preservation is wrong.")

    # A(T) is defined over the complete triple set, including predicates.
    predicate_atoms = literal_atoms_from_triples(
        ["A | route 66 | B"],
        "en",
    )
    if predicate_atoms != {"66"}:
        raise AssertionError("A(T) must inspect all triple components.")

    mismatched = predicate_component_diagnostics(
        pd.Series([["A | p | B", "A | q | C"]]),
        pd.Series([["A | p | B"]]),
        scorer,
    )
    if not np.isnan(float(mismatched.loc[0, "predicate_similarity"])):
        raise AssertionError("PredSim must be undefined for unequal triple counts.")


def parse_original_language_data(
    repo_root: Path,
    max_per_language: int,
) -> pd.DataFrame:
    data_root = repo_root / "WebNLG_CA_BT"
    if not data_root.exists():
        raise FileNotFoundError(
            f"Original corpus not found at {data_root}"
        )

    rows = []
    for path in sorted(data_root.rglob("*.xml")):
        relative = path.relative_to(data_root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.name.endswith("-checkpoint.xml"):
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        for entry in root.findall(".//entry"):
            eid = (entry.get("eid") or "").strip()
            record_key = f"{relative.as_posix()}::{eid}"
            lexicalisations = defaultdict(dict)
            for node in entry.findall("lex"):
                language = (node.get("lang") or "").strip()
                lid = (node.get("lid") or "").strip()
                text = normalize_space(node.text)
                if language and lid and text:
                    lexicalisations[language][lid] = text

            for language in ("en", "es", "ca"):
                for _, text in lexicalisations.get(language, {}).items():
                    rows.append({
                        "record_key": record_key,
                        "language": language,
                        "text": text,
                    })

    frame = pd.DataFrame(rows).drop_duplicates()
    sampled = []
    for language, group in frame.groupby("language"):
        n = min(max_per_language, len(group))
        sampled.append(
            group.sample(n=n, random_state=RANDOM_SEED)
        )
    return pd.concat(sampled, ignore_index=True)


def train_language_detector(
    repo_root: Path,
    cache_dir: Path,
    max_per_language: int,
    force: bool = False,
):
    model_path = (
        cache_dir
        / f"qa3_language_detector_{max_per_language}.joblib"
    )
    metrics_path = (
        cache_dir
        / f"qa3_language_detector_{max_per_language}.json"
    )
    if (
        joblib is not None
        and model_path.exists()
        and metrics_path.exists()
        and not force
    ):
        model = joblib.load(model_path)
        metadata = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
        return model, metadata

    data = parse_original_language_data(
        repo_root,
        max_per_language,
    )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train_index, test_index = next(
        splitter.split(
            data["text"],
            data["language"],
            groups=data["record_key"],
        )
    )
    train = data.iloc[train_index]
    test = data.iloc[test_index]

    model = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=60_000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            SGDClassifier(
                loss="log_loss",
                max_iter=50,
                tol=1e-3,
                class_weight="balanced",
                average=True,
                random_state=RANDOM_SEED,
            ),
        ),
    ])
    model.fit(train["text"], train["language"])
    prediction = model.predict(test["text"])

    metadata = {
        "grouped_holdout_accuracy": float(
            accuracy_score(test["language"], prediction)
        ),
        "training_examples": int(len(train)),
        "holdout_examples": int(len(test)),
        "max_per_language": int(max_per_language),
        "entity_masking": False,
        "code_switch_definition": (
            "English probability >= 0.70 in at least 25% "
            "of 10-token/8-stride windows"
        ),
    }

    if joblib is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metrics_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    return model, metadata


def predict_language(model, texts: pd.Series):
    texts = pd.Series(texts, dtype="string").fillna("").astype(str)
    probabilities = model.predict_proba(texts)
    classes = model.named_steps["classifier"].classes_
    index = probabilities.argmax(axis=1)
    predicted = classes[index]
    confidence = probabilities.max(axis=1)
    positions = {
        label: position
        for position, label in enumerate(classes)
    }
    return predicted, confidence, probabilities, positions



# ---------------------------------------------------------------------------
# QA3.1 — target-language validity and source-language leakage
# ---------------------------------------------------------------------------
#
# Table~\ref{tab:audit-framework-qa31} defines:
#
#   P_ell(e): probability assigned to the expected target language;
#   V_ell(e): valid-language indicator;
#   W_ell(e): wrong-language flag;
#   C_ell(e): exact English-copy flag; and
#   CS_ell(e): English code-switch flag.
#
# The code-switch implementation uses overlapping 10-token windows with a
# stride of 8 and retains windows containing at least 6 tokens. A target text
# with no eligible window cannot satisfy the code-switch condition.

QA31_LANGUAGE_CONFIDENCE = 0.70
QA31_CODE_SWITCH_ENGLISH_PROBABILITY = 0.70
QA31_CODE_SWITCH_WINDOW_RATE = 0.25
QA31_WINDOW_SIZE = 10
QA31_WINDOW_STRIDE = 8
QA31_MINIMUM_WINDOW_TOKENS = 6


def qa31_eligible_windows(
    text: object,
    window_size: int = QA31_WINDOW_SIZE,
    stride: int = QA31_WINDOW_STRIDE,
    minimum_tokens: int = QA31_MINIMUM_WINDOW_TOKENS,
) -> list[str]:
    """Return the eligible overlapping windows in W(v).

    The final partial window is retained whenever it contains at least
    ``minimum_tokens`` tokens. This means that a six- or seven-token text has
    one eligible window, which is required by the stated window definition.
    """

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if minimum_tokens < 1 or minimum_tokens > window_size:
        raise ValueError(
            "minimum_tokens must lie between 1 and window_size."
        )

    tokens = normalize_space(text).split()
    return [
        " ".join(tokens[start:start + window_size])
        for start in range(0, len(tokens), stride)
        if len(tokens[start:start + window_size]) >= minimum_tokens
    ]


def qa31_code_switch_profile(
    model,
    text: object,
    english_probability_threshold: float = (
        QA31_CODE_SWITCH_ENGLISH_PROBABILITY
    ),
    window_rate_threshold: float = QA31_CODE_SWITCH_WINDOW_RATE,
    window_size: int = QA31_WINDOW_SIZE,
    stride: int = QA31_WINDOW_STRIDE,
    minimum_tokens: int = QA31_MINIMUM_WINDOW_TOKENS,
) -> dict[str, object]:
    """Calculate the window evidence used by CS_ell(e)."""

    if not 0.0 <= english_probability_threshold <= 1.0:
        raise ValueError(
            "english_probability_threshold must lie in [0, 1]."
        )
    if not 0.0 <= window_rate_threshold <= 1.0:
        raise ValueError("window_rate_threshold must lie in [0, 1].")

    windows = qa31_eligible_windows(
        text,
        window_size=window_size,
        stride=stride,
        minimum_tokens=minimum_tokens,
    )
    if not windows:
        return {
            "eligible_window_count": 0,
            "english_window_count": 0,
            "english_window_fraction": np.nan,
            "code_switch_detected": False,
        }

    probabilities = np.asarray(
        model.predict_proba(windows),
        dtype=float,
    )
    classes = list(
        model.named_steps["classifier"].classes_
    )
    if "en" not in classes:
        raise ValueError(
            "The language-identification model has no English class."
        )
    english_index = classes.index("en")
    english_windows = (
        probabilities[:, english_index]
        >= english_probability_threshold
    )
    english_count = int(english_windows.sum())
    fraction = float(english_windows.mean())

    return {
        "eligible_window_count": int(len(windows)),
        "english_window_count": english_count,
        "english_window_fraction": fraction,
        "code_switch_detected": bool(
            fraction >= window_rate_threshold
        ),
    }


def code_switch_flag(
    model,
    text: str,
    expected_language: str,
) -> bool:
    """Backward-compatible wrapper for CS_ell(e).

    ``expected_language`` is retained because earlier runners supplied it,
    although the table defines the flag solely from English probabilities in
    the target text windows.
    """

    _ = expected_language
    return bool(
        qa31_code_switch_profile(
            model,
            text,
        )["code_switch_detected"]
    )


def qa31_compute_metrics(
    target_texts: Sequence[object],
    source_texts: Sequence[object],
    expected_languages: Sequence[object],
    model,
    language_confidence_threshold: float = QA31_LANGUAGE_CONFIDENCE,
    english_window_probability_threshold: float = (
        QA31_CODE_SWITCH_ENGLISH_PROBABILITY
    ),
    code_switch_window_rate_threshold: float = (
        QA31_CODE_SWITCH_WINDOW_RATE
    ),
    window_size: int = QA31_WINDOW_SIZE,
    window_stride: int = QA31_WINDOW_STRIDE,
    minimum_window_tokens: int = QA31_MINIMUM_WINDOW_TOKENS,
) -> pd.DataFrame:
    """Calculate all QA3.1 indicators for aligned RDF-to-text instances."""

    target = pd.Series(target_texts, dtype="string").fillna("").astype(str)
    source = pd.Series(source_texts, dtype="string").fillna("").astype(str)
    expected = (
        pd.Series(expected_languages, dtype="string")
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if not (len(target) == len(source) == len(expected)):
        raise ValueError(
            "Target texts, source texts, and expected languages must "
            "have equal length."
        )
    if not 0.0 <= language_confidence_threshold <= 1.0:
        raise ValueError(
            "language_confidence_threshold must lie in [0, 1]."
        )

    predicted, confidence, probabilities, positions = predict_language(
        model,
        target,
    )
    missing_languages = sorted(
        set(expected) - set(positions)
    )
    if missing_languages:
        raise ValueError(
            "Expected target languages are absent from the detector: "
            f"{missing_languages}"
        )

    target_probability = np.asarray([
        probabilities[index, positions[language]]
        for index, language in enumerate(expected)
    ], dtype=float)

    predicted_series = pd.Series(predicted, dtype="string")
    confidence_series = pd.Series(confidence, dtype=float)
    expected_series = expected.reset_index(drop=True)

    valid_language = (
        predicted_series.eq(expected_series)
        & pd.Series(target_probability).ge(
            language_confidence_threshold
        )
    )
    wrong_language = (
        predicted_series.ne(expected_series)
        & confidence_series.ge(language_confidence_threshold)
    )
    exact_copy = (
        target.map(normalize_space)
        == source.map(normalize_space)
    )

    code_switch_profiles = [
        qa31_code_switch_profile(
            model,
            text,
            english_probability_threshold=(
                english_window_probability_threshold
            ),
            window_rate_threshold=(
                code_switch_window_rate_threshold
            ),
            window_size=window_size,
            stride=window_stride,
            minimum_tokens=minimum_window_tokens,
        )
        for text in target
    ]

    return pd.DataFrame({
        "predicted_language": predicted_series,
        "language_confidence": confidence_series,
        "target_language_probability": target_probability,
        "valid_target_language": valid_language.astype(bool),
        "wrong_language_detected": wrong_language.astype(bool),
        "full_english_copy_detected": exact_copy.astype(bool),
        "code_switch_eligible_window_count": [
            profile["eligible_window_count"]
            for profile in code_switch_profiles
        ],
        "code_switch_english_window_count": [
            profile["english_window_count"]
            for profile in code_switch_profiles
        ],
        "code_switch_english_window_fraction": [
            profile["english_window_fraction"]
            for profile in code_switch_profiles
        ],
        "code_switch_detected": [
            profile["code_switch_detected"]
            for profile in code_switch_profiles
        ],
    })


def assert_qa31_metric_contract() -> None:
    """Verify that the QA3.1 implementation matches its table."""

    class _Classifier:
        classes_ = np.array(["ca", "en", "es"])

    class _Model:
        named_steps = {"classifier": _Classifier()}

        def predict_proba(self, texts):
            rows = []
            for value in texts:
                text = normalize_space(value)
                if text == "expected-es":
                    rows.append([0.10, 0.10, 0.80])
                elif text == "uncertain-en":
                    rows.append([0.20, 0.60, 0.20])
                else:
                    rows.append([0.10, 0.80, 0.10])
            return np.asarray(rows, dtype=float)

    model = _Model()
    target = [
        "expected-es",
        "wrong-en",
        "uncertain-en",
        "english english english english english english",
    ]
    source = [
        "  expected-es  ",
        "different source",
        "different source",
        "different source",
    ]
    expected = ["es", "es", "es", "es"]

    result = qa31_compute_metrics(
        target,
        source,
        expected,
        model,
    )

    if not math.isclose(
        float(result.loc[0, "target_language_probability"]),
        0.80,
    ):
        raise AssertionError("P_ell(e) is not the expected-language probability.")
    if not bool(result.loc[0, "valid_target_language"]):
        raise AssertionError("A confident expected-language prediction must pass V.")
    if bool(result.loc[0, "wrong_language_detected"]):
        raise AssertionError("A valid target-language prediction must not trigger W.")
    if not bool(result.loc[0, "full_english_copy_detected"]):
        raise AssertionError("C must use whitespace-normalised exact equality.")

    if not bool(result.loc[1, "wrong_language_detected"]):
        raise AssertionError("A confident alternative-language prediction must trigger W.")
    if bool(result.loc[2, "wrong_language_detected"]):
        raise AssertionError("An alternative prediction below 0.70 must not trigger W.")

    if int(result.loc[3, "code_switch_eligible_window_count"]) != 1:
        raise AssertionError("A six-token text must form one eligible window.")
    if not bool(result.loc[3, "code_switch_detected"]):
        raise AssertionError("A confidently English eligible window must trigger CS.")

    no_windows = qa31_code_switch_profile(model, "five token text only here")
    if int(no_windows["eligible_window_count"]) != 0:
        raise AssertionError("Texts shorter than six tokens must have no windows.")
    if bool(no_windows["code_switch_detected"]):
        raise AssertionError("CS must be false when no eligible windows exist.")


def legacy_internal_chrf(
    reference: str,
    hypothesis: str,
    max_order: int = 6,
) -> float:
    reference = normalize_space(reference)
    hypothesis = normalize_space(hypothesis)
    if not reference or not hypothesis:
        return 0.0

    scores = []
    for order in range(1, max_order + 1):
        reference_ngrams = defaultdict(int)
        hypothesis_ngrams = defaultdict(int)
        for index in range(
            max(0, len(reference) - order + 1)
        ):
            reference_ngrams[
                reference[index:index + order]
            ] += 1
        for index in range(
            max(0, len(hypothesis) - order + 1)
        ):
            hypothesis_ngrams[
                hypothesis[index:index + order]
            ] += 1

        overlap = sum(
            min(count, hypothesis_ngrams[gram])
            for gram, count in reference_ngrams.items()
        )
        precision = overlap / max(
            1,
            sum(hypothesis_ngrams.values()),
        )
        recall = overlap / max(
            1,
            sum(reference_ngrams.values()),
        )
        scores.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (
                precision + recall
            )
        )
    return float(np.mean(scores))


# Backward-compatible name only. This measure is not part of the v2 core audit.
internal_chrf = legacy_internal_chrf

# ---------------------------------------------------------------------------
# QA3.2 — target-language variation and aligned-reference advantage
# ---------------------------------------------------------------------------
#
# Table~\ref{tab:audit-framework-qa32} defines:
#
#   ER_ell(e): target-token count divided by the aligned English-token count;
#   Delta_align,ell(e,a,b): similarity to the aligned English lexicalisation
#   minus similarity to another English lexicalisation from the same entry.
#
# Tokenisation is the whitespace tokenisation obtained after
# ``normalize_space``. Aligned-reference advantage is defined only when all
# three texts are non-empty, the target and aligned English references share
# lexicalisation identifier a, the comparison reference has a different
# identifier b, and all references belong to the same RDF entry.


def qa32_tokens(text: object) -> list[str]:
    """Return tok(v): whitespace tokens after superficial normalisation."""

    normalized = normalize_space(text)
    return normalized.split() if normalized else []


def qa32_select_comparison_references(
    frame: pd.DataFrame,
    record_column: str = "record_key",
    source_lid_column: str = "source_lid",
    source_text_column: str = "source_text",
) -> pd.DataFrame:
    """Select one deterministic different English reference per row.

    Candidate English lexicalisations are restricted to the same RDF entry,
    must have a non-empty identifier and text, and must have an identifier
    different from the aligned identifier. Candidates are sorted by
    lexicalisation identifier and text; the first eligible candidate is used.
    """

    required = {
        record_column,
        source_lid_column,
        source_text_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "QA3.2 comparison-reference selection is missing columns: "
            f"{sorted(missing)}"
        )

    references = frame[
        [record_column, source_lid_column, source_text_column]
    ].copy()
    references[record_column] = (
        references[record_column].fillna("").astype(str)
    )
    references[source_lid_column] = (
        references[source_lid_column].fillna("").astype(str)
    )
    references[source_text_column] = (
        references[source_text_column].fillna("").astype(str)
    )
    references = references[
        references[record_column].map(normalize_space).ne("")
        & references[source_lid_column].map(normalize_space).ne("")
        & references[source_text_column].map(normalize_space).ne("")
    ].drop_duplicates()
    references = references.sort_values(
        [record_column, source_lid_column, source_text_column],
        kind="mergesort",
    )

    by_record: dict[str, list[tuple[str, str]]] = {
        str(record): list(
            group[
                [source_lid_column, source_text_column]
            ].itertuples(index=False, name=None)
        )
        for record, group in references.groupby(
            record_column,
            sort=False,
        )
    }

    comparison_lids: list[str] = []
    comparison_texts: list[str] = []
    for row in frame[
        [record_column, source_lid_column]
    ].itertuples(index=False, name=None):
        record, aligned_lid = (
            "" if pd.isna(row[0]) else str(row[0]),
            "" if pd.isna(row[1]) else str(row[1]),
        )
        candidates = [
            (lid, text)
            for lid, text in by_record.get(record, [])
            if lid != aligned_lid
        ]
        if candidates:
            comparison_lid, comparison_text = candidates[0]
        else:
            comparison_lid, comparison_text = "", ""
        comparison_lids.append(comparison_lid)
        comparison_texts.append(comparison_text)

    return pd.DataFrame(
        {
            "comparison_source_lid": comparison_lids,
            "comparison_source_text": comparison_texts,
        },
        index=frame.index,
    )


def qa32_compute_metrics(
    aligned_source_texts: Sequence[object],
    target_texts: Sequence[object],
    comparison_source_texts: Sequence[object],
    scorer: SemanticScorer,
    *,
    target_lids: Sequence[object] | None = None,
    aligned_source_lids: Sequence[object] | None = None,
    comparison_source_lids: Sequence[object] | None = None,
    target_record_keys: Sequence[object] | None = None,
    aligned_source_record_keys: Sequence[object] | None = None,
    comparison_source_record_keys: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Calculate the QA3.2 indicators for aligned lexicalisation instances.

    ``aligned_source_texts`` is both the denominator reference for ER and the
    aligned English reference in Delta_align. The advantage is left undefined
    when the table's text, identifier, or same-entry conditions are not met.
    """

    aligned = (
        pd.Series(aligned_source_texts, dtype="string")
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )
    target = (
        pd.Series(target_texts, dtype="string")
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )
    comparison = (
        pd.Series(comparison_source_texts, dtype="string")
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )

    if not (len(aligned) == len(target) == len(comparison)):
        raise ValueError(
            "QA3.2 aligned, target, and comparison text collections "
            "must have equal length."
        )

    english_token_count = aligned.map(lambda value: len(qa32_tokens(value)))
    target_token_count = target.map(lambda value: len(qa32_tokens(value)))
    expansion_ratio = (
        target_token_count.astype(float)
        / english_token_count.clip(lower=1).astype(float)
    )

    eligible = (
        aligned.map(normalize_space).ne("")
        & target.map(normalize_space).ne("")
        & comparison.map(normalize_space).ne("")
    )

    identifier_arguments = (
        target_lids,
        aligned_source_lids,
        comparison_source_lids,
    )
    if any(value is not None for value in identifier_arguments):
        if not all(value is not None for value in identifier_arguments):
            raise ValueError(
                "Target, aligned-source, and comparison-source LIDs "
                "must be supplied together."
            )
        target_lid = (
            pd.Series(target_lids, dtype="string")
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )
        aligned_lid = (
            pd.Series(aligned_source_lids, dtype="string")
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )
        comparison_lid = (
            pd.Series(comparison_source_lids, dtype="string")
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )
        if not (
            len(target_lid)
            == len(aligned_lid)
            == len(comparison_lid)
            == len(target)
        ):
            raise ValueError(
                "QA3.2 lexicalisation-ID collections must match the "
                "text collection length."
            )
        eligible &= (
            target_lid.map(normalize_space).ne("")
            & aligned_lid.map(normalize_space).ne("")
            & comparison_lid.map(normalize_space).ne("")
            & target_lid.eq(aligned_lid)
            & aligned_lid.ne(comparison_lid)
        )

    record_arguments = (
        target_record_keys,
        aligned_source_record_keys,
        comparison_source_record_keys,
    )
    if any(value is not None for value in record_arguments):
        if not all(value is not None for value in record_arguments):
            raise ValueError(
                "Target, aligned-source, and comparison-source record "
                "keys must be supplied together."
            )
        target_record = (
            pd.Series(target_record_keys, dtype="string")
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )
        aligned_record = (
            pd.Series(aligned_source_record_keys, dtype="string")
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )
        comparison_record = (
            pd.Series(comparison_source_record_keys, dtype="string")
            .fillna("")
            .astype(str)
            .reset_index(drop=True)
        )
        if not (
            len(target_record)
            == len(aligned_record)
            == len(comparison_record)
            == len(target)
        ):
            raise ValueError(
                "QA3.2 record-key collections must match the text "
                "collection length."
            )
        eligible &= (
            target_record.map(normalize_space).ne("")
            & target_record.eq(aligned_record)
            & target_record.eq(comparison_record)
        )

    aligned_similarity = np.full(len(target), np.nan, dtype=float)
    comparison_similarity = np.full(len(target), np.nan, dtype=float)
    advantage = np.full(len(target), np.nan, dtype=float)

    eligible_array = eligible.to_numpy(dtype=bool)
    if eligible_array.any():
        eligible_index = np.flatnonzero(eligible_array)
        aligned_scores = scorer.pairwise(
            aligned.iloc[eligible_index],
            target.iloc[eligible_index],
        )
        comparison_scores = scorer.pairwise(
            comparison.iloc[eligible_index],
            target.iloc[eligible_index],
        )
        aligned_similarity[eligible_index] = aligned_scores
        comparison_similarity[eligible_index] = comparison_scores
        advantage[eligible_index] = (
            aligned_scores - comparison_scores
        )

    return pd.DataFrame(
        {
            "english_token_count": english_token_count.astype(int),
            "target_token_count": target_token_count.astype(int),
            "expansion_ratio": expansion_ratio.astype(float),
            "aligned_reference_eligible": eligible.astype(bool),
            "aligned_reference_similarity": aligned_similarity,
            "comparison_reference_similarity": comparison_similarity,
            "aligned_reference_advantage": advantage,
        }
    )


def assert_qa32_metric_contract() -> None:
    """Verify that the QA3.2 implementation matches its table."""

    class _ContractScorer:
        backend = "sentence_transformer"

        def pairwise(self, left, right):
            values = []
            for source, target in zip(left, right):
                source_text = normalize_space(source)
                target_text = normalize_space(target)
                if target_text == "texto de destino":
                    values.append(
                        0.85
                        if source_text == "aligned english"
                        else 0.35
                    )
                else:
                    values.append(0.50)
            return np.asarray(values, dtype=float)

    scorer = _ContractScorer()
    result = qa32_compute_metrics(
        aligned_source_texts=[
            "one two",
            "",
            "aligned english",
            "aligned english",
        ],
        target_texts=[
            "uno dos tres",
            "dos tokens",
            "texto de destino",
            "texto de destino",
        ],
        comparison_source_texts=[
            "",
            "",
            "comparison english",
            "comparison english",
        ],
        scorer=scorer,
        target_lids=["a", "a", "a", "a"],
        aligned_source_lids=["a", "a", "a", "a"],
        comparison_source_lids=["", "", "b", "a"],
        target_record_keys=["e1", "e2", "e3", "e4"],
        aligned_source_record_keys=["e1", "e2", "e3", "e4"],
        comparison_source_record_keys=["e1", "e2", "e3", "e4"],
    )

    if not math.isclose(float(result.loc[0, "expansion_ratio"]), 1.5):
        raise AssertionError(
            "ER must divide target tokens by aligned English tokens."
        )
    if not math.isclose(float(result.loc[1, "expansion_ratio"]), 2.0):
        raise AssertionError(
            "ER must use one when the English token count is zero."
        )
    if not bool(result.loc[2, "aligned_reference_eligible"]):
        raise AssertionError(
            "A non-empty same-entry a-versus-b comparison must be eligible."
        )
    if not math.isclose(
        float(result.loc[2, "aligned_reference_advantage"]),
        0.50,
    ):
        raise AssertionError(
            "Delta_align must subtract comparison similarity from "
            "aligned-reference similarity."
        )
    if bool(result.loc[3, "aligned_reference_eligible"]):
        raise AssertionError(
            "A comparison with a = b must be excluded."
        )
    if pd.notna(result.loc[3, "aligned_reference_advantage"]):
        raise AssertionError(
            "Ineligible aligned-reference comparisons must be undefined."
        )

    references = qa32_select_comparison_references(
        pd.DataFrame(
            {
                "record_key": ["e", "e", "e"],
                "source_lid": ["2", "1", "3"],
                "source_text": ["two", "one", "three"],
            }
        )
    )
    if references.loc[0, "comparison_source_lid"] != "1":
        raise AssertionError(
            "Comparison-reference selection must be deterministic."
        )

