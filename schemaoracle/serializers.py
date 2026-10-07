"""
Column serialisation and value sampling, ported from SchemaContext.

Two changes from the original: the LLM-based encoding modes are dropped, and
every stochastic mode takes a seed instead of relying on a global
`np.random.seed`, so a given (values, n, mode, seed) always gives the same
sample.
"""

from __future__ import annotations

import re

import mmh3
import numpy as np
import pandas as pd

PHI_FRACTION = 0.6180339887  # golden ratio conjugate, phi - 1

# Inlined from magneto.utils.constants.
NULL_REPRESENTATIONS = {
    "nan", "none", "null", "na", "n/a", "nil", "", "missing", "unknown",
    "not available", "not applicable", "#n/a", "#na",
}

BINARY_VALUES = {
    "yes", "no", "true", "false", "1", "0", "y", "n", "t", "f",
    "positive", "negative", "pos", "neg",
}

KEY_REPRESENTATIONS = [
    "id", "key", "code", "uuid", "guid", "identifier", "pk", "fk",
]


def _split_camel_case(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


def _remove_invalid_chars(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", s)


def clean_column_name(col_name: str) -> str:
    """Lowercase, split camelCase, strip punctuation, collapse whitespace."""
    col_name = str(col_name).strip()
    col_name = _split_camel_case(col_name)
    col_name = col_name.lower()
    col_name = _remove_invalid_chars(col_name)
    return re.sub(r"\s+", " ", col_name).strip()


def _clean_element(x):
    if isinstance(x, str):
        if x.lower() in NULL_REPRESENTATIONS:
            return None
        val = _split_camel_case(x)
        val = _remove_invalid_chars(val.strip().lower())
        return val if val else None
    return x


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise cleaning of every cell, as in Magneto's clean_df."""
    return df.apply(lambda col: col.apply(_clean_element))


def detect_column_type(
    col: pd.Series,
    key_threshold: float = 0.8,
    numeric_threshold: float = 0.90,
) -> str:
    """Coarse semantic type, used in the serialised profile."""
    temp_col = pd.to_numeric(col, errors="coerce")
    if not temp_col.isnull().all():
        return "numerical"

    col_name_lower = str(col.name or "").lower()
    if "gene" in col_name_lower:
        return "gene"
    if "date" in col_name_lower:
        return "date"

    unique_values = col.dropna().unique()
    if len(unique_values) == 0:
        return "unknown"

    if (
        len(unique_values) / max(len(col), 1) > key_threshold
        and col.dtype not in [np.float64, np.float32, np.float16]
    ):
        return "key"

    if any(
        col_name_lower.startswith(r) or col_name_lower.endswith(r)
        for r in KEY_REPRESENTATIONS
    ):
        return "key"

    if col.dtype in [np.float64, np.int64]:
        return "numerical"

    numeric_unique = pd.to_numeric(pd.Series(unique_values), errors="coerce").dropna()
    if not numeric_unique.empty:
        if len(numeric_unique) / len(unique_values) > numeric_threshold:
            if len(numeric_unique) > 2:
                return "numerical"
            if set(map(int, numeric_unique)).issubset({0, 1}):
                return "binary"
            return "numerical"

    if len(unique_values) == 2 and all(
        str(v).lower() in BINARY_VALUES for v in unique_values
    ):
        return "binary"

    return "categorical"


def _fibonacci_hash(x: int) -> float:
    """
    Map an integer into [0, 1) by golden-ratio hashing. `mmh3.hash` is signed;
    Python's `%` keeps the result non-negative where C's would not.
    """
    return (x * PHI_FRACTION) % 1


# Stands in for a hash of exactly 0.0, which would divide by zero in
# `priority_sampling`. It gives that value the largest finite priority.
_MIN_PRIORITY_HASH = 2.0**-40


def _priority_denominator(value: str) -> float:
    """The u_i of priority sampling: a hash-derived draw in (0, 1)."""
    h = _fibonacci_hash(mmh3.hash(value, 42))
    return h if h else _MIN_PRIORITY_HASH


def _sorted_strings(values) -> list[str]:
    """
    Sort, then stringify. The other order puts "10.0" before "2.0" and changes
    the embedding of every numeric column. Mixed types fall back to string
    order.
    """
    values = list(values)
    try:
        return [str(v) for v in sorted(values)]
    except TypeError:
        return sorted(str(v) for v in values)


SAMPLING_MODES = [
    "random",
    "frequent",
    "mixed",
    "weighted",
    "priority_sampling",
    "consistent_sampling",
]


def get_samples(
    values: pd.Series,
    n: int = 10,
    mode: str = "priority_sampling",
    seed: int = 0,
) -> list[str]:
    """
    Sample up to *n* representative values from a Series.

        random              uniform draw over unique values
        frequent            the n most common
        mixed               half most frequent, half evenly spaced
        weighted            proportional to value counts
        priority_sampling   freq / fibonacci_hash(mmh3(value))
        consistent_sampling the hash alone, frequency independent

    The two hashed modes are deterministic and coordinated: a value hashes the
    same in every column, so columns over the same domain sample overlapping
    values. Note that `priority_sampling` is not "most frequent": a rare value
    with a low hash outranks a common one.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode {mode!r}. Valid: {SAMPLING_MODES}")

    unique_values = values.dropna().unique()
    total_unique = len(unique_values)

    if total_unique <= n:
        return sorted(str(v) for v in unique_values)

    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(total_unique, size=n, replace=False)
        return _sorted_strings(unique_values[idx])

    if mode == "frequent":
        top = values.dropna().value_counts().head(n).index.tolist()
        return _sorted_strings(top)

    if mode == "mixed":
        n_freq = n // 2
        n_div = n - n_freq
        freq_vals = values.dropna().value_counts().head(n_freq).index.tolist()
        step = max(1, total_unique // n_div) if n_div > 0 else 1
        div_vals = unique_values[::step][:n_div]
        return sorted({str(v) for v in list(freq_vals) + list(div_vals)})

    if mode == "weighted":
        rng = np.random.default_rng(seed)
        counts = values.dropna().value_counts(sort=False)
        weights = (
            (counts / counts.sum())
            .reindex(pd.Index(unique_values))
            .fillna(0)
            .values
        )
        weights = weights / weights.sum()
        idx = rng.choice(total_unique, size=n, replace=False, p=weights)
        return [str(unique_values[i]) for i in idx]

    if mode == "priority_sampling":
        counts = values.dropna().value_counts(sort=False)
        priorities = pd.Series(
            {
                val: freq / _priority_denominator(str(val))
                for val, freq in counts.items()
            }
        )
        return [str(v) for v in priorities.nlargest(n).index.tolist()]

    if mode == "consistent_sampling":
        counts = values.dropna().value_counts(sort=False)
        priorities = pd.Series(
            {val: _fibonacci_hash(mmh3.hash(str(val), 42)) for val in counts.keys()}
        )
        return [str(v) for v in priorities.nlargest(n).index.tolist()]

    raise ValueError(f"Unreachable: unknown mode {mode!r}")


def get_distribution(values: pd.Series) -> dict[str, str] | None:
    """{'min': ..., 'max': ...} for numeric columns, or None."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return {"min": str(numeric.min()), "max": str(numeric.max())}


ENCODING_MODES = [
    "header_values_default",   # header [sep] type [sep] values
    "header_values_repeat",    # header x5 [sep] type [sep] values
    "header_values_verbose",   # Column: ... Type: ... Values: ...
    "header_values_xml",       # <column><name>...<type>...<values>...
    "header_only",             # column name only
]

_SEP = " "


def serialize_column(
    col_name: str,
    series: pd.Series,
    encoding_mode: str = "header_values_verbose",
    sampling_mode: str = "priority_sampling",
    n_samples: int = 10,
    seed: int = 0,
) -> str:
    """One string for a column, for embedding. The default matches Magneto's."""
    if encoding_mode not in ENCODING_MODES:
        raise ValueError(
            f"Unknown encoding mode {encoding_mode!r}. Valid: {ENCODING_MODES}"
        )

    header = clean_column_name(col_name)
    tokens = get_samples(series, n=n_samples, mode=sampling_mode, seed=seed)
    data_type = detect_column_type(series)
    values_str = _SEP.join(tokens)

    if encoding_mode == "header_values_default":
        return f"{header}{_SEP}{data_type}{_SEP}{values_str}"

    if encoding_mode == "header_values_repeat":
        repeated = _SEP.join([header] * 5)
        return f"{repeated}{_SEP}{data_type}{_SEP}{values_str}"

    if encoding_mode == "header_values_verbose":
        return f"Column: {header}{_SEP}Type: {data_type}{_SEP}Values: {values_str}"

    if encoding_mode == "header_values_xml":
        parts = [
            "<column>",
            f"<name>{header}</name>",
            f"<type>{data_type}</type>",
            f"<values>{values_str}</values>",
        ]
        if data_type == "numerical":
            dist = get_distribution(series)
            if dist is not None:
                parts.append(f"<min>{dist['min']}</min>")
                parts.append(f"<max>{dist['max']}</max>")
        parts.append("</column>")
        return "".join(parts)

    if encoding_mode == "header_only":
        return header

    raise ValueError(f"Unreachable: unknown mode {encoding_mode!r}")


def build_column_profiles(
    df: pd.DataFrame,
    encoding_mode: str = "header_values_verbose",
    sampling_mode: str = "priority_sampling",
    n_samples: int = 10,
    seed: int = 0,
) -> dict[str, str]:
    """Serialise every column: column name -> profile string."""
    return {
        col: serialize_column(
            col_name=col,
            series=df[col],
            encoding_mode=encoding_mode,
            sampling_mode=sampling_mode,
            n_samples=n_samples,
            seed=seed,
        )
        for col in df.columns
    }
