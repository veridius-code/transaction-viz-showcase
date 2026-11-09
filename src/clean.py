"""Clean the raw transaction export and log every fix."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATE_FORMATS = {
    "ISO (2025-03-05)": "%Y-%m-%d",
    "UK day-first (05/03/2025)": "%d/%m/%Y",
    "Text month (Mar 05, 2025)": "%b %d, %Y",
}
DIRECTION_MAP = {
    "in": "in", "inflow": "in", "credit": "in",
    "out": "out", "outflow": "out", "debit": "out",
}


@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, detail: str, count: int) -> None:
        self.steps.append({"step": step, "detail": detail, "rows_affected": int(count)})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "detail", "rows_affected"])


def _pick_spelling(variants: pd.Series) -> str:
    """Most common mixed-case spelling; ALL-CAPS / all-lower exports are ignored."""
    mixed = variants[~(variants.str.isupper() | variants.str.islower())]
    if len(mixed):
        return mixed.value_counts().index[0]
    return variants.iloc[0].capitalize()


def _canonical_text(s: pd.Series) -> pd.Series:
    """Map every casing variant of a value to one spelling."""
    key = s.str.lower()
    nonblank = s != ""
    canon = s[nonblank].groupby(key[nonblank]).agg(_pick_spelling)
    return key.map(canon).fillna(s)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    df = raw.copy()
    log.add("Load", "Rows in raw export", len(df))
    for col in ("transaction_id", "date", "amount", "direction", "category", "counterparty"):
        df[col] = df[col].fillna("").astype(str)

    # 1. Whitespace
    n_ws = 0
    for col in ("category", "counterparty", "direction", "transaction_id"):
        stripped = df[col].str.strip()
        n_ws += int((stripped != df[col]).sum())
        df[col] = stripped
    log.add("Trim whitespace", "Leading/trailing spaces removed from text fields", n_ws)

    # 2. Casing
    n_case = 0
    for col in ("category", "counterparty"):
        fixed = _canonical_text(df[col])
        n_case += int((fixed != df[col]).sum())
        df[col] = fixed
    log.add("Standardise casing", "Category/counterparty variants mapped to one spelling", n_case)

    # 3. Direction labels
    d = df["direction"].str.lower().map(DIRECTION_MAP)
    if d.isna().any():
        raise ValueError(f"Unknown direction labels: {sorted(df.loc[d.isna(), 'direction'].unique())}")
    log.add("Normalise direction", "Labels like IN / credit / outflow mapped to in/out",
            int((d != df["direction"]).sum()))
    df["direction"] = d

    # 4. Dates (explicit formats only; day-first is the documented source convention)
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    counts = {}
    for label, fmt in DATE_FORMATS.items():
        todo = parsed.isna()
        p = pd.to_datetime(df.loc[todo, "date"], format=fmt, errors="coerce")
        counts[label] = int(p.notna().sum())
        parsed.loc[p.index] = p
    if parsed.isna().any():
        raise ValueError(f"{int(parsed.isna().sum())} dates could not be parsed")
    n_converted = sum(v for k, v in counts.items() if not k.startswith("ISO"))
    detail = "; ".join(f"{k}: {v:,}" for k, v in counts.items())
    log.add("Normalise dates", f"Converted to ISO dates ({detail})", n_converted)
    df["date"] = parsed

    # 5. Amounts
    a = df["amount"]
    is_currency = a.str.contains(r"[$,]", regex=True)
    is_signed = a.str.startswith("-")
    num = pd.to_numeric(a.str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    if num.isna().any():
        raise ValueError(f"{int(num.isna().sum())} amounts could not be parsed")
    log.add("Parse currency text", 'Strings like "$1,234.50" converted to numbers', int(is_currency.sum()))
    log.add("Remove sign duplication", "Negative amounts on outflows made positive (direction carries the sign)",
            int(is_signed.sum()))
    df["amount"] = num.abs().round(2)

    # 6. Duplicates: the same transaction id exported more than once
    dup = df.duplicated("transaction_id", keep="first")
    conflicts = (
        df[df["transaction_id"].isin(df.loc[dup, "transaction_id"])]
        .groupby("transaction_id")[["date", "amount", "direction", "counterparty"]]
        .nunique().gt(1).any(axis=1).sum()
    )
    if conflicts:
        raise ValueError(f"{conflicts} duplicated ids have conflicting values")
    df = df[~dup].copy()
    log.add("Remove duplicates", "Repeated transaction IDs dropped (values identical after cleaning)", int(dup.sum()))

    # 7. Missing categories: infer from counterparty where it is unambiguous
    blank = df["category"] == ""
    known = df[~blank].groupby("counterparty")["category"].agg(lambda v: v.value_counts().index[0])
    unique = df[~blank].groupby("counterparty")["category"].nunique() == 1
    inferable = blank & df["counterparty"].map(unique).fillna(False).astype(bool)
    df.loc[inferable, "category"] = df.loc[inferable, "counterparty"].map(known)
    log.add("Fill missing categories", "Recovered from the counterparty's only known category", int(inferable.sum()))
    still = df["category"] == ""
    df.loc[still, "category"] = "Uncategorised"
    log.add("Flag unknown categories", 'Left as "Uncategorised"', int(still.sum()))

    df = df.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    df = df[["transaction_id", "date", "amount", "direction", "category", "counterparty"]]
    log.add("Result", "Clean rows ready for analysis", len(df))
    return df, log
