"""Generate a synthetic 12-month transaction ledger.

Everything here is invented. The data is seeded so results are reproducible.
Two files are written:

* data/transactions_clean_truth.csv  - the ground-truth ledger
* data/transactions_raw.csv          - the same ledger with realistic export mess
* data/planted_anomalies.json        - the three anomalies planted on purpose
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config

# category -> (lognormal mu, sigma, relative frequency, counterparties)
OUTFLOW_CATEGORIES = {
    "Suppliers":   (6.6, 0.50, 0.26, [f"Supplier {c}" for c in "ABCDEFGH"]),
    "Logistics":   (6.1, 0.45, 0.16, [f"Logistics Co {i}" for i in range(1, 5)]),
    "Software":    (5.4, 0.45, 0.11, [f"Software Vendor {i}" for i in range(1, 6)]),
    "Travel":      (5.8, 0.50, 0.10, [f"Travel Agency {i}" for i in range(1, 4)]),
    "Utilities":   (5.6, 0.35, 0.06, ["Utility Provider 1", "Utility Provider 2"]),
    "Marketing":   (6.3, 0.50, 0.08, [f"Agency {c}" for c in "PQR"]),
    "Contractors": (6.9, 0.40, 0.08, [f"Contractor {i}" for i in range(1, 7)]),
    "Equipment":   (7.1, 0.45, 0.03, [f"Equipment Dealer {i}" for i in range(1, 4)]),
    "Office":      (4.8, 0.45, 0.12, ["Office Supplies Ltd", "Cleaning Service", "Catering Co"]),
}
INFLOW_CATEGORIES = {
    "Customer receipts": (7.26, 0.30, 0.90, [f"Customer {i:02d}" for i in range(1, 31)]),
    # Distinct suffix so every counterparty maps to exactly one category.
    "Refunds":           (5.5, 0.60, 0.08, [f"Supplier {c} (refund)" for c in "ABCD"]),
    "Interest":          (4.5, 0.30, 0.02, ["Bank Interest"]),
}

# Expected transaction counts per weekday (Mon=0 ... Sun=6). Counts are drawn
# as Binomial(2*lambda, 0.5): same mean as Poisson, a little less dispersed,
# which is closer to how a business with steady trading volumes behaves.
OUT_LAMBDA = [20, 21, 21, 20, 19, 9, 8]
IN_LAMBDA = [16, 16, 16, 16, 16, 7, 7]
MIN_TX, MAX_TX = 15, 40

PAYROLL_BASE = 165_000.0
RENT_BASE = 24_000.0
TREND_PER_YEAR = 0.08

# Planted anomalies (dates are inclusive)
PLANTED = [
    {"id": "A", "kind": "spend_cluster", "direction": "out", "category": "Equipment",
     "start": "2025-03-11", "end": "2025-03-13", "multiplier": 2.6},
    {"id": "B", "kind": "spend_cluster", "direction": "out", "category": "Contractors",
     "start": "2025-08-11", "end": "2025-08-15", "multiplier": 2.0},
    {"id": "C", "kind": "inflow_drought", "direction": "in", "category": "Customer receipts",
     "start": "2025-10-14", "end": "2025-10-19", "keep_fraction": 0.05},
]


def _last_business_day(month_start: pd.Timestamp) -> pd.Timestamp:
    d = month_start + pd.offsets.MonthEnd(0)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _expected_operating(dow: int, trend: float) -> float:
    probs = np.array([v[2] for v in OUTFLOW_CATEGORIES.values()])
    means = np.array([np.exp(v[0] + v[1] ** 2 / 2) for v in OUTFLOW_CATEGORIES.values()])
    return float(OUT_LAMBDA[dow] * (probs / probs.sum()) @ means * trend)


def _draw(rng, spec: dict, n: int, trend: float) -> list[tuple[str, float, str]]:
    names = list(spec)
    probs = np.array([spec[k][2] for k in names])
    probs = probs / probs.sum()
    cats = rng.choice(names, size=n, p=probs)
    rows = []
    for cat in cats:
        mu, sigma, _, parties = spec[cat]
        amt = float(np.exp(rng.normal(mu, sigma))) * trend
        rows.append((cat, round(amt, 2), str(rng.choice(parties))))
    return rows


def generate_truth(seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    planted = {p["id"]: (pd.Timestamp(p["start"]), pd.Timestamp(p["end"]), p) for p in PLANTED}

    records = []
    for i, day in enumerate(days):
        trend = 1.0 + TREND_PER_YEAR * i / len(days)
        dow = day.weekday()
        n_out = rng.binomial(2 * OUT_LAMBDA[dow], 0.5)
        n_in = rng.binomial(2 * IN_LAMBDA[dow], 0.5)
        n_out = max(n_out, MIN_TX - n_in)
        n_out = min(n_out, MAX_TX - n_in)

        outs = _draw(rng, OUTFLOW_CATEGORIES, n_out, trend)
        ins = _draw(rng, INFLOW_CATEGORIES, n_in, trend)

        # Inflow drought: most customer receipts simply do not arrive.
        s, e, p = planted["C"]
        if s <= day <= e:
            ins = [r for r in ins if r[0] != "Customer receipts" or rng.random() < p["keep_fraction"]]

        # Spend clusters: extra spend in one category, sized relative to the
        # expected operating spend for that weekday.
        for key in ("A", "B"):
            s, e, p = planted[key]
            if s <= day <= e:
                extra_total = _expected_operating(dow, trend) * (p["multiplier"] - 1.0)
                mu, sigma, _, parties = OUTFLOW_CATEGORIES[p["category"]]
                while extra_total > 0:
                    amt = min(extra_total, float(np.exp(rng.normal(mu, sigma))) * trend)
                    outs.append((p["category"], round(amt, 2), str(rng.choice(parties))))
                    extra_total -= amt

        for cat, amt, cp in outs:
            records.append((day, amt, "out", cat, cp))
        for cat, amt, cp in ins:
            records.append((day, amt, "in", cat, cp))

        # Scheduled fixed costs
        if day.day == 1:
            records.append((day, round(RENT_BASE * trend, 2), "out", "Rent", "Landlord Property Co"))
        if day == _last_business_day(day.replace(day=1)):
            pay = PAYROLL_BASE * trend * float(rng.normal(1.0, 0.01))
            records.append((day, round(pay, 2), "out", "Payroll", "Payroll Provider"))

    df = pd.DataFrame(records, columns=["date", "amount", "direction", "category", "counterparty"])
    df = df.sort_values(["date", "direction"], kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{n:06d}" for n in range(1, len(df) + 1)])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def make_messy(truth: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Add the sort of mess a real export tends to have."""
    rng = np.random.default_rng(seed + 1)
    raw = truth.copy()
    n = len(raw)
    dates = pd.to_datetime(raw["date"])

    # Mixed date formats: ISO, UK day-first, and "Mar 05, 2025"
    fmt_choice = rng.choice(3, size=n, p=[0.6, 0.25, 0.15])
    raw["date"] = np.where(
        fmt_choice == 0, dates.dt.strftime("%Y-%m-%d"),
        np.where(fmt_choice == 1, dates.dt.strftime("%d/%m/%Y"), dates.dt.strftime("%b %d, %Y")),
    )

    # Amounts as text: currency strings, plain numbers, signed outflows
    amt_choice = rng.choice(3, size=n, p=[0.4, 0.45, 0.15])
    amounts = []
    for a, d, c in zip(truth["amount"], truth["direction"], amt_choice):
        if c == 0:
            amounts.append(f"${a:,.2f}")
        elif c == 1:
            amounts.append(f"{a:.2f}")
        else:
            amounts.append(f"-{a:.2f}" if d == "out" else f"{a:.2f}")
    raw["amount"] = amounts

    # Direction labels vary between source systems
    dir_map = {"in": ["in", "IN", "In", "inflow", "credit"], "out": ["out", "OUT", "Out", "outflow", "debit"]}
    raw["direction"] = [str(rng.choice(dir_map[d])) for d in truth["direction"]]

    # Inconsistent casing / stray whitespace in text fields
    def scuff(s: str) -> str:
        r = rng.random()
        if r < 0.15:
            s = s.upper()
        elif r < 0.30:
            s = s.lower()
        if rng.random() < 0.15:
            s = f"  {s} "
        return s

    raw["category"] = [scuff(c) for c in raw["category"]]
    raw["counterparty"] = [scuff(c) for c in raw["counterparty"]]

    # Missing categories (~3%)
    missing = rng.random(n) < 0.03
    raw.loc[missing, "category"] = ""

    # Duplicate rows (~2%): the same transaction exported twice, sometimes
    # with different formatting, so exact-row matching alone would miss it.
    dup_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    dups = raw.iloc[dup_idx].copy()
    redo = rng.random(len(dups)) < 0.5
    dups.loc[redo, "date"] = pd.to_datetime(truth.iloc[dup_idx]["date"]).dt.strftime("%d/%m/%Y")[redo].values
    raw = pd.concat([raw, dups], ignore_index=True)
    raw = raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return raw


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    truth = generate_truth()
    raw = make_messy(truth)
    truth.to_csv(config.TRUTH_CSV, index=False)
    raw.to_csv(config.RAW_CSV, index=False)
    with open(config.PLANTED_JSON, "w") as f:
        json.dump(PLANTED, f, indent=2)
    print(f"truth rows: {len(truth):,}  raw rows: {len(raw):,}")


if __name__ == "__main__":
    main()
