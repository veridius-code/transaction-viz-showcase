"""How well does the detector do on data it was not tuned on?

Re-generates the ledger with other random seeds (same planted anomalies,
different noise) and scores the detector at default settings. This guards
against a method that only works on the one published sample.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config, generate_data
from src.analysis import DetectionConfig, daily_series, detect_clusters

SEEDS = [s for s in range(1, 31) if s != config.SEED]
TOL_DAYS = 1
OUT = config.DATA_DIR / "robustness.json"


def score(seed: int, cfg: DetectionConfig) -> dict:
    truth = generate_data.generate_truth(seed)
    truth["date"] = pd.to_datetime(truth["date"])
    _, clusters = detect_clusters(daily_series(truth), cfg)
    planted = [(pd.Timestamp(p["start"]), pd.Timestamp(p["end"]), p["kind"]) for p in generate_data.PLANTED]
    found_exact = found_any = 0
    for s, e, kind in planted:
        overlaps = [c for c in clusters if c.kind == kind and c.start <= e and c.end >= s]
        found_any += bool(overlaps)
        found_exact += any(abs((c.start - s).days) <= TOL_DAYS and abs((c.end - e).days) <= TOL_DAYS
                           for c in overlaps)
    false = sum(not any(c.kind == k and c.start <= e and c.end >= s for s, e, k in planted) for c in clusters)
    return {"seed": seed, "found": found_any, "found_within_1_day": found_exact, "false_clusters": false}


def main() -> dict:
    cfg = DetectionConfig()
    rows = [score(s, cfg) for s in SEEDS]
    df = pd.DataFrame(rows)
    summary = {
        "n_seeds": len(rows),
        "planted_total": 3 * len(rows),
        "found": int(df["found"].sum()),
        "found_within_1_day": int(df["found_within_1_day"].sum()),
        "false_clusters": int(df["false_clusters"].sum()),
        "seeds_with_false_clusters": int((df["false_clusters"] > 0).sum()),
        "rows": rows,
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    s = main()
    print({k: v for k, v in s.items() if k != "rows"})
