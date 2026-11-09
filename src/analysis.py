"""Baseline, cluster detection and cluster-vs-normal comparison."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.clean import CleaningLog, clean

MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class DetectionConfig:
    window: int = 28            # rolling baseline window, days
    spread_window: int = 56     # window for the MAD spread (longer = steadier)
    trailing: bool = True       # use the previous N days (else centred)
    z_threshold: float = 2.5    # robust z needed to flag a day
    min_run: int = 2            # consecutive flagged days needed to call a cluster
    max_passes: int = 5         # re-estimate the baseline without flagged days
    n_boot: int = 5000          # bootstrap resamples for the cluster-mean CI
    seed: int = config.SEED


@dataclass
class Cluster:
    kind: str                   # "spend_cluster" | "inflow_drought"
    series: str                 # column of the daily table it was found in
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    total: float                # actual money over the cluster
    mean_adj: float             # mean day-of-week-adjusted daily value
    typical: float              # median of same-length normal window means
    xfold: float                # mean_adj / typical
    pct_below: float            # % of normal windows with a lower mean
    ci_low: float
    ci_high: float
    ref_means: np.ndarray = field(repr=False)

    @property
    def label(self) -> str:
        return date_range_label(self.start, self.end)

    @property
    def plain_percentile(self) -> str:
        """Descriptive wording for the empirical percentile (not a p-value)."""
        n = len(self.ref_means)
        if self.kind == "spend_cluster":
            share = self.pct_below
            word = "higher"
        else:
            share = 100.0 - self.pct_below
            word = "lower"
        if share >= 100.0:
            return f"{word.capitalize()} than all {n} normal {self.n_days}-day periods"
        return f"{word.capitalize()} than {share:.0f}% of normal {self.n_days}-day periods"


@dataclass
class Results:
    transactions: pd.DataFrame
    log: CleaningLog
    daily: pd.DataFrame
    weekly: pd.DataFrame
    clusters: list[Cluster]
    cfg: DetectionConfig


def date_range_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start.month == end.month:
        return f"{start:%b} {start.day}-{end.day}"
    return f"{start:%b} {start.day} - {end:%b} {end.day}"


# --------------------------------------------------------------------------- #
# Daily series
# --------------------------------------------------------------------------- #
def daily_series(tx: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(tx["date"].min(), tx["date"].max(), freq="D")
    out = tx[tx["direction"] == "out"]
    sched = out["category"].isin(config.SCHEDULED_CATEGORIES)

    def per_day(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("date")["amount"].sum().reindex(days, fill_value=0.0)

    d = pd.DataFrame(index=days)
    d.index.name = "date"
    d["outflow_total"] = per_day(out)
    d["outflow_scheduled"] = per_day(out[sched])
    d["outflow_operating"] = per_day(out[~sched])
    d["inflow"] = per_day(tx[tx["direction"] == "in"])
    d["net"] = d["inflow"] - d["outflow_total"]
    d["cumulative_net"] = d["net"].cumsum()
    d["operating_7d"] = d["outflow_operating"].rolling(7, min_periods=1).mean()
    d["inflow_7d"] = d["inflow"].rolling(7, min_periods=1).mean()
    return d


def weekly_net(daily: pd.DataFrame) -> pd.DataFrame:
    w = daily[["inflow", "outflow_total", "net"]].resample("W-MON", label="left", closed="left").sum()
    w.index.name = "week_start"
    return w


# --------------------------------------------------------------------------- #
# Robust rolling baseline
# --------------------------------------------------------------------------- #
def _rolling_median(x: pd.Series, window: int, trailing: bool) -> pd.Series:
    """Rolling median that skips NaNs (masked days). Needs half a window of data."""
    vals = x.to_numpy(dtype=float)
    n, h = len(vals), window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = (i - window, i) if trailing else (i - h, i + h + 1)
        w = vals[max(0, lo): min(n, hi)]
        w = w[~np.isnan(w)]
        if len(w) >= window // 2:
            out[i] = np.median(w)
    return pd.Series(out, index=x.index)


def _runs(flags: pd.Series, min_run: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    runs, start, prev = [], None, None
    for day, f in flags.items():
        if f and start is None:
            start = day
        if not f and start is not None:
            runs.append((start, prev))
            start = None
        prev = day
    if start is not None:
        runs.append((start, prev))
    return [(s, e) for s, e in runs if (e - s).days + 1 >= min_run]


def robust_baseline(series: pd.Series, direction: str, cfg: DetectionConfig) -> pd.DataFrame:
    """Robust rolling baseline that allows for day-of-week patterns.

    1. Divide each day by its weekday factor (e.g. Sundays run at ~45% of an
       average day) so every day is on the same footing.
    2. Baseline = rolling median of the adjusted values over `window` days;
       spread = MAD x 1.4826 of the deviations from that baseline, over
       `spread_window` days (a longer window keeps the spread stable).
    3. Weekend days are also noisier (fewer transactions), so the robust z is
       rescaled by each weekday's own typical z-spread.
    4. Days inside flagged runs are removed and steps 1-3 repeated until the
       runs settle, so a cluster never inflates its own baseline. (Removing
       every single high day would trim the normal upper tail and make the
       spread too narrow, so isolated days stay in.)

    direction="high" flags unusually high days, "low" unusually low days.
    """
    dow = pd.Series(series.index.day_of_week, index=series.index)
    mask = pd.Series(False, index=series.index)
    for _ in range(cfg.max_passes):
        factor = series.where(~mask).groupby(dow).median()
        factor = factor / factor.mean()
        f = dow.map(factor)
        adj = series / f
        med = _rolling_median(adj.where(~mask), cfg.window, cfg.trailing)
        resid = adj - med
        spread = MAD_TO_SD * _rolling_median(resid.where(~mask).abs(), cfg.spread_window, cfg.trailing)
        z_raw = resid / spread
        z_kept = z_raw.where(~mask)
        dow_scale = z_kept.groupby(dow).agg(lambda v: MAD_TO_SD * (v - v.median()).abs().median())
        z = z_raw / dow.map(dow_scale)
        flagged = z > cfg.z_threshold if direction == "high" else z < -cfg.z_threshold
        in_run = pd.Series(False, index=series.index)
        for s, e in _runs(flagged, cfg.min_run):
            in_run.loc[s:e] = True
        if in_run.equals(mask):
            break
        mask = in_run
    return pd.DataFrame({
        "value": series,
        "dow_factor": f,
        "adjusted": adj,                        # $ on a typical-day footing
        "baseline": med,                        # typical adjusted $ per day
        "band_low": (med - spread).clip(lower=0),   # typical range (+/- 1 robust SD)
        "band_high": med + spread,
        "z": z,
        "flagged": flagged,
    })


# --------------------------------------------------------------------------- #
# Cluster vs normal-window comparison
# --------------------------------------------------------------------------- #
def compare_to_normal(adj: pd.Series, start, end, excluded: pd.Series,
                      cfg: DetectionConfig) -> dict:
    n_days = (end - start).days + 1
    cluster_vals = adj.loc[start:end].to_numpy()
    cluster_mean = float(cluster_vals.mean())

    win_means = adj.rolling(n_days).mean()
    win_has_excl = excluded.astype(float).rolling(n_days).max().fillna(1.0) > 0
    ref = win_means[~win_has_excl].dropna().to_numpy()

    rng = np.random.default_rng(cfg.seed)
    boot = rng.choice(cluster_vals, size=(cfg.n_boot, n_days), replace=True).mean(axis=1)
    typical = float(np.median(ref))
    return {
        "mean_adj": cluster_mean,
        "typical": typical,
        "xfold": cluster_mean / typical,
        "pct_below": float((ref < cluster_mean).mean() * 100),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "ref_means": ref,
    }


def detect_clusters(daily: pd.DataFrame, cfg: DetectionConfig) -> tuple[pd.DataFrame, list[Cluster]]:
    specs = [
        ("spend_cluster", "outflow_operating", "high", "out"),
        ("inflow_drought", "inflow", "low", "in"),
    ]
    daily = daily.copy()
    clusters: list[Cluster] = []
    for kind, col, direction, prefix in specs:
        b = robust_baseline(daily[col], direction, cfg)
        for c in ("dow_factor", "adjusted", "baseline", "band_low", "band_high", "z", "flagged"):
            daily[f"{prefix}_{c}"] = b[c]
        runs = _runs(b["flagged"], cfg.min_run)
        in_cluster = pd.Series(False, index=daily.index)
        for s, e in runs:
            in_cluster.loc[s:e] = True
        daily[f"{prefix}_in_cluster"] = in_cluster
        for s, e in runs:
            stats = compare_to_normal(b["adjusted"], s, e, in_cluster, cfg)
            clusters.append(Cluster(
                kind=kind, series=col, start=s, end=e, n_days=(e - s).days + 1,
                total=float(daily.loc[s:e, col].sum()), **stats,
            ))
    clusters.sort(key=lambda c: c.start)
    return daily, clusters


def cluster_table(clusters: list[Cluster]) -> pd.DataFrame:
    return pd.DataFrame([{
        "period": c.label,
        "type": "Spend cluster" if c.kind == "spend_cluster" else "Inflow drought",
        "days": c.n_days,
        "total ($)": round(c.total),
        "mean per day, adj. ($)": round(c.mean_adj),
        "typical per day ($)": round(c.typical),
        "x typical": round(c.xfold, 2),
        "95% CI per day ($)": f"{c.ci_low:,.0f} - {c.ci_high:,.0f}",
        "compared with normal periods": c.plain_percentile,
    } for c in clusters])


def run_pipeline(cfg: DetectionConfig | None = None, raw_path=config.RAW_CSV) -> Results:
    cfg = cfg or DetectionConfig()
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    tx, log = clean(raw)
    daily = daily_series(tx)
    daily, clusters = detect_clusters(daily, cfg)
    return Results(tx, log, daily, weekly_net(daily), clusters, cfg)
