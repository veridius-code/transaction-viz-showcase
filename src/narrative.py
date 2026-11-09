"""Titles and plain-language insights, written from the computed results.

Keeping the wording in one place means the static charts, the interactive
charts, the notebook and the site always quote the same numbers.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.analysis import Cluster, Results


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def money_k(x: float) -> str:
    v = abs(x)
    s = f"${v / 1e6:,.2f}m" if v >= 1e6 else f"${v / 1e3:,.0f}k"
    return f"-{s}" if x < 0 else s


def spend(r: Results) -> list[Cluster]:
    return [c for c in r.clusters if c.kind == "spend_cluster"]


def droughts(r: Results) -> list[Cluster]:
    return [c for c in r.clusters if c.kind == "inflow_drought"]


def callout(c: Cluster, r: Results | None = None) -> str:
    if c.kind == "spend_cluster":
        text = f"{c.label}: {c.xfold:.1f}x typical daily spend"
        return f"{text}\nmostly {top_category(r, c).lower()}" if r is not None else text
    return f"{c.label}: inflows at {c.xfold:.0%} of typical"


def kind_label(c: Cluster) -> str:
    return "spending burst" if c.kind == "spend_cluster" else "inflow drought"


NUM_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


# --------------------------------------------------------------------------- #
# Chart 1 - velocity
# --------------------------------------------------------------------------- #
def velocity_title(r: Results) -> str:
    s = spend(r)
    if not s:
        return "Daily spending stayed within its normal range all year"
    top = max(s, key=lambda c: c.xfold)
    n = NUM_WORDS.get(len(s), str(len(s)))
    word = "burst" if len(s) == 1 else "bursts"
    return f"{n} short {word} pushed daily spending up to {top.xfold:.1f}x its normal level"


VELOCITY_SUBTITLE = (f"{config.COMPANY_SHORT}: daily operating spend, 2025 (excludes scheduled payroll and rent). "
                     "Blue band = typical range, line = 7-day average.")


VELOCITY_SUBTITLE_WEB = ("Daily operating spend, 2025 (excludes scheduled payroll and rent). Blue band = typical "
                         "range, black line = 7-day average, orange bars = flagged days.")


def velocity_insight(r: Results) -> str:
    s = spend(r)
    d = r.daily
    typical = d.loc[~d["out_in_cluster"], "outflow_operating"].median()
    normal = d.loc[~d["out_in_cluster"], "outflow_operating"]
    q1 = normal[normal.index.quarter == 1].median()
    q4 = normal[normal.index.quarter == 4].median()
    parts = [f"On a normal day the business spends about {money_k(typical)} on operations, "
             f"and that level moved {q4 / q1 - 1:+.0%} between Q1 and Q4."]
    if s:
        bits = [f"{c.label} ({c.n_days} days, {money_k(c.total)} in total, {c.xfold:.1f}x typical)" for c in s]
        parts.append("Two periods break clearly away from that baseline: " + " and ".join(bits) + "."
                     if len(s) == 2 else "Flagged periods: " + "; ".join(bits) + ".")
        if _single_category(r, s):
            cats = ", ".join(f"{top_category(r, c)} in {c.start:%B}" for c in s)
            parts.append(f"Each was driven by one category ({cats}), so each can be checked quickly "
                         "against approvals or purchase orders.")
        else:
            parts.append("Each is worth checking against approvals or purchase orders.")
    return " ".join(parts)


def _single_category(r: Results, clusters: list[Cluster]) -> bool:
    tx = r.transactions
    for c in clusters:
        out = tx[(tx["direction"] == "out") & tx["date"].between(c.start, c.end)]
        normal = tx[(tx["direction"] == "out") & ~tx["date"].between(c.start, c.end)]
        share_in = out.groupby("category")["amount"].sum() / out["amount"].sum()
        share_norm = normal.groupby("category")["amount"].sum() / normal["amount"].sum()
        jump = (share_in - share_norm.reindex(share_in.index).fillna(0)).max()
        if jump < 0.25:
            return False
    return True


def top_category(r: Results, c: Cluster) -> str:
    tx = r.transactions
    out = tx[(tx["direction"] == "out") & tx["date"].between(c.start, c.end)]
    normal = tx[(tx["direction"] == "out") & ~tx["date"].between(c.start, c.end)]
    share_in = out.groupby("category")["amount"].sum() / out["amount"].sum()
    share_norm = normal.groupby("category")["amount"].sum() / normal["amount"].sum()
    return (share_in - share_norm.reindex(share_in.index).fillna(0)).idxmax()


# --------------------------------------------------------------------------- #
# Chart 2 - cash flow
# --------------------------------------------------------------------------- #
def _payroll_weeks(r: Results) -> pd.Index:
    sched = r.daily["outflow_scheduled"]
    days = sched[sched > 50_000].index
    return pd.Index(sorted({d - pd.Timedelta(days=d.weekday()) for d in days}))


def cashflow_title(r: Results) -> str:
    final = r.daily["cumulative_net"].iloc[-1]
    verb = "grew" if final >= 0 else "fell"
    return f"Cash {verb} by {money_k(abs(final))} over the year, with dips each payroll week"


CASHFLOW_SUBTITLE = (f"{config.COMPANY_SHORT}. Top: cumulative net cash flow since 1 Jan. "
                     "Bottom: net cash flow per week (money in minus money out).")


def cashflow_insight(r: Results) -> str:
    w = r.weekly
    final = r.daily["cumulative_net"].iloc[-1]
    neg = w[w["net"] < 0]
    payroll = _payroll_weeks(r)
    other_neg = neg[~neg.index.isin(payroll)]
    parts = [f"Money in exceeded money out by {money_k(final)} across 2025. "
             f"The {len(neg[neg.index.isin(payroll)])} negative weeks that line up with month-end payroll "
             f"are expected and recover within days."]
    if len(other_neg):
        wk = other_neg.index[0]
        dr = [c for c in droughts(r) if c.start <= wk + pd.Timedelta(days=6) and c.end >= wk]
        why = f", caused by the {dr[0].label} inflow drought" if dr else ""
        parts.append(f"Only {len(other_neg)} other week went negative (week of {wk:%d %b}: "
                     f"{money_k(other_neg['net'].iloc[0])}){why}." if len(other_neg) == 1 else
                     f"{len(other_neg)} other weeks went negative.")
    s = spend(r)
    if s:
        normal_net = w.loc[~w.index.isin(payroll) & (w["net"] > 0), "net"].median()
        weeks = [w.loc[c.start - pd.Timedelta(days=c.start.weekday()), "net"] for c in s]
        cut = 1 - max(weeks) / normal_net
        below = any(x < 0 for x in weeks)
        parts.append(("The spending bursts pushed their weeks below zero." if below else
                      f"The spending bursts did not push any week below zero, but they cut those weeks' "
                      f"net inflow by at least {cut:.0%} compared with a normal week."))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Chart 3 - cluster vs normal
# --------------------------------------------------------------------------- #
COMPARE_TITLE = "Every flagged period sits outside the range of normal periods of the same length"
COMPARE_SUBTITLE = ("Grey: average per day in every other same-length stretch of 2025. "
                    "Orange: the flagged period and its 95% bootstrap interval.")


def compare_detail(c: Cluster) -> str:
    return (f"{c.xfold:.1f}x typical · 95% CI {money(c.ci_low)}-{money(c.ci_high)}"
            if c.kind == "spend_cluster" else
            f"{c.xfold:.0%} of typical · 95% CI {money(c.ci_low)}-{money(c.ci_high)}")


def compare_insight(r: Results) -> str:
    n_min = min(len(c.ref_means) for c in r.clusters) if r.clusters else 0
    outside = all((c.pct_below == 100) if c.kind == "spend_cluster" else (c.pct_below == 0) for c in r.clusters)
    middle = ("None of those ordinary stretches reached the flagged level, so these are not just a run "
              "of busy or quiet days." if outside else
              "A few ordinary stretches came close to some flagged periods, so treat those as borderline.")
    return (f"Each panel compares one flagged period with at least {n_min} ordinary stretches of the same "
            f"length from the same year. {middle} The percentages describe where a period sits among "
            "normal periods; they are not significance tests.")


# --------------------------------------------------------------------------- #
# Page helpers: summary findings and "worth checking" detail
# --------------------------------------------------------------------------- #
def worth_checking(r: Results, c: Cluster) -> pd.DataFrame:
    """Largest payees in the category that drove a spending burst."""
    tx = r.transactions
    cat = top_category(r, c)
    sel = tx[(tx["direction"] == "out") & (tx["category"] == cat) & tx["date"].between(c.start, c.end)]
    g = (sel.groupby("counterparty")["amount"].agg(["count", "sum"])
         .sort_values("sum", ascending=False).reset_index())
    g.columns = ["Paid to", "Payments", "Total"]
    return g


def drought_shortfall(r: Results, c: Cluster) -> tuple[float, float]:
    """(expected, actual) money in over the drought, using the weekday-adjusted baseline."""
    d = r.daily.loc[c.start:c.end]
    expected = float((d["in_baseline"] * d["in_dow_factor"]).sum())
    return expected, float(d["inflow"].sum())


def summary_points(r: Results) -> list[dict]:
    d = r.daily
    final = d["cumulative_net"].iloc[-1]
    pts = []
    s = spend(r)
    if s:
        parts = [f"{c.label} ({top_category(r, c).lower()}, {c.xfold:.1f}x normal)" for c in s]
        pts.append({
            "headline": f"{NUM_WORDS.get(len(s), str(len(s)))} short spending {'burst' if len(s) == 1 else 'bursts'} stood out",
            "detail": " and ".join(parts) + ". Each came from a single category, so it can be checked quickly.",
            "anchor": "velocity-section",
        })
    pts.append({
        "headline": f"Cash grew by {money_k(final)} over the year",
        "detail": "Every month dips at payroll and recovers within days. That is the normal rhythm, not a problem.",
        "anchor": "cashflow-section",
    })
    for c in droughts(r):
        exp, act = drought_shortfall(r, c)
        pts.append({
            "headline": f"Customer payments almost stopped for {c.n_days} days in {c.start:%B}",
            "detail": f"{c.label}: about {money_k(act)} arrived against roughly {money_k(exp)} expected. "
                      "Worth confirming whether invoices went out late or payments were held.",
            "anchor": "compare-section",
        })
    return pts
