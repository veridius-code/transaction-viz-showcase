"""Static, report-ready charts (PNG at 300 dpi + vector PDF, 13.33 x 7.5 in)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, narrative as nv, style as st  # noqa: E402
from src.analysis import Results, run_pipeline  # noqa: E402

MONTHS = mdates.MonthLocator()
MONTH_FMT = mdates.DateFormatter("%b")
SOURCE_NOTE = f"Source: {config.COMPANY} ledger (fictional, generated with seed {config.SEED})"


def _month_axis(ax) -> None:
    ax.xaxis.set_major_locator(MONTHS)
    ax.xaxis.set_major_formatter(MONTH_FMT)
    ax.set_xlim(pd.Timestamp(config.START_DATE) - pd.Timedelta(days=1),
                pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1))


def _kfmt(x, _pos=None):
    if not x:
        return "$0"
    sign = "-" if x < 0 else ""
    v = abs(x)
    return f"{sign}${v / 1e6:,.1f}m" if v >= 1e6 else f"{sign}${v / 1e3:,.0f}k"


# --------------------------------------------------------------------------- #
def chart_velocity(r: Results):
    d = r.daily
    fig = plt.figure(figsize=st.FIGSIZE)
    st.add_title(fig, nv.velocity_title(r), nv.VELOCITY_SUBTITLE)
    ax = fig.add_axes([0.075, 0.12, 0.895, 0.68])

    in_cl = d["out_in_cluster"]
    ax.bar(d.index, d["outflow_operating"], width=0.8, color=np.where(in_cl, st.ACCENT, st.GREY_BAR),
           linewidth=0, zorder=2)
    band = d[["out_band_low", "out_band_high"]].dropna()
    ax.fill_between(band.index, band["out_band_low"], band["out_band_high"], color=st.SECONDARY_FILL,
                    alpha=0.75, linewidth=0, zorder=1, step="mid")
    ax.plot(d.index, d["operating_7d"], color=st.INK, linewidth=2.0, zorder=3)

    ymax = d["outflow_operating"].max() * 1.22
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(_kfmt)
    _month_axis(ax)

    # Direct labels for the two reference marks
    lab_day = pd.Timestamp("2025-06-01")
    ax.annotate("Typical range", xy=(lab_day, band.loc[lab_day, "out_band_high"]),
                xytext=(lab_day, ymax * 0.58), color=st.SECONDARY, fontsize=14, fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="-", color=st.SECONDARY, lw=1))
    lab2 = pd.Timestamp("2025-10-25")
    ax.annotate("7-day average", xy=(lab2, d.loc[lab2, "operating_7d"]), xytext=(lab2, ymax * 0.58),
                color=st.INK, fontsize=14, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=st.INK, lw=1))

    for c in nv.spend(r):
        peak = d.loc[c.start:c.end, "outflow_operating"].max()
        mid = c.start + (c.end - c.start) / 2
        ax.annotate(nv.callout(c, r), xy=(mid, peak), xytext=(mid + pd.Timedelta(days=9), ymax * 0.93),
                    fontsize=15, fontweight="bold", color=st.INK, ha="left", va="center", linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", color=st.ACCENT, lw=1.4,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    st.add_footer(fig, SOURCE_NOTE)
    return fig


# --------------------------------------------------------------------------- #
def chart_cashflow(r: Results):
    d, w = r.daily, r.weekly
    fig = plt.figure(figsize=st.FIGSIZE)
    st.add_title(fig, nv.cashflow_title(r), nv.CASHFLOW_SUBTITLE)
    ax1 = fig.add_axes([0.085, 0.47, 0.885, 0.33])
    ax2 = fig.add_axes([0.085, 0.12, 0.885, 0.30], sharex=ax1)

    for c in r.clusters:
        for ax in (ax1, ax2):
            ax.axvspan(c.start - pd.Timedelta(hours=12), c.end + pd.Timedelta(hours=12),
                       color=st.GREY_FILL, linewidth=0, zorder=0)

    ax1.plot(d.index, d["cumulative_net"], color=st.INK, linewidth=2.0, zorder=3)
    ax1.axhline(0, color="#9E9E9E", linewidth=0.8, zorder=1)
    ax1.yaxis.set_major_formatter(_kfmt)
    top = d["cumulative_net"].max()
    ax1.set_ylim(min(d["cumulative_net"].min() * 1.3, -top * 0.08), top * 1.28)
    end = d.index[-1]
    ax1.annotate(f"{nv.money_k(d['cumulative_net'].iloc[-1])}", xy=(end, d["cumulative_net"].iloc[-1]),
                 xytext=(-14, -6), textcoords="offset points", ha="right", va="top", fontsize=15,
                 fontweight="bold",
                 color=st.INK)
    for c in r.clusters:
        mid = c.start + (c.end - c.start) / 2
        ax1.text(mid, top * 1.22, f"{c.label}\n{nv.kind_label(c)}", ha="center", va="top", fontsize=14,
                 color=st.INK_2, linespacing=1.15)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Cumulative", fontsize=14)

    centres = w.index + pd.Timedelta(days=3)
    colors = np.where(w["net"] >= 0, st.SECONDARY, st.ACCENT)
    ax2.bar(centres, w["net"], width=5.6, color=colors, linewidth=0, zorder=2)
    ax2.axhline(0, color="#9E9E9E", linewidth=0.8, zorder=3)
    ax2.yaxis.set_major_formatter(_kfmt)
    ax2.set_ylabel("Weekly net", fontsize=14)
    lo, hi = w["net"].min(), w["net"].max()
    ax2.set_ylim(lo * 1.15, hi * 1.45)
    first_pos = w.index[w["net"] > 0][1] + pd.Timedelta(days=3)
    ax2.text(first_pos, hi * 1.12, "Money in > money out", color=st.SECONDARY, fontsize=13.5,
             fontweight="bold", ha="left", va="bottom")
    first_neg = w.index[w["net"] < 0][0] + pd.Timedelta(days=3)
    ax2.text(first_neg + pd.Timedelta(days=5), lo * 0.55, "Payroll\nweeks", color=st.ACCENT, fontsize=14,
             fontweight="bold", ha="left", va="center", linespacing=1.1)
    _month_axis(ax1)
    st.add_footer(fig, SOURCE_NOTE)
    return fig


# --------------------------------------------------------------------------- #
def chart_compare(r: Results):
    clusters = r.clusters
    fig = plt.figure(figsize=st.FIGSIZE)
    st.add_title(fig, nv.COMPARE_TITLE, nv.COMPARE_SUBTITLE)
    n = len(clusters)
    left, gap, width = 0.06, 0.035, (0.91 - 0.035 * (n - 1)) / max(n, 1)
    for i, c in enumerate(clusters):
        ax = fig.add_axes([left + i * (width + gap), 0.15, width, 0.53])
        ref = c.ref_means
        lo_x = min(ref.min(), c.ci_low)
        hi_x = max(ref.max(), c.ci_high)
        pad = (hi_x - lo_x) * 0.08
        bins = np.linspace(ref.min(), ref.max(), 22)
        counts, _, _ = ax.hist(ref, bins=bins, color=st.GREY_BAR, edgecolor="white", linewidth=1.2, zorder=2)
        ymax = counts.max() * 1.5
        ax.set_ylim(0, ymax)
        ax.set_xlim(max(0, lo_x - pad), hi_x + pad)
        ax.axvline(c.mean_adj, color=st.ACCENT, linewidth=2.5, zorder=3)
        ax.plot([c.ci_low, c.ci_high], [ymax * 0.70] * 2, color=st.ACCENT, linewidth=2.5,
                solid_capstyle="butt", zorder=4)
        for xv in (c.ci_low, c.ci_high):
            ax.plot([xv, xv], [ymax * 0.67, ymax * 0.73], color=st.ACCENT, linewidth=2.0, zorder=4)
        ax.plot([c.typical, c.typical], [0, ymax * 0.84], color=st.INK_2, linewidth=1.0, zorder=3)

        # value label beside the orange line, on the side with room
        right_room = (hi_x + pad - c.mean_adj) > (c.mean_adj - max(0, lo_x - pad))
        ax.text(c.mean_adj, ymax * 0.80, f"  {nv.money(c.mean_adj)}/day  ",
                ha="left" if right_room else "right", va="bottom", fontsize=14, fontweight="bold",
                color=st.INK)
        tx = c.typical
        ax.text(tx, ymax * 0.97, f"typical\n{nv.money(tx)}", ha="center", va="top", fontsize=14,
                color=st.INK_2, linespacing=1.1)

        ax.set_title(f"{c.label}  ·  {nv.kind_label(c)}", fontsize=17, pad=58, loc="left")
        ax.text(0, 1.115, c.plain_percentile, transform=ax.transAxes, fontsize=14.5, color=st.INK,
                ha="left", va="bottom", fontweight="bold")
        ax.text(0, 1.03, nv.compare_detail(c), transform=ax.transAxes, fontsize=13.5, color=st.INK_2,
                ha="left", va="bottom")
        ax.xaxis.set_major_formatter(_kfmt)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        unit = "operating spend" if c.kind == "spend_cluster" else "inflow"
        ax.set_xlabel(f"Average {unit} per day", fontsize=13.5)
    st.add_footer(fig, "Days adjusted for day-of-week; percentiles are descriptive, not p-values")
    return fig


def build_all(r: Results | None = None) -> dict[str, tuple]:
    st.apply_mpl_style()
    r = r or run_pipeline()
    out = {}
    for name, fn in (("1_spending_velocity", chart_velocity),
                     ("2_cash_flow", chart_cashflow),
                     ("3_cluster_vs_normal", chart_compare)):
        out[name] = st.save(fn(r), name, config.OUT_PNG, config.OUT_PDF)
    return out


if __name__ == "__main__":
    for name, (png, pdf) in build_all().items():
        print(name, png.relative_to(config.ROOT), pdf.relative_to(config.ROOT))
