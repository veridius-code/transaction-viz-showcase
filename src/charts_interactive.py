"""Interactive Plotly versions of the three charts, for the web page.

Each chart is exported as {data, layout, layoutMobile}: the page switches to
the mobile layout (fewer annotations, tighter margins) on narrow screens.
Titles live in the page's HTML, so the figures carry none.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from src import narrative as nv, style as st
from src.analysis import Cluster, Results

HALF_DAY = pd.Timedelta(hours=12)


def _year_range(d: pd.DataFrame) -> list:
    # Fixed range: otherwise Plotly widens the axis to fit annotation text.
    return [d.index[0] - HALF_DAY, d.index[-1] + HALF_DAY]


def _export(fig: go.Figure, mobile: dict) -> dict:
    out = json.loads(pio.to_json(fig, validate=True, pretty=False))
    layout_m = json.loads(json.dumps(out["layout"]))
    layout_m.update(json.loads(json.dumps(mobile, cls=PlotlyJSONEncoder)))
    out["layoutMobile"] = layout_m
    return out


def _annot(text, x, y, **kw) -> dict:
    a = dict(text=text, x=x, y=y, showarrow=False, font=dict(size=14, color=st.INK))
    a.update(kw)
    return a


# --------------------------------------------------------------------------- #
def velocity(r: Results) -> dict:
    d = r.daily
    in_cl = d["out_in_cluster"]
    fig = go.Figure()
    band = d[["out_band_low", "out_band_high"]].dropna()
    fig.add_trace(go.Scatter(x=band.index, y=band["out_band_low"], mode="lines", line=dict(width=0),
                             line_shape="hvh", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band.index, y=band["out_band_high"], mode="lines", line=dict(width=0),
                             line_shape="hvh", fill="tonexty", fillcolor="rgba(0,114,178,0.18)",
                             name="Typical range", customdata=band["out_band_low"],
                             hovertemplate="Typical range: %{customdata:$,.0f} - %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=d.index, y=d["outflow_operating"], name="Daily spend",
                         marker=dict(color=[st.ACCENT if f else st.GREY_BAR for f in in_cl], line=dict(width=0)),
                         hovertemplate="Spend: %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d.index, y=d["operating_7d"], mode="lines", name="7-day average",
                             line=dict(color=st.INK, width=2),
                             hovertemplate="7-day average: %{y:$,.0f}<extra></extra>"))
    ymax = d["outflow_operating"].max() * 1.25
    annotations = []
    for c in nv.spend(r):
        mid = c.start + (c.end - c.start) / 2
        peak = d.loc[c.start:c.end, "outflow_operating"].max()
        annotations.append(dict(
            text="<b>" + nv.callout(c, r).replace("\n", "<br>") + "</b>", x=mid, y=peak, ax=40, ay=-40 if peak > ymax * 0.7 else -80,
            showarrow=True, arrowhead=0, arrowwidth=1.4, arrowcolor=st.ACCENT, xanchor="left", align="left",
            font=dict(size=14, color=st.INK), bgcolor="rgba(255,255,255,0.85)"))
    lab = pd.Timestamp("2025-06-01")
    lab2 = pd.Timestamp("2025-10-25")
    labels = [
        _annot("<b>Typical range</b>", lab, band.loc[lab, "out_band_high"], yanchor="bottom",
               font=dict(size=14, color=st.SECONDARY), showarrow=True, ax=0, ay=-55,
               arrowcolor=st.SECONDARY, arrowwidth=1, arrowhead=0),
        _annot("<b>7-day average</b>", lab2, d.loc[lab2, "operating_7d"], yanchor="bottom",
               showarrow=True, ax=0, ay=-110, arrowcolor=st.INK, arrowwidth=1, arrowhead=0),
    ]
    fig.update_layout(**st.plotly_base_layout(
        bargap=0.15, annotations=annotations + labels,
        xaxis=st.plotly_axis(type="date", dtick="M1", tickformat="%b", hoverformat="%a %d %b %Y",
                             range=_year_range(d)),
        yaxis=st.plotly_yaxis(range=[0, ymax]),
    ))
    mid_year = pd.Timestamp("2025-07-01")
    mobile_ann = []
    for c, a in zip(nv.spend(r), annotations):
        left = a["x"] < mid_year
        mobile_ann.append(dict(a, text=f"<b>{c.label}: {c.xfold:.1f}x typical</b>", ax=10 if left else -10, ay=-30,
                               xanchor="left" if left else "right", font=dict(size=12, color=st.INK)))
    mobile = dict(annotations=mobile_ann,
                  margin=dict(l=50, r=10, t=20, b=40),
                  xaxis=dict(fig.layout.xaxis.to_plotly_json(), dtick="M2"))
    return _export(fig, mobile)


# --------------------------------------------------------------------------- #
def cashflow(r: Results) -> dict:
    d, w = r.daily, r.weekly
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=d.index, y=d["cumulative_net"], mode="lines", line=dict(color=st.INK, width=2),
                             hovertemplate="Cumulative net: %{y:$,.0f}<extra></extra>"), row=1, col=1)
    centres = w.index + pd.Timedelta(days=3)
    fig.add_trace(go.Bar(
        x=centres, y=w["net"], width=5.6 * 86_400_000,
        marker=dict(color=[st.SECONDARY if v >= 0 else st.ACCENT for v in w["net"]], line=dict(width=0)),
        customdata=list(zip(w.index.strftime("%d %b"), w["inflow"], w["outflow_total"])),
        hovertemplate=("Week of %{customdata[0]}<br>In: %{customdata[1]:$,.0f}<br>"
                       "Out: %{customdata[2]:$,.0f}<br><b>Net: %{y:$,.0f}</b><extra></extra>"),
    ), row=2, col=1)
    top = d["cumulative_net"].max()
    shapes, annotations = [], []
    for c in r.clusters:
        for yref in ("y domain", "y2 domain"):
            shapes.append(dict(type="rect", xref="x", yref=yref, x0=c.start - HALF_DAY, x1=c.end + HALF_DAY,
                               y0=0, y1=1, fillcolor=st.GREY_FILL, line=dict(width=0), layer="below"))
        annotations.append(_annot(f"{c.label}<br>{nv.kind_label(c)}", c.start + (c.end - c.start) / 2, 1.0,
                                  xref="x", yref="y domain", yanchor="top", font=dict(size=13, color=st.INK_2)))
    final = d["cumulative_net"].iloc[-1]
    annotations.append(_annot(f"<b>{nv.money_k(final)}</b>", d.index[-1], final, xref="x", yref="y",
                              xanchor="right", yanchor="top", xshift=-8, yshift=-4, font=dict(size=15, color=st.INK)))
    first_pos = w.index[w["net"] > 0][1] + pd.Timedelta(days=3)
    first_neg = w.index[w["net"] < 0][0] + pd.Timedelta(days=8)
    annotations += [
        _annot("<b>Money in > money out</b>", first_pos, 1.0, xref="x", yref="y2 domain", xanchor="left",
               yanchor="top", font=dict(size=13, color=st.SECONDARY)),
        _annot("<b>Payroll<br>weeks</b>", first_neg, w["net"].min() * 0.55, xref="x", yref="y2",
               xanchor="left", font=dict(size=13, color=st.ACCENT)),
    ]
    fig.update_layout(**st.plotly_base_layout(
        shapes=shapes, annotations=annotations, hovermode="x unified",
        xaxis=st.plotly_axis(type="date", dtick="M1", tickformat="%b", showticklabels=False,
                             hoverformat="%a %d %b %Y", range=_year_range(d)),
        xaxis2=st.plotly_axis(type="date", dtick="M1", tickformat="%b", hoverformat="%a %d %b %Y",
                              range=_year_range(d)),
        yaxis=st.plotly_yaxis(title=dict(text="Cumulative", font=dict(size=13)),
                              range=[min(d["cumulative_net"].min() * 1.3, -top * 0.08), top * 1.3]),
        yaxis2=st.plotly_yaxis(title=dict(text="Weekly net", font=dict(size=13)),
                               range=[w["net"].min() * 1.15, w["net"].max() * 1.45]),
    ))
    mobile_ann = []
    for i, c in enumerate(r.clusters):
        short = "burst" if c.kind == "spend_cluster" else "drought"
        mobile_ann.append(_annot(f"{c.start:%b} {short}", c.start + (c.end - c.start) / 2, 1.0 if i % 2 == 0 else 0.86,
                                 xref="x", yref="y domain", yanchor="top", font=dict(size=11, color=st.INK_2),
                                 bgcolor="rgba(255,255,255,0.8)"))
    mobile = dict(
        annotations=mobile_ann,
        margin=dict(l=48, r=8, t=16, b=36),
        xaxis2=dict(fig.layout.xaxis2.to_plotly_json(), dtick="M2"),
        yaxis=dict(fig.layout.yaxis.to_plotly_json(), tickformat="$~s", title=None),
        yaxis2=dict(fig.layout.yaxis2.to_plotly_json(), tickformat="$~s", title=None),
    )
    return _export(fig, mobile)


# --------------------------------------------------------------------------- #
def compare_one(c: Cluster) -> dict:
    ref = c.ref_means
    fig = go.Figure()
    counts, edges = np.histogram(ref, bins=24)
    centres = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centres, y=counts, width=(edges[1] - edges[0]) * 0.92,
        marker=dict(color=st.GREY_BAR, line=dict(width=0)),
        customdata=np.stack([edges[:-1], edges[1:]], axis=1),
        hovertemplate="%{customdata[0]:$,.0f}-%{customdata[1]:$,.0f} per day: %{y} normal periods<extra></extra>",
    ))
    ymax = counts.max() * 1.65
    fig.add_trace(go.Scatter(
        x=[c.mean_adj], y=[ymax * 0.68], mode="markers",
        marker=dict(color=st.ACCENT, size=11, symbol="line-ns-open", line=dict(width=0)),
        error_x=dict(type="data", symmetric=False, array=[c.ci_high - c.mean_adj],
                     arrayminus=[c.mean_adj - c.ci_low], color=st.ACCENT, thickness=2.5, width=8),
        hovertemplate=(f"<b>{c.label}</b><br>${c.mean_adj:,.0f} per day<br>"
                       f"95% CI ${c.ci_low:,.0f} - ${c.ci_high:,.0f}<extra></extra>"),
    ))
    lo_x = min(ref.min(), c.ci_low)
    hi_x = max(ref.max(), c.ci_high)
    pad = (hi_x - lo_x) * 0.1
    right_room = (hi_x - c.mean_adj) > (c.mean_adj - lo_x)
    shapes = [
        dict(type="line", x0=c.mean_adj, x1=c.mean_adj, y0=0, y1=ymax * 0.9, line=dict(color=st.ACCENT, width=2.5)),
        dict(type="line", x0=c.typical, x1=c.typical, y0=0, y1=ymax * 0.8, line=dict(color=st.INK_2, width=1)),
    ]
    annotations = [
        _annot(f"<b>{nv.money(c.mean_adj)}/day</b>", c.mean_adj, ymax * 0.9, xanchor="left" if right_room else "right",
               xshift=6 if right_room else -6, yanchor="top", font=dict(size=14, color=st.INK)),
        _annot(f"typical<br>{nv.money(c.typical)}", c.typical, ymax * 0.82, yanchor="bottom",
               font=dict(size=13, color=st.INK_2)),
    ]
    unit = "operating spend" if c.kind == "spend_cluster" else "inflow"
    fig.update_layout(**st.plotly_base_layout(
        shapes=shapes, annotations=annotations, hovermode="closest", bargap=0.05,
        margin=dict(l=20, r=20, t=10, b=60),
        xaxis=st.plotly_axis(range=[max(0, lo_x - pad) if c.kind == "spend_cluster" else 0, hi_x + pad],
                             tickformat="$~s", nticks=5,
                             title=dict(text=f"Average {unit} per day", font=dict(size=13))),
        yaxis=dict(visible=False, range=[0, ymax]),
    ))
    return _export(fig, dict(margin=dict(l=10, r=10, t=10, b=55)))


def build_all(r: Results) -> dict[str, dict]:
    figs = {"velocity": velocity(r), "cashflow": cashflow(r)}
    for i, c in enumerate(r.clusters, start=1):
        figs[f"compare{i}"] = compare_one(c)
    return figs
