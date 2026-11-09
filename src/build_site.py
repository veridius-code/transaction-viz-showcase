"""Build the static site in docs/ (works from file:// and on GitHub Pages)."""
from __future__ import annotations

import html
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import plotly

from src import charts_interactive, config, narrative as nv
from src.analysis import Results, run_pipeline

TEMPLATE = Path(__file__).with_name("site_template.html")
ASSETS = config.DOCS_DIR / "assets"
DOWNLOADS = config.DOCS_DIR / "downloads"
NBVIEWER_URL = (f"https://nbviewer.org/github/{config.GITHUB_USER}/{config.REPO_NAME}"
                "/blob/main/notebook/analysis.ipynb")


def _table(df: pd.DataFrame, numeric: set[str], cls: str = "") -> str:
    head = "".join(f'<th class="n">{html.escape(c)}</th>' if c in numeric else f"<th>{html.escape(c)}</th>"
                   for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in numeric:
                txt = f"{v:,.2f}x" if isinstance(v, float) and c.startswith("x") else (
                    f"{v:,.0f}" if isinstance(v, (int, float)) else str(v))
                cells.append(f'<td class="n">{html.escape(txt)}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    attr = f' class="{cls}"' if cls else ""
    return f"<table{attr}><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _cluster_table(r: Results) -> str:
    df = pd.DataFrame([{
        "Period": c.label,
        "Type": nv.kind_label(c).capitalize(),
        "Days": str(c.n_days),
        "Total": nv.money(c.total),
        "Per day (adj.)": nv.money(c.mean_adj),
        "Typical per day": nv.money(c.typical),
        "vs typical": f"{c.xfold:.2f}x",
        "95% CI per day": f"{nv.money(c.ci_low)} - {nv.money(c.ci_high)}",
        "Among normal periods": c.plain_percentile,
    } for c in r.clusters])
    return _table(df, {"Days", "Total", "Per day (adj.)", "Typical per day", "vs typical", "95% CI per day"}, "wide")


def _findings(r: Results) -> str:
    out = []
    for p in nv.summary_points(r):
        out.append(f"""      <div class="card">
        <div class="k">Finding</div>
        <h3>{html.escape(p['headline'])}</h3>
        <p>{html.escape(p['detail'])}</p>
        <a href="#{p['anchor']}">See the chart &darr;</a>
      </div>""")
    return "\n".join(out)


def _spend_checks(r: Results) -> str:
    out = []
    for c in nv.spend(r):
        cat = nv.top_category(r, c)
        payees = nv.worth_checking(r, c).head(3)
        df = pd.DataFrame({"Paid to": payees["Paid to"], "Payments": payees["Payments"].astype(str),
                           "Total": payees["Total"].map(nv.money)})
        out.append(f"      <h4>{html.escape(c.label)} · {html.escape(cat)}</h4>\n      "
                   + _table(df, {"Payments", "Total"}))
    if not out:
        return "      <p>Nothing stood out this year.</p>"
    out.append("      <p>Largest payees in the category behind each burst. Matching these against purchase orders "
               "is usually enough to explain them.</p>")
    return "\n".join(out)


def _cash_check(r: Results) -> str:
    out = []
    tx = r.transactions
    for c in nv.droughts(r):
        exp, act = nv.drought_shortfall(r, c)
        before = tx[(tx["direction"] == "in") & (tx["category"] == "Customer receipts")
                    & tx["date"].between(c.start - pd.Timedelta(days=28), c.start - pd.Timedelta(days=1))]
        during = tx[(tx["direction"] == "in") & tx["date"].between(c.start, c.end)]
        regular = before.groupby("counterparty")["amount"].agg(["count", "sum"])
        regular = regular[(regular["count"] >= 3) & ~regular.index.isin(during["counterparty"])]
        regular = regular.sort_values("sum", ascending=False).head(4)
        out.append(f"      <h4>{html.escape(c.label)} · customer payments</h4>")
        out.append(f"      <p>About {nv.money_k(act)} arrived against roughly {nv.money_k(exp)} expected, "
                   f"a shortfall of about {nv.money_k(exp - act)}.</p>")
        if len(regular):
            df = pd.DataFrame({"Usually pays, nothing received": regular.index,
                               "Prior 4 weeks": regular["sum"].map(nv.money).to_numpy()})
            out.append("      " + _table(df, {"Prior 4 weeks"}))
        out.append("      <p>Check whether invoices went out on time and whether these customers have since paid.</p>")
    return "\n".join(out) if out else "      <p>No unusual dips in money coming in.</p>"


def _quality_summary(r: Results) -> str:
    log = r.log.to_frame().set_index("step")["rows_affected"]
    return (f"The raw export had {log['Load']:,} rows. {log['Remove duplicates']:,} were duplicates and were removed. "
            f"{log['Normalise dates']:,} dates in non-standard formats, {log['Parse currency text']:,} amounts stored "
            f"as text and {log['Fill missing categories']:,} missing categories were fixed. Nothing else was dropped, "
            f"and the cleaned data matches the generator's known answer exactly.")


def _panels(r: Results) -> str:
    out = []
    for i, c in enumerate(r.clusters, start=1):
        out.append(f"""    <div class="panel">
      <h3>{html.escape(c.label)} · {html.escape(nv.kind_label(c))}</h3>
      <p class="pct">{html.escape(c.plain_percentile)}</p>
      <p class="det">{html.escape(nv.compare_detail(c))}</p>
      <div class="plot" id="compare{i}" role="img" aria-label="{html.escape(c.label)} compared with normal periods"></div>
    </div>""")
    return "\n".join(out)


def build(r: Results | None = None) -> Path:
    r = r or run_pipeline()
    ASSETS.joinpath("charts").mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    # Plotly.js, bundled locally so the page works offline and from file://
    from plotly.offline import get_plotlyjs_version
    pver = get_plotlyjs_version()
    for old in ASSETS.glob("plotly-*.min.js"):
        if old.name != f"plotly-{pver}.min.js":
            old.unlink()
    src_js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    shutil.copyfile(src_js, ASSETS / f"plotly-{pver}.min.js")

    figs = charts_interactive.build_all(r)
    (ASSETS / "charts" / "charts.js").write_text(
        "// Generated by src/build_site.py - synthetic data\nwindow.CHARTS = " + json.dumps(figs, separators=(",", ":")) + ";\n")

    for f in sorted(config.OUT_PNG.glob("*.png")) + sorted(config.OUT_PDF.glob("*.pdf")):
        shutil.copyfile(f, DOWNLOADS / f.name)

    rob = json.loads((config.DATA_DIR / "robustness.json").read_text())
    d = r.daily
    raw_rows = int(r.log.to_frame().set_index("step").loc["Load", "rows_affected"])
    typical = d.loc[~d["out_in_cluster"], "outflow_operating"].median()

    log = r.log.to_frame().rename(columns={"step": "Step", "detail": "What was fixed", "rows_affected": "Rows"})

    values = {
        "AUTHOR": config.AUTHOR_LINE,
        "COMPANY": config.COMPANY,
        "COMPANY_SHORT": config.COMPANY_SHORT,
        "MONOGRAM": "".join(w[0] for w in config.COMPANY_SHORT.split())[:2].upper(),
        "DISCLAIMER": config.DISCLAIMER,
        "PERIOD": config.PERIOD_LABEL.replace("-", "–"),
        "FINDINGS": _findings(r),
        "CHECKS": _spend_checks(r),
        "CASH_CHECK": _cash_check(r),
        "QUALITY_SUMMARY": _quality_summary(r),
        "REPO_URL": config.REPO_URL,
        "NBVIEWER_URL": NBVIEWER_URL,
        "GH_USER": config.GITHUB_USER,
        "SEED": str(config.SEED),
        "N_TX": f"{len(r.transactions):,}",
        "N_RAW": f"{raw_rows:,}",
        "TYPICAL_SPEND": nv.money_k(typical),
        "NET": nv.money_k(d["cumulative_net"].iloc[-1]),
        "N_FLAGGED": str(len(r.clusters)),
        "VELOCITY_TITLE": nv.velocity_title(r),
        "VELOCITY_SUB": nv.VELOCITY_SUBTITLE_WEB,
        "VELOCITY_INSIGHT": nv.velocity_insight(r),
        "CASHFLOW_TITLE": nv.cashflow_title(r),
        "CASHFLOW_SUB": nv.CASHFLOW_SUBTITLE,
        "CASHFLOW_INSIGHT": nv.cashflow_insight(r),
        "COMPARE_TITLE": nv.COMPARE_TITLE,
        "COMPARE_SUB": nv.COMPARE_SUBTITLE,
        "COMPARE_INSIGHT": nv.compare_insight(r),
        "PANELS": _panels(r),
        "CLUSTER_TABLE": _cluster_table(r),
        "CLEANING_TABLE": _table(log, {"Rows"}),
        "WINDOW": str(r.cfg.window),
        "SPREAD_WINDOW": str(r.cfg.spread_window),
        "MIN_RUN": str(r.cfg.min_run),
        "Z": f"{r.cfg.z_threshold:g}",
        "N_BOOT": f"{r.cfg.n_boot:,}",
        "ROB_SEEDS": str(rob["n_seeds"]),
        "ROB_TOTAL": str(rob["planted_total"]),
        "ROB_FOUND": str(rob["found"]),
        "ROB_EXACT": str(rob["found_within_1_day"]),
        "ROB_FALSE": str(rob["false_clusters"]),
        "PLOTLY_VERSION": pver,
        "BUILD_DATE": date.today().strftime("%d %B %Y"),
    }
    page = TEMPLATE.read_text()
    escape_free = {"PANELS", "CLUSTER_TABLE", "CLEANING_TABLE", "FINDINGS", "CHECKS", "CASH_CHECK"}
    for k, v in values.items():
        page = page.replace("{{" + k + "}}", v if k in escape_free else html.escape(v, quote=True))
    leftover = page.split("{{")[1:]
    if leftover:
        raise ValueError(f"Unfilled template keys: {[t.split('}}')[0] for t in leftover]}")
    out = config.DOCS_DIR / "index.html"
    out.write_text(page)
    (config.DOCS_DIR / ".nojekyll").write_text("")
    return out


if __name__ == "__main__":
    print(build().relative_to(config.ROOT))
