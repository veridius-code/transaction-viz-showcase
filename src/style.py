"""One visual system shared by the static and interactive charts."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Grey neutrals + one accent (Okabe-Ito vermillion) + one secondary (Okabe-Ito
# blue). Both are colourblind-safe and clear 3:1 contrast on white.
INK = "#1F1F1F"          # titles, key lines
INK_2 = "#4D4D4D"        # body text, axis labels
INK_3 = "#767676"        # footers, minor text (4.5:1 on white)
GREY_BAR = "#C9C9C9"     # neutral data bars
GREY_FILL = "#E6E6E6"    # neutral shading
GRID = "#E9E9E9"
ACCENT = "#D55E00"       # unusual / needs attention
SECONDARY = "#0072B2"    # normal / positive
SECONDARY_FILL = "#CFE3F1"
ACCENT_FILL = "#F8DCC8"

FOOTER = "Synthetic data - illustrative sample"
FIGSIZE = (13.33, 7.5)
# Arial locally and Liberation Sans in CI have identical metrics, so layouts
# match wherever the charts are rendered.
FONT = ["Arial", "Liberation Sans"]


def apply_mpl_style() -> None:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    if not available.intersection(FONT):
        raise RuntimeError(f"None of {FONT} is installed; text would not fit the layout. "
                           "Install Arial or Liberation Sans (apt: fonts-liberation).")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelsize": 15,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": "#BDBDBD",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.parse_math": False,    # "$" is a currency sign here, not maths
        "pdf.fonttype": 42,          # editable text in the PDF
        "svg.fonttype": "none",
    })


def money(x: float, _pos=None) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


MONEY_FMT = FuncFormatter(money)


def add_title(fig, title: str, subtitle: str) -> None:
    fig.text(0.045, 0.955, title, fontsize=23, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.045, 0.895, subtitle, fontsize=15.5, color=INK_2, ha="left", va="top")


def add_footer(fig, note: str = "") -> None:
    fig.text(0.045, 0.025, FOOTER, fontsize=13, color=INK_3, ha="left", va="bottom", style="italic")
    if note:
        fig.text(0.955, 0.025, note, fontsize=13, color=INK_3, ha="right", va="bottom")


def save(fig, name: str, png_dir, pdf_dir) -> tuple:
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = png_dir / f"{name}.png", pdf_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# Plotly layout shared by every interactive chart
PLOTLY_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


def plotly_base_layout(**overrides) -> dict:
    layout = dict(
        template="none",
        font=dict(family=PLOTLY_FONT, size=14, color=INK_2),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=30, b=60),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font=dict(family=PLOTLY_FONT, size=13, color=INK)),
        separators=".,",
    )
    layout.update(overrides)
    return layout


def plotly_axis(**overrides) -> dict:
    ax = dict(showgrid=False, zeroline=False, linecolor="#BDBDBD", ticks="", tickfont=dict(size=13))
    ax.update(overrides)
    return ax


def plotly_yaxis(**overrides) -> dict:
    ax = dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
              tickformat="$,.0f", tickfont=dict(size=13))
    ax.update(overrides)
    return ax
