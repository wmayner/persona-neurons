"""House style and colour roles for matplotlib figures. Fonts are read from the fonts/ folder next to this file."""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

INK = "#1F1F1F"
MUTED = "#6B6B6B"
GREY = "#A3A3A3"
LIGHT = "#D9D9D9"
ACCENT = "#D55E00"
BLUE = "#0072B2"
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
ROLE_COLORS = {"control": GREY, "intervention": ACCENT, "second": BLUE}

# Each size's default figure size in inches and its font sizes
SIZES = {
    "screen": {"figsize": (7, 4), "font": 13, "ticks": 12, "title": 14},
    "post": {"figsize": (6, 4.8), "font": 13, "ticks": 12, "title": 15},   # 1200 x 960 px at 200 dpi
}

_FONTS = Path(__file__).resolve().with_name("fonts")
_size = "screen"
_roles = {}    # label -> colour

STYLE = {
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "legend.frameon": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#ECECEC",
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "errorbar.capsize": 0,
    "patch.edgecolor": "none",
}


def use(size="screen", roles=None):
    """Apply the house style for a size. roles: {role: label or labels}, naming each condition's colour role once."""
    global _size
    _size = size
    s = SIZES[size]
    k = s["font"] / 13  # lines and markers scale with the text
    for f in sorted(_FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update(STYLE)
    mpl.rcParams.update({
        "figure.figsize": s["figsize"], "savefig.dpi": 200,
        "font.size": s["font"], "axes.labelsize": s["font"], "axes.titlesize": s["font"],
        "xtick.labelsize": s["ticks"], "ytick.labelsize": s["ticks"], "legend.fontsize": s["ticks"],
        "lines.linewidth": 2.2 * k, "lines.markersize": 6 * k, "axes.linewidth": 0.9 * k, "grid.linewidth": 0.8 * k,
    })
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Inter", "DejaVu Sans"]
    _roles.clear()
    for role, labels in (roles or {}).items():
        labels = [labels] if isinstance(labels, str) else labels
        _roles.update({_key(lab): ROLE_COLORS[role] for lab in labels})


def title(fig, text):
    """The figure's title, left-aligned."""
    w = fig.get_size_inches()[0]
    pad = mpl.rcParams["figure.constrained_layout.w_pad"]
    return fig.suptitle(text, x=pad / w, ha="left", fontsize=SIZES[_size]["title"], fontweight="semibold")


def _key(label):
    return " ".join(str(label).split())  # a label wrapped onto two lines is still the same condition


def color(label, default=GREY):
    """The colour of a condition's role, or default without one."""
    return _roles.get(_key(label), default)


def save(fig, path):
    """Save the figure. It is drawn once first, since constrained layout settles over successive draws."""
    fig.canvas.draw()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
