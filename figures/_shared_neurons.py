"""Shared by fig6a_shared_roles.py and fig6b_sommelier_neurons.py: the axes placed by hand in inches from the top, so
that both images share one left edge, and the dots: every role grey, the other named roles dark grey, the sommelier
orange."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import figures

fk = figures.kit("post")
plt.rcParams["figure.constrained_layout.use"] = False   # the axes are placed by hand
W, LEFT, RIGHT, PAD = 6.0, 0.92, 0.08, 0.06   # width, axes margins and the title's indent, inches (1200 px wide)
HALO = [patheffects.withStroke(linewidth=3, foreground=plt.rcParams["figure.facecolor"])]
NAMES = {"assistant": figures.wording("assistant role", "assistant persona")}   # names written next to the dots, where they differ from the role's


def figure(height, title):
    fig = plt.figure(figsize=(W, height))
    fk.title(fig, title).set_position((PAD / W, 1 - 0.10 / height))
    return fig


def axes(fig, top, height, **kw):
    """Axes across the plotting width, their top edge `top` inches below the figure's."""
    h = fig.get_size_inches()[1]
    return fig.add_axes([LEFT / W, 1 - (top + height) / h, (W - LEFT - RIGHT) / W, height / h], **kw)


def text(fig, x, y, s, **kw):
    """Text x inches from the figure's left edge and y inches below its top."""
    return fig.text(x / W, 1 - y / fig.get_size_inches()[1], s, **kw)


def draw_roles(ax, d, x, y, size, named, somm):
    """One dot per role: grey, the named roles dark grey with their names, the sommelier orange and largest. Names
    are offset from their dots by (dx, dy) points."""
    rest = d[~d.role.isin([*named, "sommelier"])]
    ax.scatter(rest[x], rest[y], s=size, color=fk.GREY, alpha=0.65, lw=0)
    for role, (dx, dy), color, s, weight in [*[(r, o, fk.MUTED, 24, "normal") for r, o in named.items()],
                                              ("sommelier", somm, fk.ACCENT, 60, "semibold")]:
        r = d[d.role == role].iloc[0]
        ax.scatter(r[x], r[y], s=s, color=color, lw=0, zorder=3)
        ax.annotate(NAMES.get(role, role), (r[x], r[y]), xytext=(dx, dy), textcoords="offset points", color=color,
                    va="center", ha="right" if dx < 0 else "left", fontweight=weight, path_effects=HALO, zorder=4)
