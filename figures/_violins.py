"""The violins of the controls figure and of the opener's step 4: one per condition over the roles, each a Gaussian
kernel density with Scott's bandwidth cut at the lowest and highest role, every violin as wide at its widest, with the
mean as a dark line and a small number. Every control is the same dark grey."""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from scipy.stats import gaussian_kde

MAIN, SET_BACK = ["vector", "neurons", "rotation_matched"], ["components", "random"]
HEAD = {"vector": "the persona\nvector", "neurons": "top 100\nneurons", "rotation_matched": "100 randomly\nrotated directions",
        "components": "top 100 principal\ncomponents", "random": "100 random\nneurons"}
POS_FIVE, POS_THREE = [0, 1.1, 2.2, 3.3, 4.4], [0, 1, 2]
AXIS = "Replies with the persona's traits (%)"


def draw(fk, d, columns: int, width: float, figsize: tuple[float, float], title: str | None = None, theme: dict | None = None):
    """The figure for the per-role table d (role, kind, in_role), with its title unless title is None. theme overrides
    the colours for a dark page: fill (by kind), line and halo."""
    kinds = MAIN + (SET_BACK if columns == 5 else [])
    pos = dict(zip(kinds, POS_FIVE if columns == 5 else POS_THREE))
    fill = {"vector": fk.color("the role's steering vector"), "neurons": fk.color("top 100 neurons"),
            "rotation_matched": fk.color("randomly rotated directions"), "components": fk.color("principal components"),
            "random": fk.color("random neurons")}
    theme = theme or {}
    fill.update(theme.get("fill", {}))
    line, halo = theme.get("line", fk.INK), theme.get("halo", "white")
    head = dict(HEAD)
    if columns == 3:
        head = {k: t.replace("\n", " ") for k, t in head.items()}
    vals = {k: d[d.kind == k].in_role.to_numpy(float) for k in kinds}
    kde = {k: gaussian_kde(v) for k, v in vals.items()}
    grid = {k: np.linspace(v.min(), v.max(), 400) for k, v in vals.items()}
    scale = {k: width / 2 / kde[k](grid[k]).max() for k in kinds}

    fig, ax = plt.subplots(figsize=figsize)
    top = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    at = lambda pt: mtransforms.offset_copy(top, fig=fig, y=pt, units="points")
    for k in kinds:
        x, v = pos[k], vals[k]
        hw = kde[k](grid[k]) * scale[k]
        ax.fill_betweenx(grid[k], x - hw, x + hw, color=fill[k], lw=0, zorder=2)
        m = v.mean()
        h = float(kde[k](m)[0]) * scale[k]
        ax.plot([x - h - 0.07, x + h + 0.07], [m, m], color=line, lw=1.3, solid_capstyle="butt", zorder=4)
        ax.text(x + h + 0.11, m, f"{m:.0f}%", ha="left", va="center", color=fill[k], fontweight="semibold", fontsize=12,
                zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground=halo)])
        ax.text(x, 1, head[k], transform=at(8), ha="center", va="bottom", fontweight="semibold", fontsize=11,
                linespacing=1.05, color=fill[k])
    ax.set(xticks=[], xlabel="", xlim=(-0.55, max(pos.values()) + 0.6), ylim=(-3, 103), yticks=[0, 25, 50, 75, 100],
           ylabel=AXIS)
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["bottom"].set_position(("data", 0))   # the baseline at 0, where the violins end
    if title is not None:
        fk.title(fig, title)
    return fig
