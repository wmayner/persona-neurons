"""Set size: the share of replies with the persona's traits at 4× strength when Qwen3-8B is steered with a role's top k
neurons, or with three controls at the same k (k randomly rotated directions, the top k principal components, k random
neurons from the same layers; draw 0 at every size, except that the rotated directions and random neurons at k = 100 are
the mean over each role's five draws), against the persona vector (dashed). Lines are means over 275 roles with 95%
bootstrap intervals, each named at its right end.

Reads role_steering/, role_steering_sizes/ and role_steering_control_sizes/ and writes figures/fig2_set_size/ in the data
directory (figure.png, figure.pdf, data.parquet, five_draws_at_100.parquet with each role's mean over the five draws).

    uv run figures/fig2_set_size.py
"""
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import figures
from persona_neurons.thread_tables import DRAWN, SET_SIZES, five_draw_per_role, set_size_with_five_draws

TITLE = "Small sets of neurons are sufficient to reproduce\nthe persona vector steering"
AXIS = "Replies with the persona's traits (%)"
XMAX = 6000

fk = figures.kit("post")
WHITE = [pe.withStroke(linewidth=3, foreground=plt.rcParams["figure.facecolor"])]   # a halo in the background's colour
# the controls share one grey, told apart by line style and marker
SERIES = [("neurons", "top\nneurons", fk.color("top neurons"), 6, None, "-", "o"),
          ("rotation_matched", "randomly\nrotated\ndirections", figures.CONTROL, 4.5, 1.6, "-", "o"),
          ("components", "principal\ncomponents", figures.CONTROL, 4.5, 1.6, (0, (4, 2)), "s"),
          ("random", "random\nneurons", figures.CONTROL, 5, 1.6, (0, (1, 1.5)), "^")]
s = set_size_with_five_draws()
v = s[s.kind == "vector"].iloc[0]

fig, ax = plt.subplots()
ax.axhline(v["mean"], color=fk.color("the role's steering vector"), lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.text(0.85, v["mean"] + 1.2, f"the persona vector  {v['mean']:.0f}%", color=fk.color("the role's steering vector"),
        va="bottom", fontsize=11, path_effects=WHITE)
plotted, ends = [v.to_frame().T], []
for z, (name, label, c, ms, lw, ls, marker) in zip((5, 4, 3, 2), SERIES):
    t = s[(s.kind == name) & s.k.isin(SET_SIZES)].sort_values("k")
    assert t.k.tolist() == SET_SIZES, name
    ax.errorbar(t.k, t["mean"], yerr=[t["mean"] - t.lo, t.hi - t["mean"]], color=c, marker=marker, ms=ms, lw=lw, ls=ls,
                elinewidth=1.4 if name == "neurons" else 1.0, zorder=z)
    ends.append((t["mean"].iloc[-1], label, c))
    plotted.append(t)
# the top line's values, each below its point and clear of the error bars (to the right, the last to the left; 100's lower, to clear 200's bar)
t = s[(s.kind == "neurons") & s.k.isin(SET_SIZES)].set_index("k")["mean"]
for k, dx, dy, ha in ((1, 5, -13, "left"), (10, 5, -13, "left"), (50, 5, -13, "left"), (100, 5, -16, "left"), (1000, -7, -13, "right")):
    ax.annotate(f"{t[k]:.0f}%", (k, t[k]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                color=SERIES[0][2], fontsize=11, fontweight="semibold", path_effects=WHITE)
ax.set(xscale="log", xlim=(0.75, XMAX), ylim=(0, 80), yticks=[0, 20, 40, 60, 80], ylabel=AXIS,
       xlabel="Neurons or directions steered (of ~230k up to layer 18)", xticks=SET_SIZES, xticklabels=[f"{k:,}" for k in SET_SIZES])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.spines["bottom"].set_bounds(0.75, 1000)
fk.title(fig, TITLE)
# Names at the right ends. The controls' ends lie within 7 points of each other, so their names are stacked from the
# bottom up, with a leader where a name moved; measured once the title has set the axes' final height.
fig.canvas.draw()
pt_per_unit = (ax.transData.transform((1, 1))[1] - ax.transData.transform((1, 0))[1]) * 72 / fig.dpi
placed, floor = [], 0
for y, label, c in sorted(ends[1:]):
    half = ((label.count("\n") + 1) * 11 * 1.25 + 4) / pt_per_unit / 2
    at = max(y, floor + half)
    placed.append((at, y, label, c))
    floor = at + half
for at, y, label, c in placed + [(ends[0][0], ends[0][0], ends[0][1], ends[0][2])]:
    ax.text(1000 * 1.3, at, label, color=c, va="center", fontsize=11, linespacing=1.05)
    if abs(at - y) > 0.5:
        ax.plot([1000 * 1.06, 1000 * 1.24], [y, at], color=c, clip_on=False, zorder=1, lw=1.2, ls=(0, (1, 1.6)))
out = figures.save(fk, fig, "fig2_set_size", pd.concat(plotted, ignore_index=True))
per_role = five_draw_per_role()
per_role[per_role.kind.isin(DRAWN)].to_parquet(out / "five_draws_at_100.parquet", index=False)
