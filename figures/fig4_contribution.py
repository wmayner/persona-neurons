"""Contribution against steering: for each set of 100 steered at 4× strength, an outlined bar for its mean contribution
to the persona vector and a solid bar for its mean share of replies with the persona's traits, each with a 95% bootstrap
interval over roles (10,000 resamples). The randomly rotated directions and the random neurons are means over five draws.

Reads role_steering/ and role_steering_draws/ and writes figures/fig4_contribution/ in the data directory (figure.png,
figure.pdf, data.parquet).

    uv run figures/fig4_contribution.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import figures
from persona_neurons.thread_tables import contribution_per_role, mean_ci

ORDER = ["neurons", "rotation_matched", "components", "random"]
NAMES = {"neurons": "top 100\nneurons", "components": "top 100\nprincipal\ncomponents",
         "rotation_matched": "100 randomly\nrotated\ndirections", "random": "100 random\nneurons"}
d = contribution_per_role()
fk = figures.kit("post")
col = {"neurons": fk.color("top 100 neurons"), "components": figures.CONTROL, "rotation_matched": figures.CONTROL,
       "random": figures.CONTROL}
stats = {k: {m: mean_ci(d[d.kind == k][m]) for m in ("share", "in_role")} for k in ORDER}
pct = lambda v: f"{v:.0f}%" if v >= 1 else "<1%"

fig, ax = plt.subplots(figsize=(6, 4.8))
WIDTH, OFFSET = 0.32, 0.17
for i, k in enumerate(ORDER):
    for dx, m, hollow in ((-OFFSET, "share", True), (OFFSET, "in_role", False)):
        mean, lo, hi = stats[k][m]
        ax.bar(i + dx, mean, width=WIDTH, facecolor="white" if hollow else col[k], edgecolor=col[k], linewidth=2.2,
               hatch="////" if hollow else None, zorder=2)
        ax.plot([i + dx, i + dx], [lo, hi], color=fk.INK, lw=1.2, zorder=3)
        ax.text(i + dx, max(hi, mean) + 1.5, pct(mean), ha="center", va="bottom", fontsize=12, fontweight="semibold",
                color=col[k])
ax.set(xticks=range(len(ORDER)), xticklabels=[NAMES[k] for k in ORDER], xlim=(-0.5, len(ORDER) - 0.5), ylim=(0, 75),
       yticks=[0, 25, 50, 75], ylabel="%")
for t, k in zip(ax.get_xticklabels(), ORDER):
    t.set_color(col[k])
    t.set_fontweight("semibold")
ax.tick_params(axis="x", length=0)
ax.legend(handles=[Patch(facecolor="white", edgecolor=figures.CONTROL, hatch="////", linewidth=1.6),
                   Patch(facecolor=figures.CONTROL, edgecolor=figures.CONTROL)],
          labels=["contribution to the persona vector", "replies with the persona's traits"],
          loc="upper right", frameon=False, fontsize=11)
fk.title(fig, "The neuron basis is privileged")
plotted = pd.DataFrame([{"kind": k, "measure": m, "mean": s[0], "lo": s[1], "hi": s[2]} for k in ORDER
                        for m, s in stats[k].items()])
figures.save(fk, fig, "fig4_contribution", plotted)
