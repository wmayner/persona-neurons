"""Shared neurons (a): one dot per role, how far its persona vector points away from the Assistant (the cosine of the
vector with the reversed Assistant Axis direction) against how many top 100 neurons it has in common with the
average other role.

Reads shared_neurons/roles.parquet and writes figures/fig6a_shared_roles/ in the data directory (figure.png, figure.pdf,
data.parquet).

    uv run figures/fig6a_shared_roles.py
"""
import pandas as pd
from scipy import stats

from _shared_neurons import axes, draw_roles, figure, figures, fk, text
from persona_neurons.config import data_dir

per = pd.read_parquet(data_dir() / "shared_neurons" / "roles.parquet")
print(f"Spearman with distance from the Assistant {stats.spearmanr(per.away, per.overlap)[0]:.3f}")

fig = figure(4.5, figures.wording("Roles", "Personas") + " far from the Assistant share more neurons")
TOP, HEIGHT = 0.50, 3.40
ax = axes(fig, TOP, HEIGHT)
draw_roles(ax, per, "away", "overlap", 22, {"assistant": (6, -6), "accountant": (6, 5), "pirate": (6, 0)}, (7, -7))
lo, hi = per.away.min(), per.away.max()
ax.set_xticks([lo, hi], ["least", "most"])
ax.get_xticklabels()[0].set_ha("left")
ax.get_xticklabels()[1].set_ha("right")
ax.set(ylim=(0, None), xlabel="How far the persona vector points away from the Assistant")
text(fig, 0.1, TOP + HEIGHT / 2, "Top 100 neurons in common\nwith the average " + figures.wording("role", "persona"), rotation=90, ha="left",
     va="center", multialignment="center")
figures.save(fk, fig, "fig6a_shared_roles", per)
