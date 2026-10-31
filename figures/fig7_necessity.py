"""Necessity: under a role's own system prompt, the drop in the share of replies with the persona's traits when each
set's change is subtracted while the model replies, as a mean over the 275 roles with its 95% bootstrap interval.

Reads necessity/paired.parquet and necessity/per_role.parquet and writes figures/fig7_necessity/ in the data directory
(figure.png, figure.pdf, data.parquet).

    uv run figures/fig7_necessity.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import figures
from persona_neurons.config import data_dir

ROWS = [  # set, label, colour role, or the colour itself
    ("vector", "the persona vector", "the role's steering vector"),
    ("neurons", "top 100 neurons", "top 100 neurons"),
    ("rotation_matched", "100 randomly rotated directions", figures.CONTROL),
    ("components", "top 100 principal components", figures.CONTROL),
    ("random", "100 random neurons", figures.CONTROL),
]

base = data_dir() / "necessity"
p = pd.read_parquet(base / "paired.parquet")
p = p[(p.patch == "subtract") & (p.measure == "in_role")].set_index("comparison")
d = pd.DataFrame([{"set": s, "label": label, "n": p.loc[f"prompt alone minus {s}", "n"],
                   **{c: 100 * p.loc[f"prompt alone minus {s}", c] for c in ("mean_diff", "lo", "hi")}}
                  for s, label, _ in ROWS])
per = pd.read_parquet(base / "per_role.parquet")
prompt_alone = 100 * per[per.patch == "prompt"].in_role.mean()

fk = figures.kit("post")
plt.rcParams["figure.constrained_layout.use"] = False  # the axis label and the grey line are placed by hand
fig, ax = plt.subplots(figsize=(6, 4.8))
ax.axvline(0, color=fk.GREY, lw=1, zorder=0)
for i, (r, (_, _, role)) in enumerate(zip(d.itertuples(), ROWS)):
    colour = role if role.startswith("#") else fk.color(role)
    ax.plot([r.lo, r.hi], [i, i], color=colour, lw=2.5, solid_capstyle="round")
    ax.plot(r.mean_diff, i, "o", ms=9, color=colour)
    ax.text(r.hi + 0.35, i, f"{r.mean_diff:.1f}", va="center", color=colour, fontweight="bold")
ax.set(yticks=range(len(d)), yticklabels=d.label, ylim=(len(d) - 0.5, -0.5), xlim=(-1.5, 13), xticks=[0, 5, 10])
for t, (_, _, role) in zip(ax.get_yticklabels(), ROWS):
    t.set_color(role if role == figures.CONTROL else fk.INK)
ax.grid(False)
ax.tick_params(axis="y", length=0)
fig.supxlabel("Drop in replies with the persona's traits (pp)", y=0.07, fontsize=plt.rcParams["axes.labelsize"])
fig.text(0.5, 0.019, f"under {figures.wording('the role\'s', 'its')} system prompt ({prompt_alone:.0f}% baseline)", ha="center",
         va="bottom", color=fk.MUTED, fontsize=plt.rcParams["xtick.labelsize"])
fk.title(fig, "Removing the top 100 neurons barely changes\na prompted persona")
fig.subplots_adjust(left=0.45, right=0.96, top=0.825, bottom=0.1875)
figures.save(fk, fig, "fig7_necessity", d)
