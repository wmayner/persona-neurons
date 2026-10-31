"""Single neurons: at layers 7 to 10, the share of steered directions ("Who are you?", one direction steered along at
0.5 to 1.25 times the residual norm, the strongest coherent reply kept) that the judge labels a persona, for five
kinds of direction steered by the same amounts: the roles' persona vectors (towards the role), the top 50 principal
components, random mixtures of the layer's neurons, single neurons and random directions. Each layer's share is a
small dot; the share pooled over the four layers is a large dot with its 95% Wilson interval. The share at a size each
direction reaches in the model's own replies (0% for every kind) is not drawn but is saved in data.parquet.

Reads single_neurons/tables/ (step 7: share.parquet, share_natural.parquet, push_size.parquet) and writes
figures/fig5_single_neurons/ (figure.png, figure.pdf, data.parquet) in the data directory.

    uv run figures/fig5_single_neurons.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import figures
from persona_neurons.config import data_dir
from persona_neurons.single_neurons import wilson

LAYERS = [7, 8, 9, 10]
# kind: (row name, sign pooled, grey line under the name, if any; {ratio} is filled from the data)
ROWS = {"role": (figures.wording("the roles' persona vectors", "persona vectors"), "+", None),
        "pc": ("top 50 principal components", "both", None),
        "combination": ("random mixtures of neurons", "both", None),
        "neuron": ("single neurons", "both", "steered to {ratio:.0f}× its largest value in real replies (median)"),
        "random": ("random directions", "both", None)}


def rows_and_pooled(path):
    """The four layers' rows of a share table, for the sign drawn, and their pool with its Wilson interval."""
    share = pd.read_parquet(path)
    share = share[share.layer.isin(LAYERS).to_numpy() & [ROWS.get(k, (0, None))[1] == s for k, s in zip(share.kind, share.sign)]]
    pooled = share.groupby("kind", as_index=False)[["n", "k"]].sum()
    pooled["share"] = pooled.k / pooled.n
    pooled["lo"], pooled["hi"] = wilson(pooled.k, pooled.n)
    return share, pooled


tables = data_dir() / "single_neurons" / "tables"
share, pooled = rows_and_pooled(tables / "share.parquet")
pooled = pooled.sort_values("share", ascending=False).reset_index(drop=True)
share_nat, pooled_nat = rows_and_pooled(tables / "share_natural.parquet")
sizes = pd.read_parquet(tables / "push_size.parquet")
ratio = sizes[sizes.layer.isin(LAYERS)].per_largest.median()

data = pd.concat([share.assign(level="layer", size="scaled"), pooled.assign(level="layers 7 to 10", size="scaled"),
                  share_nat.assign(level="layer", size="natural"),
                  pooled_nat.assign(level="layers 7 to 10", size="natural")], ignore_index=True)
data = data.assign(row=data.kind.map(lambda k: ROWS[k][0]), neuron_push_over_largest=ratio)

fk = figures.kit("post")
fig, ax = plt.subplots(figsize=(6, 4.8))
for i, r in pooled.iterrows():
    name, _, line = ROWS[r.kind]
    c = fk.color(name)
    d = share[share.kind == r.kind]
    ax.scatter(100 * d.share, [i + 0.08] * len(d), s=22, color=c, alpha=0.5, lw=0, zorder=2)
    ax.plot([100 * r.lo, 100 * r.hi], [i + 0.08] * 2, color=c, lw=2, solid_capstyle="round", zorder=3)
    ax.scatter([100 * r.share], [i + 0.08], s=90, color=c, edgecolor=plt.rcParams["figure.facecolor"], lw=0.8, zorder=4)
    ax.text(100 * max(r.hi, d.share.max()) + 0.8, i + 0.08, f"{100 * r.share:.0f}%", va="center", color=c,
            fontweight="semibold")
    ax.text(0, i - (0.56 if line else 0.26), name, fontweight="semibold", color=fk.INK if r.kind == "role" else c, va="center")
    if line:
        ax.text(0, i - 0.26, line.format(ratio=ratio), color=fk.MUTED, va="center")
ax.set(ylim=(len(pooled) - 0.55, -0.55), yticks=[], xlim=(0, 37), xlabel="Steered directions yielding a persona (%), layers 7 to 10")
ax.spines["left"].set_visible(False)
ax.grid(axis="y", visible=False)
title = fk.title(fig, "Single neurons yield personas no more often than mixtures")
title.set_fontsize(0.97 * title.get_fontsize())   # the longer word would otherwise reach the right edge
figures.save(fk, fig, "fig5_single_neurons", data)
