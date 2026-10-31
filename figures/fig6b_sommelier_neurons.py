"""Shared neurons (b): three neurons in the sommelier's top 100, one row each, every role's change in the neuron's
activation against the neuron's contribution to the role's vector, on a shared horizontal axis. One neuron's panel
quotes a reply from the public Qwen3-8B neuron persona library, which has no licence: the script fetches it from the
library's site, and draws the figure without the quote if it cannot.

Reads shared_neurons/sommelier_neurons.parquet and writes figures/fig6b_sommelier_neurons/ in the data directory
(figure.png, figure.pdf, data.parquet).

    uv run figures/fig6b_sommelier_neurons.py
"""
import pandas as pd
from matplotlib.ticker import FuncFormatter

from _shared_neurons import axes, draw_roles, figure, figures, fk, text
from persona_neurons.config import data_dir
from persona_neurons.neuron_library import first_sentence

d_all = pd.read_parquet(data_dir() / "shared_neurons" / "sommelier_neurons.parquet")
WORD = figures.wording("role", "persona")
ROW_H, ROW_TOPS = 1.49, (1.02, 2.93, 4.84)   # room for the title's two lines
fig = figure(6.9, f"A {WORD}'s top neurons span a spectrum\nfrom shared to {WORD}-specific")
ROWS = [  # how shared, layer, neuron, other roles to name, and the sommelier's label offset (points)
    ("Highly shared", 10, 3032, {"poet": (-6, 0)}, (8, 9)),
    ("Less shared", 10, 288, {"chef": (0, 9), "bartender": (-6, 6)}, (-4, 12)),
    ("Not shared", 9, 10358, {}, (9, 0)),
]
rows = []
for top, (how, layer, idx, named, somm) in zip(ROW_TOPS, ROWS):
    ax = axes(fig, top, ROW_H, sharex=rows[0] if rows else None)
    rows.append(ax)
    d = d_all[(d_all.layer == layer) & (d_all.neuron == idx)]
    k = int((d["rank"] > 0).sum())    # roles whose top 100 holds the neuron
    draw_roles(ax, d, "x", "y", 14, named, somm)
    who = f"in {k} {figures.wording('roles', 'personas')}' top 100" if k > 1 else "only in the sommelier's top 100"
    ax.set_title(f"{how}: L{layer} N{idx}, {who}", loc="left", fontsize=12, color=fk.INK)
    ymax = max(d.y.max(), 0.3)
    ax.set_ylim(-0.12 * ymax, 1.3 * ymax)
    ax.set_yticks([0, 0.5, 1] if ymax >= 0.9 else [0, 0.5] if ymax >= 0.5 else [0, 0.2])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.axvline(0, color=fk.LIGHT, lw=0.8, zorder=0)
    ax.tick_params(labelbottom=top == ROW_TOPS[-1])
# the public library's reply when this neuron is steered alone in the direction the sommelier moves it (down)
LAYER, NEURON = ROWS[-1][1:3]
sommelier = d_all[(d_all.layer == LAYER) & (d_all.neuron == NEURON) & (d_all.role == "sommelier")].iloc[0]
quote = first_sentence(LAYER, NEURON, sign=-1)
if quote:
    rows[-1].annotate(f"steered down alone: “{quote}”", (round(sommelier.x, 2), round(sommelier.y, 2)), xytext=(9, -19),
                      textcoords="offset points", color=fk.MUTED, fontsize=11, va="center")
rows[-1].set_xlim(-27, 14.5)
rows[-1].set_xlabel("Change in activation under the " + figures.wording("role", "persona") + " (SD)")
text(fig, 0.1, (ROW_TOPS[0] + ROW_TOPS[-1] + ROW_H) / 2, "Contribution to the persona vector", rotation=90,
     ha="left", va="center")
figures.save(fk, fig, "fig6b_sommelier_neurons", d_all)
