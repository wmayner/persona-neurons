"""The halves: each role's top 100 neurons split in two by role-specificity, a neuron's |c| for the role as a share of its summed
|c| over all 275 roles: the 50 with the highest share are the most role-specific, and the other 50 the least. The share of replies with the persona's traits at 4× strength for the top 100, the 50 most role-specific and the 50 least:
means over the roles with 95% bootstrap intervals over roles (10,000 resamples).

Reads role_steering/ and role_steering_sizes/ and writes figures/fig6c_halves/ in the data directory (figure.png,
figure.pdf, data.parquet with the means, per_role.parquet with each role's shares, paired.parquet with each half minus
the top 100 over roles).

    uv run figures/fig6c_halves.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import figures
from persona_neurons.config import data_dir
from persona_neurons.thread_tables import mean_ci, set_size_per_role

WORD = figures.wording("role", "persona")
BARS = [("neurons", 100, "top 100\nneurons"), ("specific", 50, f"the 50 most\n{WORD}-specific"), ("shared", 50, f"the 50 least\n{WORD}-specific")]
# the same orange, set back (lighter on white, darker on the dark theme): none of the bars is a control
LIGHT_ACCENT = "#8A5A38" if figures.DARK else "#F0B48A"

fk = figures.kit("post")
per_role = set_size_per_role()
per_role = pd.concat([per_role[(per_role.kind == k) & (per_role.k == n)] for k, n, _ in BARS], ignore_index=True)
assert per_role.groupby("kind").role.nunique().eq(275).all()
stats = {k: mean_ci(per_role[per_role.kind == k].in_role) for k, _, _ in BARS}

wide = per_role.pivot(index="role", columns="kind", values="in_role")
paired = pd.DataFrame([{"comparison": f"{k} minus neurons", "mean": m, "lo": lo, "hi": hi, "roles": len(wide),
                        "ahead": int((wide[k] > wide.neurons).sum()), "tied": int((wide[k] == wide.neurons).sum()),
                        "behind": int((wide[k] < wide.neurons).sum())}
                       for k in ("specific", "shared") for m, lo, hi in [mean_ci(wide[k] - wide.neurons)]])
print(pd.DataFrame(stats, index=["mean", "lo", "hi"]).T.round(3))
print(paired.round(3).to_string(index=False))

fig, ax = plt.subplots()
for i, (k, _, _) in enumerate(BARS):
    mean, lo, hi = stats[k]
    ax.bar(i, mean, width=0.62, color=fk.ACCENT if k == "specific" else LIGHT_ACCENT, zorder=2)
    ax.plot([i, i], [lo, hi], color=fk.INK, lw=1.2, zorder=3)
    ax.text(i, hi + 1.5, f"{mean:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="semibold", color=fk.ACCENT)
ax.set(xticks=range(len(BARS)), xticklabels=[n for _, _, n in BARS], xlim=(-0.6, len(BARS) - 0.4), ylim=(0, 80),
       yticks=[0, 20, 40, 60, 80], ylabel="Replies with the persona's traits (%)")
for t, (k, _, _) in zip(ax.get_xticklabels(), BARS):
    t.set_color(fk.ACCENT if k == "specific" else fk.INK)
    t.set_fontweight("semibold")
ax.tick_params(axis="x", length=0)
fk.title(fig, f"The 50 most {WORD}-specific neurons\nsteer as well as all 100")
plotted = pd.DataFrame([{"kind": k, "k": n, "measure": "in_role", "mean": stats[k][0], "lo": stats[k][1], "hi": stats[k][2]}
                        for k, n, _ in BARS])
out = figures.save(fk, fig, "fig6c_halves", plotted)
per_role.to_parquet(out / "per_role.parquet", index=False)
paired.to_parquet(out / "paired.parquet", index=False)
