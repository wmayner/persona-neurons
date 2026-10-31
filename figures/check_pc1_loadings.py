"""Each role's loading on PC1 of persona space in Qwen3-8B against its loading in Lu et al.'s published Qwen3-32B vectors.
Reads check/pc1_loadings.parquet (steps/check_published.py) and writes figures/check_pc1_loadings/ in the data directory.

    uv run figures/check_pc1_loadings.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persona_neurons import figures
from persona_neurons.config import data_dir

N_NAMED = 3     # roles named at each end of Qwen3-8B's PC1


def name_end(ax, fk, end: pd.DataFrame, side: str, step: float, dx: float, y_range: tuple[float, float]):
    """Label the roles in `end` in a column beside their points, swapping labels until no two leader lines cross."""
    end = end.sort_values("Qwen3-8B")
    ys = end["Qwen3-8B"].mean() + (np.arange(len(end)) - (len(end) - 1) / 2) * step
    ys += max(0, y_range[0] - ys.min()) - max(0, ys.max() - y_range[1])
    x = end["Qwen3-32B"].max() + dx if side == "right" else end["Qwen3-32B"].min() - dx
    pts = end[["Qwen3-32B", "Qwen3-8B"]].to_numpy()

    def cross(i, j):  # do the segments from (x, ys[i]) to pts[i] and from (x, ys[j]) to pts[j] intersect?
        def turn(a, b, c):
            return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        a, b, c, d = (x, ys[i]), pts[i], (x, ys[j]), pts[j]
        return turn(a, b, c) != turn(a, b, d) and turn(c, d, a) != turn(c, d, b)

    for _ in range(10):
        pairs = [(i, j) for i in range(len(ys)) for j in range(i + 1, len(ys)) if cross(i, j)]
        if not pairs:
            break
        i, j = pairs[0]
        ys[[i, j]] = ys[[j, i]]
    for (role, row), y in zip(end.iterrows(), ys):
        ax.annotate(role, (row["Qwen3-32B"], row["Qwen3-8B"]), xytext=(x, y), textcoords="data",
                    ha="left" if side == "right" else "right", va="center", color=fk.INK,
                    arrowprops=dict(arrowstyle="-", color=fk.GREY, lw=0.7, shrinkA=2, shrinkB=3))


def main():
    df = pd.read_parquet(data_dir() / "check" / "pc1_loadings.parquet")
    wide = df.pivot(index="role", columns="model", values="loading").dropna()  # roles in both models
    r = np.corrcoef(wide["Qwen3-32B"], wide["Qwen3-8B"])[0, 1]

    fk = figures.kit()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(wide["Qwen3-32B"], wide["Qwen3-8B"], s=14, color=fk.color("Qwen3-8B"), alpha=0.6, lw=0)
    step = 0.09 * (wide["Qwen3-8B"].max() - wide["Qwen3-8B"].min())
    dx = 0.04 * (wide["Qwen3-32B"].max() - wide["Qwen3-32B"].min())
    y_range = wide["Qwen3-8B"].min(), wide["Qwen3-8B"].max()
    name_end(ax, fk, wide.nsmallest(N_NAMED, "Qwen3-8B"), "right", step, 5 * dx, y_range)
    name_end(ax, fk, wide.nlargest(N_NAMED, "Qwen3-8B"), "left", step, 5 * dx, y_range)
    ax.set(xlabel="PC1 loading in the published Qwen3-32B vectors", ylabel="PC1 loading in Qwen3-8B")
    ax.grid(axis="x")

    fk.title(fig, f"PC1 loadings correlate r = {r:.2f} with the published Qwen3-32B")
    print(f"r = {r:.3f} over {len(wide)} roles; wrote {figures.save(fk, fig, 'check_pc1_loadings', df)}")


if __name__ == "__main__":
    main()
