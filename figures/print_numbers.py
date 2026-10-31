"""Every number the thread states, labelled by its tweet, then the numbers behind the figures, from the data directory.
Run `uv run steps/7_single_neurons.py tables` first.

    uv run figures/print_numbers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons.thread_tables import mean_ci

CONDITIONS = ["vector", "neurons", "rotation_matched", "components", "random"]
SINGLE_LAYERS = [7, 8, 9, 10]
N_LAYERS = 36


def steering(per_role: pd.DataFrame) -> pd.DataFrame:
    """The mean over roles of each condition's shares, in the thread's order."""
    t = per_role.groupby("kind")[["in_role", "fully"]].mean().reindex(CONDITIONS)
    return t.assign(draws=per_role.groupby("kind").draws.max(), roles=per_role.groupby("kind").role.nunique()).reset_index()


def paired(per_role: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Each pair's mean difference over roles in the share with the persona's traits, with its 95% bootstrap interval, and
    how many roles lead, tie and trail."""
    w = per_role.pivot(index="role", columns="kind", values="in_role")
    rows = []
    for a, b in pairs:
        diff = (w[a] - w[b]).to_numpy()
        mean, lo, hi = mean_ci(diff)
        rows.append({"comparison": f"{a} minus {b}", "mean": mean, "lo": lo, "hi": hi, "ahead": int((diff > 0).sum()),
                     "tied": int((diff == 0).sum()), "behind": int((diff < 0).sum())})
    return pd.DataFrame(rows)


def sommelier_counts(judged: pd.DataFrame) -> pd.DataFrame:
    """The sommelier's replies with the persona's traits and fully taking on the persona, out of those judged."""
    s = judged[judged.score.notna()]
    return (s.assign(has_traits=s.score >= 2, fully=s.score == 3).groupby("kind", as_index=False)
            .agg(has_traits=("has_traits", "sum"), fully=("fully", "sum"), judged=("score", "size")))


def single_neuron_shares(share: pd.DataFrame, natural: pd.DataFrame) -> pd.DataFrame:
    """Each kind of direction's share labelled a persona, pooled over layers 7 to 10 in the sign the figure draws, and at a
    size the direction reaches in real replies."""
    drawn = {"role": "+", "pc": "both", "combination": "both", "neuron": "both", "random": "both"}

    def pooled(t):
        t = t[t.layer.isin(SINGLE_LAYERS) & (t.sign == t.kind.map(drawn))]
        return t.groupby("kind")[["n", "k"]].sum()

    a, b = pooled(share), pooled(natural)
    return pd.DataFrame({"directions": a.n, "percent": 100 * a.k / a.n, "percent_natural": 100 * b.k / b.n}
                        ).sort_values("percent", ascending=False).reset_index()


def show(name: str, t: pd.DataFrame) -> None:
    print(f"\n{name}\n{t.round(4).to_string(index=False)}")


def tweet(text: str) -> None:
    print(f"\n=== {text}")


if __name__ == "__main__":
    from persona_neurons.config import data_dir
    from persona_neurons.necessity import paired_summary
    from persona_neurons.thread_tables import contribution_per_role, five_draw_per_role, set_size_per_role, set_size_summary, set_size_with_five_draws

    base = data_dir()
    per_role = five_draw_per_role()
    table = steering(per_role).set_index("kind")
    sizes = set_size_summary()
    sizes = sizes[sizes.measure == "in_role"]
    top = sizes[sizes.kind == "neurons"].set_index("k")["mean"]
    c = contribution_per_role()
    share = c.groupby("kind").share.mean()

    tweet("Opener tweet: ~50 of ~440k neurons give most of the effect; other bases' top directions do not")
    with np.load(base / "split" / "shared_inputs.npz") as z:
        per_layer = z["sd"].shape[1]
    print(f"neurons in Qwen3-8B: {N_LAYERS} layers × {per_layer:,} = {N_LAYERS * per_layer:,}; in layers 0 to 18: {19 * per_layer:,}")
    print(f"replies with the persona's traits at 4× strength: top 50 neurons {top[50]:.1f}%, the persona vector {table.in_role['vector']:.1f}%")

    tweet("Set-size tweet: ~50 neurons are enough for most of the effect, and ~100 nearly reproduce it")
    print(f"top 50 neurons {top[50]:.1f}%, top 100 neurons {top[100]:.1f}%, the persona vector {table.in_role['vector']:.1f}%")

    tweet("Contribution tweet: rotated and principal-component directions barely steer; top PCs contribute ~2× as much as top neurons")
    show("Replies with the persona's traits at 4× strength, mean over roles (%); the drawn sets are means over five draws", table.reset_index())
    show("Contribution to the persona vector (% of its length), mean over roles", share.reindex(CONDITIONS[1:]).reset_index())
    print(f"top 100 principal components over top 100 neurons: {share['components'] / share['neurons']:.2f} times")
    show("Paired over roles, replies with the persona's traits (points)",
         paired(per_role, [("neurons", "rotation_matched"), ("vector", "neurons"), ("neurons", "components")]))

    tweet("Shared-neurons tweet: personas farther from the Assistant share more of their top neurons")
    shared = pd.read_parquet(base / "shared_neurons" / "roles.parquet")
    print(f"Spearman correlation of distance from the Assistant with neurons in common: "
          f"{stats.spearmanr(shared.away, shared.overlap)[0]:.3f}, over {len(shared)} roles")
    show("Named roles", shared[shared.role.isin(["assistant", "accountant", "sommelier", "pirate"])][["role", "away", "overlap"]])

    tweet("Halves tweet: the 50 most persona-specific are as effective as the top 100 (64%); the 50 least persona-specific yield 33%")
    show("Replies with the persona's traits (%)", sizes[(sizes.kind.isin(["specific", "shared"])) | ((sizes.kind == "neurons") & (sizes.k == 100))][["kind", "k", "mean", "lo", "hi"]])

    halves = set_size_per_role().query("(k == 50 and kind in ['specific', 'shared']) or (k == 100 and kind == 'neurons')").pivot(index="role", columns="kind", values="in_role")
    show("Paired over roles (points); specific is the 50 most persona-specific, and shared the 50 least", pd.DataFrame([
        {"comparison": f"{a} minus {b}"} | paired_summary(halves[a], halves[b]) for a, b in (("specific", "shared"), ("specific", "neurons"), ("shared", "neurons"))])[
        ["comparison", "n", "mean_diff", "lo", "hi", "bf10_r0.707", "bf10_r1.0"]])

    tweet("Spectrum tweet, with the sommelier figure: shared neurons write along the Assistant Axis, pointing away from the Assistant")
    axis = pd.read_parquet(base / "shared_neurons" / "on_axis_summary.parquet")
    show("Cosine of each set's direct write with the axis (cos_axis), and with the part of the role's vector off the axis (cos_off_axis); shared is the 50 least persona-specific, specific the 50 most",
         axis[axis.metric.isin(["cos_axis", "cos_off_axis"])][["metric", "set", "mean", "lo", "hi"]])
    p = pd.read_parquet(base / "shared_neurons" / "on_axis_paired.parquet")
    show("Paired over roles", p[(p.comparison == "shared minus specific") & p.metric.isin(["cos_axis", "cos_off_axis"])][
        ["metric", "comparison", "mean_diff", "lo", "hi", "roles_a_higher", "n", "bf10_r0.707", "bf10_r1.0"]])
    rho = pd.read_parquet(base / "shared_neurons" / "on_axis_spearman.parquet")
    print(f"Within a role's top 100, Spearman correlation of how many lists hold a neuron with its write's cosine with the axis: "
          f"mean {rho.spearman.mean():.3f}, positive in {(rho.spearman > 0).sum()} of {len(rho)} roles")
    print(f"The same with its role-specificity in place of the count: mean {rho.spearman_specificity.mean():.3f}, "
          f"negative in {(rho.spearman_specificity < 0).sum()} of {len(rho)} roles")
    cells = pd.read_parquet(base / "shared_neurons" / "sommelier_neurons.parquet")
    show("The sommelier's three neurons: how many roles' top 100 hold each", cells.assign(held=cells["rank"] > 0).groupby(["layer", "neuron"], as_index=False).held.sum())

    tweet("Necessity tweet: removal under the role's prompt barely changes the persona (0.2pp; whole vector 10pp)")
    n = pd.read_parquet(base / "necessity" / "paired.parquet")
    n = n[(n.patch == "subtract") & (n.measure == "in_role") & n.comparison.str.startswith("prompt alone minus")]
    n = n.assign(set=n.comparison.str.replace("prompt alone minus ", "")).set_index("set").reindex(CONDITIONS).reset_index()
    n[["mean_diff", "lo", "hi"]] *= 100
    show("Drop in replies with the persona's traits when the set's change is subtracted (points)", n[["set", "n", "mean_diff", "lo", "hi", "bf10_r0.707", "bf10_r1.0"]])
    alone = pd.read_parquet(base / "necessity" / "per_role.parquet")
    print(f"With the role's system prompt alone, {100 * alone[alone.patch == 'prompt'].in_role.mean():.1f}% have the traits")
    along = alone[alone.patch == "subtract"].groupby("set").write_along_v.mean()
    print(f"The top 100's subtraction writes {100 * along['neurons'] / along['vector']:.1f}% as far along the vector as the vector's own")

    tweet("Necessity tweet: the prompt uses the top 100 about as much as other neurons that contribute a similar amount to the vector")
    same = base / "necessity_same_size"
    d = pd.read_parquet(same / "paired.parquet")
    d = d[(d.subset == "all replies") & (d.roles == "all")]
    show("Drop in log-probability, role reply minus Assistant reply, under the role's prompt (nats per token)",
         d[(d.measure == "role reply minus Assistant reply") & d.comparison.str.startswith("drop: ")][["comparison", "n", "mean_diff", "lo", "hi"]])
    show("The top 100's drop minus each same-size removal's, paired over roles (nats per token)",
         d[d.measure.isin(["role reply", "role reply minus Assistant reply"]) & d.comparison.str.startswith("top 100 neurons minus ")
           & d.comparison.str.contains("lower-ranked|slice")][["measure", "comparison", "mean_diff", "lo", "hi", "bf10_r0.707", "bf10_r1.0"]])
    s = pd.read_parquet(same / "sizes_summary.parquet")
    show("The drop per unit of the change's measured norm at layer 18; the slice is matched to the top 100 along the vector only",
         s.dropna(subset=["difference_per_norm_measured"])[
             ["condition", "measured_norm", "drop_difference", "difference_per_norm_measured", "difference_per_norm_measured_lo", "difference_per_norm_measured_hi"]])
    r = pd.read_parquet(same / "share_of_vector_drop.parquet")
    show("As a share of the persona vector's drop", r[(r.subset == "all replies") & (r.measure == "role reply minus Assistant reply")][["condition", "share", "lo", "hi", "alpha_mean"]])
    show("Assistant reply: its rise under each removal is the negative of this drop (nats per token)",
         d[(d.measure == "Assistant reply") & d.comparison.str.startswith("drop: ")][["comparison", "mean_diff", "lo", "hi"]])
    jd = pd.read_parquet(same / "judged_paired.parquet")
    jd = jd[(jd.measure == "in_role") & (jd.comparison == "drop")].assign(mean_diff=lambda t: 100 * t.mean_diff, lo=lambda t: 100 * t.lo, hi=lambda t: 100 * t.hi)
    show("Judged: drop in replies with the persona's traits from the same-size removals (points)", jd[["set", "n", "mean_diff", "lo", "hi", "bf10_r0.707", "bf10_r1.0"]])

    tweet("Single-neuron tweet: one neuron, far past its natural range, gives a persona as often as a random mixture, and less often than a principal component")
    tables = base / "single_neurons" / "tables"
    show("Single directions at layers 7 to 10: share labelled a persona (%)",
         single_neuron_shares(pd.read_parquet(tables / "share.parquet"), pd.read_parquet(tables / "share_natural.parquet")))
    steered = pd.read_parquet(tables / "push_size.parquet")
    print(f"single neurons are steered to a median {steered[steered.layer.isin(SINGLE_LAYERS)].per_largest.median():.1f} times their largest value in real replies")

    tweet("Takeaway tweet: sparse patterns of neurons are as effective as difference-of-means vectors")
    print(f"top 100 neurons {table.in_role['neurons']:.1f}%, the persona vector {table.in_role['vector']:.1f}% (paired difference above)")

    tweet("The figures: the opener's sommelier and the set-size lines")
    j = pd.read_parquet(base / "role_steering" / "judged_replies.parquet")
    j = pd.concat([j[j.role == "sommelier"], pd.read_parquet(base / "role_steering" / "unsteered_sommelier.parquet")])
    show("The sommelier's 10 replies, draw 0", sommelier_counts(j[j.kind.isin(["none", *CONDITIONS])]))
    plotted = set_size_with_five_draws()
    for kind, name in (("neurons", "Top neurons"), ("rotation_matched", "Randomly rotated directions"),
                       ("components", "Top principal components"), ("random", "Random neurons")):
        show(f"{name} at 4× strength, by set size (%), as the figure plots them", plotted[plotted.kind == kind][["k", "draws", "mean", "lo", "hi"]])

    tweet("Not in the thread: the scaled run, on the webpage")
    z = pd.read_parquet(base / "necessity_scaled" / "paired.parquet")
    z = z[(z.patch == "scaled") & (z.measure == "in_role") & z.comparison.str.startswith("prompt alone minus")]
    show("Drop in replies with the persona's traits, the top 100 scaled 13.4×, and random neurons with a change of the same size (points)",
         z.assign(mean_diff=100 * z.mean_diff, lo=100 * z.lo, hi=100 * z.hi)[["comparison", "mean_diff", "lo", "hi"]])
    u = pd.read_parquet(base / "necessity_scaled" / "usual_range" / "summary.parquet")
    show("The median neuron's shift at that size and at its own size, in SDs over the Assistant's replies (bound and subsample)",
         u[(u.patch == "scaled") & (u.set == "neurons")][["measure", "50%"]])
