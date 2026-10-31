"""How roles share neurons, from step 5's split.

`role_table`: per role, `away`, the cosine of its persona vector with the unit vector from the default Assistant towards
the mean role, and `overlap`, the neurons in both its top 100 and another role's, averaged over the other roles.
`neuron_cells`: for chosen neurons and every role, the change in the neuron's activation in the neuron's SDs (`x`), its
contribution as a percentage of the role's vector (`y`), and its place in the role's top 100 (`rank`, 0 if absent).
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from persona_neurons.necessity import paired_summary
from persona_neurons.role_steering import layer_matched_random, write_vector
from persona_neurons.thread_tables import mean_ci

HALVES = ("shared", "specific")
COSINES = ("cos_axis", "cos_axis_loo", "cos_v", "cos_off_axis")


def pooled_sd(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Each column's SD around its own group's mean, pooled over the groups (rows of x are replies)."""
    order = np.argsort(group, kind="stable")
    _, sizes = np.unique(group[order], return_counts=True)
    x = np.asarray(x, dtype=np.float64)[order]
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    means = np.add.reduceat(x, starts, axis=0) / sizes[:, None]
    return np.sqrt(((x - np.repeat(means, sizes, axis=0)) ** 2).sum(0) / sizes.sum())


def away_from_assistant(v: np.ndarray) -> np.ndarray:
    """v: (roles, d_model), each role's persona vector. With u = −mean_s v_s / |mean_s v_s|, the unit vector from the mean
    role to the default Assistant, each role's −u · v̂_r."""
    v = np.asarray(v, np.float64)
    mean = v.mean(0)
    return (v @ mean) / (np.linalg.norm(mean) * np.linalg.norm(v, axis=1))


def role_table(roles: list[str], away: np.ndarray, top: pd.DataFrame) -> pd.DataFrame:
    """top: role, layer, neuron, one row per member of each role's top 100. mean_others is the mean, over the role's 100
    neurons, of how many other roles' lists hold the neuron; overlap = 100 / (roles − 1) × mean_others is the number of
    neurons the role's list has in common with another role's, averaged over the other roles."""
    n = top.groupby(["layer", "neuron"]).role.size().rename("n_lists").reset_index()
    m = top.merge(n, on=["layer", "neuron"]).assign(others=lambda d: d.n_lists - 1)
    mean_others = m.groupby("role").others.mean().reindex(roles)
    return pd.DataFrame({"role": roles, "away": away, "mean_others": mean_others.to_numpy(),
                         "overlap": 100 / (len(roles) - 1) * mean_others.to_numpy()})


def neuron_cells(roles: list[str], delta: np.ndarray, p: np.ndarray, norm: np.ndarray, sd: np.ndarray, top: pd.DataFrame,
                 neurons: list[tuple[int, int]]) -> pd.DataFrame:
    """One row per neuron and role, for the (layer, neuron) pairs in `neurons`. delta and p: (roles, layers, neurons);
    norm: each role's |v_r|; sd: (layers, neurons); top: role, rank, layer, neuron."""
    parts = []
    for layer, neuron in neurons:
        d = np.asarray(delta[:, layer, neuron], np.float64)
        rank = top[(top.layer == layer) & (top.neuron == neuron)].set_index("role")["rank"]
        parts.append(pd.DataFrame({"layer": layer, "neuron": neuron, "role": roles, "x": d / float(sd[layer, neuron]),
                                   "y": 100 * d * np.asarray(p[:, layer, neuron], np.float64) / np.asarray(norm, np.float64),
                                   "rank": rank.reindex(roles).fillna(0).astype(int).to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def unit_mean(v: np.ndarray, leave_out: int | None = None) -> np.ndarray:
    """The unit vector along the mean of the rows of v, without row leave_out if given. For persona vectors this is û, the
    unit vector from the default Assistant towards the mean role (the reverse of the Assistant Axis)."""
    m = v.mean(0) if leave_out is None else (v.sum(0) - v[leave_out]) / (len(v) - 1)
    return m / np.linalg.norm(m)


def write_cosines(w: torch.Tensor, axis: torch.Tensor, loo: torch.Tensor, v_r: torch.Tensor) -> dict:
    """The cosines of a set's direct write w = Σ Δa W_down[:, n] with û (with û from the other roles too), with v̂_r, and with
    v_r's part off the axis, x_r = v_r − (v_r · û) û."""
    x = v_r - (v_r @ axis) * axis
    return {"cos_axis": (w @ axis / w.norm()).item(), "cos_axis_loo": (w @ loo / w.norm()).item(),
            "cos_v": (w @ v_r / (w.norm() * v_r.norm())).item(), "cos_off_axis": (w @ x / (w.norm() * x.norm())).item()}


def on_axis_table(roles: list[str], v: np.ndarray, delta: np.ndarray, halves: pd.DataFrame, w_down, n_draws: int, seed: int) -> pd.DataFrame:
    """One row per set, draw and role: the cosines of write_cosines for the "shared" and "specific" halves and, as controls,
    draws of 50 random neurons with as many in each layer as each half, carrying the role's own Δa. v: (roles, d_model);
    delta: (roles, layers, neurons); halves: role, kind, layer, neuron; w_down(layer): (d_model, neurons)."""
    axis = torch.tensor(unit_mean(v))
    by_role = dict(tuple(halves.groupby("role")))
    rows = []
    for i, role in enumerate(roles):
        v_r, loo = torch.tensor(v[i]), torch.tensor(unit_mean(v, leave_out=i))

        def record(members: pd.DataFrame, kind: str, draw: int) -> None:
            values = members[["layer", "neuron"]].assign(value=delta[i, members.layer, members.neuron].astype(np.float64))
            rows.append(write_cosines(write_vector(values, w_down), axis, loo, v_r) | {"role": role, "set": kind, "draw": draw, "seed": seed})

        for code, half in enumerate(HALVES, start=1):
            s = by_role[role][by_role[role].kind == half]
            record(s, half, 0)
            for d in range(n_draws):
                record(layer_matched_random(s, delta.shape[2], [seed, i, code, d]), f"random like {half}", d)
    return pd.DataFrame(rows)


def on_axis_summaries(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means over roles with 95% bootstrap intervals, each role's draws averaged first, and the paired comparisons."""
    per = t.groupby(["set", "role"], as_index=False)[list(COSINES)].mean()
    summary = [{"set": kind, "metric": m, "mean": g[m].mean(), "lo": ci[1], "hi": ci[2], "median": g[m].median(), "roles": len(g)}
               for kind, g in per.groupby("set") for m in COSINES for ci in [mean_ci(g[m])]]
    paired = []
    for m in COSINES:
        x = per.pivot(index="role", columns="set", values=m)
        for name, a, b in (("shared minus specific", x["shared"], x["specific"]),
                           ("random like shared minus random like specific", x["random like shared"], x["random like specific"]),
                           ("shared minus random like shared", x["shared"], x["random like shared"]),
                           ("specific minus random like specific", x["specific"], x["random like specific"])):
            paired.append(paired_summary(a, b) | {"metric": m, "comparison": name, "roles_a_higher": int((a > b).sum())})
    return pd.DataFrame(summary), pd.DataFrame(paired)


def on_axis_neurons(roles: list[str], v: np.ndarray, top: pd.DataFrame, w_down) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The spectrum, neuron by neuron. One row per role and top 100 neuron with cos_axis, the cosine of its write (sign of Δa
    times W_down[:, n]) with û; and, per role, the Spearman correlation of cos_axis with n_lists and with specificity. top: role,
    layer, neuron, delta, n_lists, specificity."""
    axis = torch.tensor(unit_mean(v))
    columns = {l: w_down(l).float().double() for l in sorted(top.layer.unique())}
    along = {l: (w.T @ axis).numpy() for l, w in columns.items()}
    size = {l: w.norm(dim=0).numpy() for l, w in columns.items()}
    w_axis = np.array([along[l][n] for l, n in zip(top.layer, top.neuron)])
    w_norm = np.array([size[l][n] for l, n in zip(top.layer, top.neuron)])
    t = top.assign(cos_axis=np.sign(top.delta) * w_axis / w_norm)
    rho = pd.DataFrame([{"role": r, "spearman": spearmanr(g.n_lists, g.cos_axis).statistic, "spearman_specificity": spearmanr(g.specificity, g.cos_axis).statistic}
                        for r, g in t.groupby("role")])
    return t, rho
