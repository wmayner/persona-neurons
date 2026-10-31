"""Per-role tables from the judged replies in the data directory.

role_steering/ (step 6) holds draw 0 of the five conditions at 4× strength. role_steering_draws/ adds draws 1 to 4 of
the two conditions with random draws (randomly rotated directions and random neurons). role_steering_sizes/ adds the
top neurons and random neurons at other set sizes, and the two 50-neuron halves of the top 100 by role-specificity;
role_steering_control_sizes/ adds the randomly rotated directions and top principal components at other set sizes.
A reply has the persona's traits when the judge scores it 2 or 3, and fully takes on the persona when it scores 3.
Replies the judge did not score are left out. Shares are percentages.
"""
import numpy as np
import pandas as pd

from persona_neurons.config import data_dir

DRAWN = ["rotation_matched", "random"]
K_DRAWN = 100   # the one size at which these have five draws
SET_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
BOOTSTRAP_SEED, N_BOOT = 20261105, 10_000   # resamples over roles
# Folders with replies at several set sizes, read in this order; a missing folder is skipped
SIZE_FOLDERS = ["role_steering", "role_steering_sizes", "role_steering_control_sizes"]


def in_role_shares(judged: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group of `by`: replies judged, and the shares with the persona's traits and fully taking on the persona."""
    scored = judged[judged.score.notna()]
    out = (scored.assign(in_role=scored.score >= 2, fully=scored.score == 3)
           .groupby(by, as_index=False).agg(in_role=("in_role", "mean"), fully=("fully", "mean"), n=("score", "size")))
    out[["in_role", "fully"]] = out[["in_role", "fully"]].astype(float) * 100
    return out


def mean_ci(values, seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """The mean and its 95% bootstrap interval."""
    values = np.asarray(values, float)
    boots = values[np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))].mean(1)
    return float(values.mean()), *map(float, np.percentile(boots, [2.5, 97.5]))


def five_draw_judged() -> pd.DataFrame:
    """Every judged reply of the five conditions at 4× strength, with the draws of the two that have them."""
    base = data_dir()
    return pd.concat([pd.read_parquet(base / "role_steering" / "judged_replies.parquet"),
                      pd.read_parquet(base / "role_steering_draws" / "judged_replies.parquet")], ignore_index=True)


def five_draw_per_role() -> pd.DataFrame:
    """One row per role and condition: the role's shares, for the rotated directions and the random neurons the mean
    over the role's shares in their five draws."""
    per_draw = in_role_shares(five_draw_judged(), ["role", "kind", "draw"])
    return (per_draw.groupby(["role", "kind"], as_index=False)
            .agg(in_role=("in_role", "mean"), fully=("fully", "mean"), n=("n", "sum"), draws=("draw", "nunique")))


def contribution_per_role() -> pd.DataFrame:
    """One row per role and set: the set's contribution to the persona vector, as a share of its length (%), and the
    share of replies with the persona's traits, each the mean over a role's draws."""
    base = data_dir()
    drawn = pd.read_parquet(base / "role_steering_draws" / "shares.parquet")
    one = pd.read_parquet(base / "role_steering" / "shares.parquet")
    shares = pd.concat([one[~one.kind.isin(DRAWN)], drawn], ignore_index=True)
    shares = shares.groupby(["role", "kind"], as_index=False).share.mean()
    return shares.merge(five_draw_per_role()[["role", "kind", "in_role"]], on=["role", "kind"], validate="1:1")


def set_size_judged() -> pd.DataFrame:
    """Every judged reply with a set size: k is 0 for the persona vector, and step 6's conditions have k = 100."""
    frames = []
    for folder in SIZE_FOLDERS:
        path = data_dir() / folder / "judged_replies.parquet"
        if path.exists():
            j = pd.read_parquet(path)
            frames.append(j if "k" in j else j.assign(k=np.where(j.kind == "vector", 0, 100)))
    j = pd.concat(frames, ignore_index=True)
    assert not j.duplicated(["role", "kind", "k", "draw", "question_index", "sample"]).any()
    return j


def set_size_per_role() -> pd.DataFrame:
    """One row per role, set and set size. The size runs have one draw of the random neurons and of the rotated
    directions; draw 0 is used at k = 100 too, so all sizes are comparable."""
    return in_role_shares(set_size_judged(), ["kind", "k", "role"])


def set_size_with_five_draws() -> pd.DataFrame:
    """set_size_summary's rows for the share with the persona's traits, with `draws`: one at every size, except that the
    rotated directions and the random neurons at k = K_DRAWN are the mean over each role's five draws."""
    s = set_size_summary().query("measure == 'in_role'").assign(draws=1)
    per_role = five_draw_per_role()
    for kind in DRAWN:
        row = (s.kind == kind) & (s.k == K_DRAWN)
        s.loc[row, ["mean", "lo", "hi"]] = mean_ci(per_role[per_role.kind == kind].sort_values("role").in_role)
        s.loc[row, "draws"] = 5
    return s


def set_size_summary(per_role: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per set, set size and measure: the mean over roles and its 95% bootstrap interval."""
    per_role = set_size_per_role() if per_role is None else per_role
    rows = []
    for (kind, k), g in per_role.groupby(["kind", "k"]):
        g = g.sort_values("role")
        for measure in ("in_role", "fully"):
            mean, lo, hi = mean_ci(g[measure])
            rows.append({"kind": kind, "k": k, "measure": measure, "mean": mean, "lo": lo, "hi": hi, "roles": len(g),
                         "replies": int(g.n.sum())})
    return pd.DataFrame(rows)
