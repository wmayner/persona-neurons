"""Patching under role prompts: while Qwen3-8B answers the five questions under a role's own system prompts, a set's
change is subtracted at every position. The sets are the persona vector at layer 18, the top 100 neurons, the top 100
principal components, 100 randomly rotated directions (draw 0) and 100 random neurons (draw 0).

The scaled run subtracts the top 100 neurons and the random neurons at larger sizes. SCALE_ALONG_V is the mean over roles
of |v_r| over the mean of the top 100's measured movement along v̂_r, so at that multiple the set moves the residual along
v̂_r about as far as the vector's subtraction. SCALE_STEERING is the steering runs' 4×, subtracted.

The same-size conditions compare the top 100's subtraction with a slice of the vector that moves the residual as far along
v̂_r, and with lower-ranked neurons of the same summed contribution.
"""
import contextlib

import numpy as np
import pandas as pd
import torch

from persona_neurons.role_steering import K, RowIntervention, _chat_kwargs, component_add_values, write_vector

SETS = ("vector", "neurons", "rotation_matched", "components", "random")
DIRECTION_SETS = ("components", "rotation_matched")
VECTOR_SETS = ("vector", "slice")
LOWER_SETS = ("lower", "lower_same_sign")
ITEMS = [(q, s) for q in range(5) for s in range(2)]   # 2 replies to each of the 5 questions
BOOT_SEED, N_BOOT = 20261105, 10_000
SCALE_ALONG_V, SCALE_STEERING = 13.4, -4.0
SCALED_PATCHES = ("scaled", "push")
MEASURES = ["in_role", "fully"]
CHANGE_COLS = ("mean_norm", "along_v")


def prompt_index(q: int, s: int) -> int:
    """The system prompt for question q, sample s: each of the five appears twice, with two different questions."""
    return (2 * q + s) % 5


def _cond(role: str, patch: str, kind: str, strength: float = np.nan) -> dict:
    return {"condition_id": f"{role}|{patch}|{kind}", "role": role, "patch": patch, "set": kind, "strength": strength}


def conditions(role: str) -> pd.DataFrame:
    """The role prompt alone, then each set subtracted."""
    return pd.DataFrame([_cond(role, "prompt", "none")] + [_cond(role, "subtract", s) for s in SETS])


def scaled_conditions(role: str) -> pd.DataFrame:
    """The top 100 neurons and random set 0, at SCALE_ALONG_V (patch "scaled") and SCALE_STEERING (patch "push")."""
    return pd.DataFrame([_cond(role, how, s, a) for how, a in (("scaled", SCALE_ALONG_V), ("push", SCALE_STEERING))
                         for s in ("neurons", "random")])


def same_size_conditions(role: str, alpha: float) -> pd.DataFrame:
    """A slice alpha × v_r of the vector and the two lower-ranked sets, each subtracted."""
    return pd.DataFrame([_cond(role, "subtract", "slice", alpha)] + [_cond(role, "subtract", s) for s in LOWER_SETS])


def lower_ranked(delta: np.ndarray, p: np.ndarray, k: int = K, same_sign: bool = False) -> tuple[pd.DataFrame, dict]:
    """The neurons ranked after the top k by |contribution|, in order, until their summed contribution first reaches the
    top k's, or else up to where it comes closest. same_sign keeps only neurons whose contribution has the sign of the top
    k's sum. Returns (layer, neuron, delta, contribution; {target, sum, n, matched})."""
    c = (np.asarray(delta, np.float64) * np.asarray(p, np.float64)).ravel()
    order = np.argsort(-np.abs(c), kind="stable")
    target = c[order[:k]].sum()
    sign = np.sign(target) or 1.0
    rest = order[k:]
    if same_sign:
        rest = rest[c[rest] * sign > 0]
    cum = np.cumsum(c[rest] * sign)
    reached = np.nonzero(cum >= abs(target))[0]
    n = int(reached[0]) + 1 if len(reached) else int(np.argmin(np.abs(cum - abs(target)))) + 1
    chosen = rest[:n]
    layer, neuron = np.divmod(chosen, delta.shape[1])
    return (pd.DataFrame({"layer": layer, "neuron": neuron, "delta": np.asarray(delta).ravel()[chosen], "contribution": c[chosen]}),
            {"target": float(target), "sum": float(c[chosen].sum()), "n": n, "matched": bool(len(reached))})


def _members(sets: pd.DataFrame, kind: str) -> pd.DataFrame:
    s = sets[(sets.kind == kind) & (sets.draw == 0)]
    if s.empty:
        raise ValueError(f"no members for {kind}")
    return s


def _delta_rows(s: pd.DataFrame, c: float = 1.0) -> pd.DataFrame:
    return s[["layer", "neuron"]].assign(value=c * s.delta.to_numpy().astype(np.float64)).reset_index(drop=True)


def patch_values(cond: dict, sets: pd.DataFrame, comps: dict):
    """What one condition does, as (kind, payload): ("none", None) for the role prompt alone; ("vector", None) for the
    vector or a slice of it; ("rows", DataFrame layer, neuron, value) for a RowIntervention, which removes each
    neuron's Δa, or the part of Δa along a set of directions, −V Vᵀ Δa."""
    kind = cond["set"]
    if cond["patch"] == "prompt":
        return "none", None
    if kind in VECTOR_SETS:
        return "vector", None
    if kind in DIRECTION_SETS:
        c = comps[(kind, 0)]
        by_layer = {int(layer): (c["v"][:, g.index.to_numpy()], g.coef.to_numpy())
                    for layer, g in c["rank"].head(K).groupby("layer", sort=True)}
        vals = component_add_values(by_layer)
        return "rows", vals.assign(value=-vals.value.to_numpy())
    return "rows", _delta_rows(_members(sets, kind), -1.0)


def scaled_patch(cond: dict, sets: pd.DataFrame, w_down, v_norm: float):
    """One scaled condition, as ("rows", c × Δa) with c < 0. "push": the write's norm is |strength| × v_norm. "scaled": the
    write's norm is strength × the norm of what the top 100's change writes, for either set."""
    s = _members(sets, cond["set"])
    if cond["patch"] == "push":
        size = abs(float(cond["strength"])) * v_norm
    else:
        size = float(cond["strength"]) * write_vector(_delta_rows(_members(sets, "neurons")), w_down).norm().item()
    return "rows", _delta_rows(s, -size / write_vector(_delta_rows(s), w_down).norm().item())


def vector_patch(model, v: torch.Tensor, layer: int):
    """The authors' ActivationSteering, subtracting v at `layer` at every position."""
    from assistant_axis.steering import ActivationSteering

    v = torch.as_tensor(v).float()
    return ActivationSteering(model, steering_vectors=[v / v.norm()], layer_indices=[layer], coefficients=[-float(v.norm())])


def _vector_of(cond: dict, v: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(v) * float(cond["strength"]) if cond["set"] == "slice" else torch.as_tensor(v)


def condition_groups(conds: pd.DataFrame, inp: dict, model, chunk: int, layer: int):
    """(context, [(cond, rows payload or None)]) in the order to run them: each vector patch alone, the rest in chunks of
    `chunk` conditions."""
    batch = []
    w_down = lambda l: model.model.layers[l].mlp.down_proj.weight
    for _, cond in conds.iterrows():
        cond = cond.to_dict()
        if cond["patch"] in SCALED_PATCHES:
            kind, payload = scaled_patch(cond, inp["sets"], w_down, float(torch.as_tensor(inp["vector"]).double().norm()))
        else:
            kind, payload = patch_values(cond, inp["sets"], inp["comps"])
        if kind == "vector":
            yield vector_patch(model, _vector_of(cond, inp["vector"]), layer), [(cond, None)]
        else:
            batch.append((cond, payload))
            if len(batch) == chunk:
                yield contextlib.nullcontext(), batch
                batch = []
    if batch:
        yield contextlib.nullcontext(), batch


def write_size(cond: dict, payload, v: torch.Tensor, sets: pd.DataFrame, w_down) -> dict:
    """What a condition writes at the target layer: its norm and its component along v̂_r; for a scaled condition, also the
    multiple of Δa, which is negative."""
    v = torch.as_tensor(v).double()
    w = -_vector_of(cond, v) if cond["set"] in VECTOR_SETS else write_vector(payload, w_down)
    row = {"patch": cond["patch"], "set": cond["set"], "write_norm": float(w.norm()), "write_along_v": float(w @ (v / v.norm()))}
    if cond["patch"] in SCALED_PATCHES:
        d = _members(sets, cond["set"]).delta.to_numpy(np.float64)
        row["scale"] = float(payload.value.to_numpy() @ d / (d @ d))
    return row


def forced_scores(model, tokenizer, conversations: list[list[dict]], continuations: list[str], rows: list | None = None,
                  batch_size: int = 32) -> list[dict]:
    """Each continuation's log-probability as the assistant's reply, under an optional RowIntervention (rows[i] for
    conversation i): logp, logp_mean, n_tokens."""
    out = []
    for start in range(0, len(conversations), batch_size):
        items = []
        for conv, cont in zip(conversations[start:start + batch_size], continuations[start:start + batch_size]):
            prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                                   **_chat_kwargs(tokenizer))
            p_ids = tokenizer(prompt).input_ids
            items.append((p_ids + tokenizer(cont, add_special_tokens=False).input_ids, len(p_ids)))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        width = max(len(ids) for ids, _ in items)
        ids = torch.full((len(items), width), pad, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for i, (seq, _) in enumerate(items):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        ctx = RowIntervention(model, rows[start:start + batch_size]) if rows is not None else contextlib.nullcontext()
        with ctx, torch.no_grad():
            logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
        for i, (seq, n_p) in enumerate(items):
            lg = logits[i, n_p - 1:len(seq) - 1].float()
            logp = torch.log_softmax(lg, -1).gather(1, torch.tensor(seq[n_p:], device=lg.device)[:, None])
            out.append({"logp": logp.sum().item(), "logp_mean": logp.mean().item(), "n_tokens": len(seq) - n_p})
    return out


def residual_at(model, ids, layer: int):
    """The residual stream after `layer`, including any change made there: (tokens, d_model), in float64 on the CPU."""
    got = {}

    def hook(module, args, output):
        got["h"] = (output[0] if isinstance(output, tuple) else output)[0].detach().double().cpu()

    h = model.model.layers[layer].register_forward_hook(hook)  # registered after any patch's hook, so it sees the change
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return got["h"]


def residual_changes(model, tok, conds, inp, convs, replies: pd.DataFrame, chunk: int, layer: int) -> pd.DataFrame:
    """Each patch's measured change to the residual at `layer`, on the role prompt alone's replies teacher-forced with and
    without it. One row per condition, reply and part (prompt or reply tokens): n_tokens; mean_norm, the mean over tokens of
    the change's norm; along_v, the mean component along v̂_r."""
    vhat = torch.as_tensor(inp["vector"]).double().cpu()
    vhat = vhat / vhat.norm()
    keys = list(replies.itertuples())
    ids, n_prompt, base = [], [], []
    for r in keys:
        prompt = tok.apply_chat_template(convs[(r.question_index, r.sample)], tokenize=False, add_generation_prompt=True,
                                         **_chat_kwargs(tok))
        p_ids = tok(prompt).input_ids
        ids.append(torch.tensor([p_ids + tok(r.response, add_special_tokens=False).input_ids], device=model.device))
        n_prompt.append(len(p_ids))
        base.append(residual_at(model, ids[-1], layer))
    rows = []
    for ctx, group in condition_groups(conds[conds.patch != "prompt"], inp, model, chunk, layer):
        with ctx:
            for cond, payload in group:
                for r, i, p, b in zip(keys, ids, n_prompt, base):
                    with RowIntervention(model, [payload]) if payload is not None else contextlib.nullcontext():
                        d = residual_at(model, i, layer) - b
                    for part, x in (("prompt", d[:p]), ("reply", d[p:])):
                        if len(x):
                            rows.append({"patch": cond["patch"], "set": cond["set"], "question_index": r.question_index,
                                         "sample": r.sample, "part": part, "n_tokens": len(x),
                                         "mean_norm": float(x.norm(dim=1).mean()), "along_v": float((x @ vhat).mean())})
    return pd.DataFrame(rows)


def per_role_table(j: pd.DataFrame) -> pd.DataFrame:
    """Per role and condition, from the judged replies: the share with the persona's traits (score 2 or 3) and fully taking
    on the persona (score 3); unscored replies are left out."""
    j = j.assign(in_role=j.score.isin([2, 3]).astype(float).where(j.score.notna()),
                 fully=j.score.eq(3).astype(float).where(j.score.notna()))
    return j.groupby(["role", "patch", "set"]).agg(n=("response", "size"), n_labelled=("score", "count"),
                                                   in_role=("in_role", "mean"), fully=("fully", "mean")).reset_index()


def add_drops(per: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Each measure's drop from the role prompt alone. base: the role prompt alone's rows, indexed by role."""
    for m in MEASURES:
        per[f"drop_{m}"] = base.loc[per.role, m].to_numpy() - per[m].to_numpy()
    return per


def merge_changes(per: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Each patch's measured change, a mean over the role's replies, as columns {part}_{measure}."""
    ch = changes.groupby(["role", "patch", "set", "part"])[list(CHANGE_COLS)].mean().unstack("part")
    ch.columns = [f"{part}_{m}" for m, part in ch.columns]
    return per.merge(ch.reset_index(), on=["role", "patch", "set"], how="left")


def jzs_bf10(t: float, n: int, r: float = 0.707) -> float:
    """The JZS Bayes factor for a paired t-test (Rouder et al. 2009, eq. 1), with a Cauchy prior of scale r on the
    standardised effect."""
    from scipy import integrate

    nu = n - 1

    def integrand(g):
        a = 1 + n * g * r ** 2
        return a ** -0.5 * (1 + t ** 2 / (a * nu)) ** (-(nu + 1) / 2) * (2 * np.pi) ** -0.5 * g ** -1.5 \
            * np.exp(-1 / (2 * g))

    num = integrate.quad(integrand, 0, np.inf, limit=200)[0]
    return float(num / (1 + t ** 2 / nu) ** (-(nu + 1) / 2))


def paired_summary(a, b, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict:
    """The mean of a − b over roles, its 95% bootstrap interval, t, and the JZS Bayes factor at Cauchy scales 0.707 and
    1.0 (NaN when every role's difference is the same, or with one role)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    boot = diff[np.random.default_rng(seed).integers(0, n, (n_boot, n))].mean(1)
    sd = diff.std(ddof=1) if n > 1 else 0.0
    t = diff.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return {"n": n, "mean_diff": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "t": float(t), "seed": seed,
            "bf10_r0.707": jzs_bf10(t, n, 0.707) if sd > 0 else np.nan,
            "bf10_r1.0": jzs_bf10(t, n, 1.0) if sd > 0 else np.nan}


def _pairs_rows(patch: str, measure: str, pairs: list) -> list[dict]:
    rows = []
    for name, a, b in pairs:
        both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        rows.append({"patch": patch, "comparison": name, "measure": measure} | paired_summary(both.a, both.b))
    return rows


def paired_table(per: pd.DataFrame) -> pd.DataFrame:
    """For each measure: the role prompt alone minus each set, and each control's measure minus the top 100's."""
    base = per[per.patch == "prompt"].set_index("role")
    p = per[per.patch == "subtract"]
    rows = []
    for m in MEASURES:
        w = p.pivot(index="role", columns="set", values=m)
        rows += _pairs_rows("subtract", m, [(f"prompt alone minus {s}", base.loc[w.index, m], w[s]) for s in w.columns]
                            + [(f"{s} minus neurons", w[s], w["neurons"]) for s in ("components", "rotation_matched", "random")])
    return pd.DataFrame(rows)


def share_of_vector_drop(per: pd.DataFrame) -> pd.DataFrame:
    """Each set's drop as a share of the vector's, with a 95% bootstrap interval from the same resamples for every set."""
    d = per[per.patch == "subtract"].pivot(index="role", columns="set", values="drop_in_role").dropna()
    return pd.DataFrame(_shares("subtract", d, ("neurons", "components", "rotation_matched", "random")))


def _shares(patch: str, d: pd.DataFrame, sets) -> list[dict]:
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(d), (N_BOOT, len(d)))
    rows = []
    for s in sets:
        boot = d[s].to_numpy()[idx].mean(1) / d["vector"].to_numpy()[idx].mean(1)
        rows.append({"patch": patch, "set": s, "n": len(d), "share_of_vector_drop": d[s].mean() / d["vector"].mean(),
                     "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)), "seed": BOOT_SEED})
    return rows


def scaled_tables(per: pd.DataFrame, first: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The scaled run against the first run's per-role table: paired summaries over roles, and the drop as a share of the
    vector's subtraction's drop in the first run."""
    base = first[first.patch == "prompt"].set_index("role")
    vec = first[(first.patch == "subtract") & (first.set == "vector")].set_index("role")
    rows, ratios = [], []
    for patch in SCALED_PATCHES:
        p = per[per.patch == patch]
        for m in MEASURES:
            w = p.pivot(index="role", columns="set", values=m)
            rows += _pairs_rows(patch, m, [(f"prompt alone minus {s}", base.loc[w.index, m], w[s]) for s in w.columns] + [
                ("random minus neurons", w["random"], w["neurons"]),
                ("the vector minus neurons", vec.loc[w.index, m], w["neurons"])])
        d = p.pivot(index="role", columns="set", values="drop_in_role").join(vec.drop_in_role.rename("vector")).dropna()
        ratios += _shares(patch, d, ("neurons", "random"))
    return pd.DataFrame(rows), pd.DataFrame(ratios)


def reply_sds(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each neuron's SD over the Assistant's replies of its per-reply mean, (n_layers, n_neurons), two ways. The first, over
    all replies, is an upper bound: the statistics hold each reply's positive and negative parts squared but not their
    product, which is never positive. The second is exact, from the replies kept whole, a subsample."""
    s = stats["sums"]["missing"]
    mean = (s["pos"] + s["neg"]).double() / s["count"]
    bound = ((s["pos_sq"] + s["neg_sq"]).double() / s["count"] - mean ** 2).clamp(min=0).sqrt()
    whole = torch.stack([x[0].double() + x[1].double() for x in stats["subsample"].values()])
    return bound.numpy(), whole.std(0).numpy()


def usual_range_table(sets: pd.DataFrame, sizes: pd.DataFrame, bound: np.ndarray, sub: np.ndarray) -> pd.DataFrame:
    """One row per scaled condition and neuron: its Δa (delta), the condition's multiple, and the size of what the condition
    adds and of Δa itself, each also in SDs over the Assistant's replies (reply_sds; NaN for a neuron that does not vary)."""
    members = sets[sets.draw == 0].rename(columns={"kind": "set"})[["role", "set", "layer", "neuron", "delta"]]
    t = sizes[["role", "patch", "set", "scale"]].merge(members, on=["role", "set"])
    t["write"], t["own_size"] = (t["scale"] * t.delta).abs(), t.delta.abs()
    for name, sd in (("bound", bound), ("subsample", sub)):
        x = sd[t.layer.to_numpy(), t.neuron.to_numpy()]
        x = np.where(x > 0, x, np.nan)
        t[f"in_sd_{name}"], t[f"own_size_in_sd_{name}"] = t.write / x, t.own_size / x
    return t


# ---- Removals of the same size: the analysis of step 8b

BASE, TOP, SLICE, LOWER, LOWER_SIGN, RANDOM, VECTOR = ("no removal", "top 100 neurons", "slice of the vector", "lower-ranked neurons",
                                                       "lower-ranked neurons, same sign", "100 random neurons", "persona vector")
OFF, DIR_TOP, DIR_SLICE = "top 100, off the vector", "random direction, size of the top 100", "random direction, size of the slice"
FORCED = (BASE, TOP, SLICE, LOWER, LOWER_SIGN, RANDOM, VECTOR, OFF, DIR_TOP, DIR_SLICE)
SET_KINDS = {TOP: "neurons", LOWER: "lower", LOWER_SIGN: "lower_same_sign", RANDOM: "random"}
ROLE_REPLY, ASSISTANT_REPLY, DIFFERENCE = "role reply", "Assistant reply", "role reply minus Assistant reply"
CONDITION_OF_SET = {"neurons": TOP, "random": RANDOM, "vector": VECTOR, "slice": SLICE, "lower": LOWER, "lower_same_sign": LOWER_SIGN}


def conversation_table(forced: pd.DataFrame, alone: pd.DataFrame) -> pd.DataFrame:
    """One row per role, condition, conversation, subset and measure, value the mean per-token log-probability. The subsets
    are all conversations and those whose role reply the judge scored 2 or 3."""
    d = forced.assign(reply_kind=forced.reply_kind.map({"role": ROLE_REPLY, "assistant": ASSISTANT_REPLY}))
    w = d.pivot(index=["role", "condition", "question_index", "sample"], columns="reply_kind", values="logp_mean").reset_index()
    w.columns.name = None
    w[DIFFERENCE] = w[ROLE_REPLY] - w[ASSISTANT_REPLY]
    w = w.merge(alone[["role", "question_index", "sample", "score"]], on=["role", "question_index", "sample"], how="left")
    long = w.melt(id_vars=["role", "condition", "question_index", "sample", "score"],
                  value_vars=[ROLE_REPLY, ASSISTANT_REPLY, DIFFERENCE], var_name="measure")
    return pd.concat([long.assign(subset="all replies"), long[long.score.isin([2, 3])].assign(subset="replies in role")],
                     ignore_index=True)


def role_means(long: pd.DataFrame) -> pd.DataFrame:
    """One row per role, condition, subset and measure: the mean over the role's conversations, and its drop from no removal."""
    per = long.groupby(["role", "condition", "subset", "measure"]).agg(value=("value", "mean"), n=("value", "size")).reset_index()
    base = per[per.condition == BASE].set_index(["role", "subset", "measure"]).value
    per["drop"] = base.reindex(pd.MultiIndex.from_frame(per[["role", "subset", "measure"]])).to_numpy() - per.value
    return per


def compare_removals(w: pd.DataFrame) -> list[dict]:
    """Paired comparisons over roles; w: role by condition. "drop: X" is no removal minus X, "X minus Y" is X's drop minus Y's."""
    rows = [{"comparison": f"drop: {c}"} | paired_summary(w[BASE], w[c]) for c in FORCED if c != BASE]
    rows += [{"comparison": f"{TOP} minus {c}"} | paired_summary(w[c], w[TOP]) for c in FORCED if c not in (BASE, TOP)]
    rows.append({"comparison": f"{SLICE} minus {DIR_SLICE}"} | paired_summary(w[DIR_SLICE], w[SLICE]))
    return rows


def same_size_tables(per: pd.DataFrame, lower: pd.DataFrame, alpha: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The paired comparisons, over all roles and over those whose lower-ranked neurons reach the top 100's sum, and each
    condition's drop as a share of the vector's."""
    rows, shares = [], []
    matched = lower[(lower.kind == "lower") & lower.matched].role
    removals = list(FORCED[1:])
    i_vec = removals.index(VECTOR)
    for (subset, measure), g in per.groupby(["subset", "measure"]):
        w = g.pivot(index="role", columns="condition", values="value").dropna()
        rows += [{"subset": subset, "measure": measure, "roles": "all"} | x for x in compare_removals(w)]
        rows += [{"subset": subset, "measure": measure, "roles": "lower-ranked sum reached"} | x
                 for x in compare_removals(w[w.index.isin(matched)])]
        drops = w[BASE].to_numpy()[:, None] - w[removals].to_numpy()
        boot = drops[np.random.default_rng(BOOT_SEED).integers(0, len(w), (N_BOOT, len(w)))].mean(1)
        boot = boot / boot[:, [i_vec]]
        shares += [{"subset": subset, "measure": measure, "condition": c, "n": len(w), "share": drops[:, j].mean() / drops[:, i_vec].mean(),
                    "lo": float(np.percentile(boot[:, j], 2.5)), "hi": float(np.percentile(boot[:, j], 97.5)), "seed": BOOT_SEED,
                    "alpha_mean": float(alpha.loc[w.index].mean())} for j, c in enumerate(removals)]
    return pd.DataFrame(rows), pd.DataFrame(shares)


def size_table(sizes: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    """One row per role and condition: what it adds at layer 18 (norm, and along_v, its component along v̂_r), its movement
    along v̂_r as a fraction of |v_r|, and its drop on each measure over all replies."""
    t = sizes.copy()
    v_norm = t[t.condition == VECTOR].set_index("role").norm
    t["fraction_along_v"] = -t.along_v / v_norm.reindex(t.role).to_numpy()
    d = per[per.subset == "all replies"].pivot(index=["role", "condition"], columns="measure", values="drop")
    d.columns = [f"drop_{c}" for c in d.columns]
    return t.merge(d.reset_index(), on=["role", "condition"], how="left")


def judged_tables(judged: pd.DataFrame, first: pd.DataFrame, sizes: pd.DataFrame, changes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The judged same-size conditions beside the first run's: per role, each measure and its drop from the first run's role
    prompt alone, with sizes and measured changes; and paired summaries over roles."""
    base = first[first.patch == "prompt"].set_index("role")
    top = first[(first.patch == "subtract") & (first.set == "neurons")].set_index("role")
    per = merge_changes(add_drops(per_role_table(judged), base).merge(sizes, on=["role", "patch", "set"], how="left"), changes)
    rows = []
    for s, g in per.groupby("set"):
        g = g.set_index("role")
        for m in MEASURES:
            for name, ref in (("drop", base), ("the set's drop minus the top 100's", top)):
                pair = pd.concat([ref[m].rename("a"), g[m].rename("b")], axis=1).dropna()
                rows.append({"set": s, "comparison": name, "measure": m} | paired_summary(pair.a, pair.b))
    return per, pd.DataFrame(rows)


def _ratio(top: np.ndarray, bottom: np.ndarray, g: np.random.Generator) -> tuple[float, float, float]:
    i = g.integers(0, len(top), (N_BOOT, len(top)))
    boot = top[i].mean(1) / bottom[i].mean(1)
    return top.mean() / bottom.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sizes_measured(sizes: pd.DataFrame, first: pd.DataFrame, judged_per_role: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each condition's drop beside its size, written and measured on the reply tokens. The drop on the role's replies is
    second order in the change, so it is divided by the squared norm; the role reply minus the Assistant reply is first order,
    so by the norm. Each is a ratio of means over roles, with a 95% bootstrap interval over roles."""
    first = first[(first.patch == "subtract") & first.set.isin(["neurons", "random", "vector"])]
    measured = pd.concat([first, judged_per_role])
    measured = measured.assign(condition=measured.set.map(CONDITION_OF_SET))[["role", "condition", "reply_mean_norm", "reply_along_v"]]
    d = sizes.merge(measured, on=["role", "condition"], how="left")
    role_reply, difference = f"drop_{ROLE_REPLY}", f"drop_{DIFFERENCE}"
    g = np.random.default_rng(BOOT_SEED)
    rows = []
    for c, x in d.groupby("condition"):
        row = {"condition": c, "n": len(x), "written_norm": x.norm.mean(), "measured_norm": x.reply_mean_norm.mean(), "seed": BOOT_SEED,
               "drop_role_reply": x[role_reply].mean(), "drop_difference": x[difference].mean()}
        for label, size in (("written", x.norm.to_numpy()), ("measured", x.reply_mean_norm.to_numpy())):
            if not np.isnan(size).any():
                for out, top, bottom in (("role_reply_per_norm2", x[role_reply].to_numpy(), size ** 2),
                                         ("difference_per_norm", x[difference].to_numpy(), size)):
                    row[f"{out}_{label}"], row[f"{out}_{label}_lo"], row[f"{out}_{label}_hi"] = _ratio(top, bottom, g)
        rows.append(row)
    return d, pd.DataFrame(rows)
