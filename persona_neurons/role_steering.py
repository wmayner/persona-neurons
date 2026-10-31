"""Steering Qwen3-8B towards a role with the persona vector, the top 100 MLP neurons, the top 100 principal components,
the top 100 randomly rotated directions, or 100 random neurons, each by the same amount.

The persona vector is v_r = (role vector) − (default vector) at layer 18 over all reply tokens, as the Assistant Axis
code builds both; it is added to layer 18's output with the authors' `ActivationSteering`. The neuron and component
sets come from the split of v_r measured without massive-activation tokens (`persona_neurons.split`): neuron (l, n)
contributes c = Δa · p to |v_r|, with Δa its mean activation as the role minus as the default and
p = W_down_l[:, n] · v̂_r. The top 100 neurons are the 100 of layers 0 to 18 with the largest |c|; the top 100 principal
components are chosen the same way among every layer's components. The rotated directions are a Haar-random rotation
of each layer's neuron basis, where direction q_j contributes (q_j · Δa)(q_j · p); each layer takes its largest, as many
as the top 100 neurons have in that layer. The 100 random neurons also match the top 100 neurons' count per layer.

Neurons and directions are changed on the input of each layer's `mlp.down_proj`, at every position of the prompt and
reply: each neuron of a set is steered along its own Δa, each direction j of layer l along (V_lᵀ Δa_l)_j, scaled so
that its output to the residual stream is 4 times the length of v_r without massive-activation tokens. Each reply
draws its tokens from its own generator, seeded by role, question and sample, so a reply is paired draw for draw
across conditions whatever batch it runs in.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch

from persona_neurons.config import TARGET_LAYER

# The Assistant Axis paper's five introspective questions, asked with no system prompt
QUESTIONS = ["Who are you?", "What is your name?", "Where did you come from?", "Are you a large language model?",
             "Are you role-playing?"]
K = 100
STRENGTH = 4.0
DRAW = 0   # the one random set and the one rotated basis
SAMPLES = 2
MAX_NEW_TOKENS = 512
REPLY_SEED, RANDOM_SEED, ROTATION_SEED = 20261103, 20261104, 20261106
KINDS = ("vector", "neurons", "components", "rotation_matched", "random")
DIRECTION_SETS = ("components", "rotation_matched")


def role_indices(all_roles) -> dict[str, int]:
    """Each role's position among the sorted role names, default excluded; it seeds the role's random set and replies."""
    order = sorted(r for r in all_roles if r != "default")
    return {r: i for i, r in enumerate(order)}


# ---- Choosing the sets

def top_neurons(delta: np.ndarray, p: np.ndarray, k: int = K) -> pd.DataFrame:
    """The k neurons (delta, p: layers × neurons) with the largest |Δa · p|, largest first (ties in index order).
    Columns: layer, neuron, delta, contribution."""
    c = (np.asarray(delta, np.float64) * np.asarray(p, np.float64)).ravel()
    order = np.argsort(-np.abs(c), kind="stable")[:k]
    layer, neuron = np.divmod(order, delta.shape[1])
    return pd.DataFrame({"layer": layer, "neuron": neuron, "delta": np.asarray(delta).ravel()[order],
                         "contribution": c[order]})


def layer_matched_random(chosen: pd.DataFrame, n_neurons: int, seed) -> pd.DataFrame:
    """A random neuron set with as many neurons in each layer as `chosen` (columns layer, neuron), drawn uniformly
    without replacement within each layer. Columns: layer, neuron."""
    rng = np.random.default_rng(seed)
    rows = []
    for layer, k in chosen.layer.value_counts().sort_index().items():
        rows += [(int(layer), int(n)) for n in np.sort(rng.choice(n_neurons, size=int(k), replace=False))]
    return pd.DataFrame(rows, columns=["layer", "neuron"])


def neuron_sets(role: str, role_index: int, delta: np.ndarray, p: np.ndarray, k: int = K,
                draw: int = DRAW) -> pd.DataFrame:
    """A role's top k neurons and a random set with the same count per layer, drawn with
    np.random.default_rng([RANDOM_SEED, role_index, k, draw]). Columns: role, kind ("neurons" or "random"), draw, layer,
    neuron, delta, contribution."""
    top = top_neurons(delta, p, k)
    out = pd.concat([top[["layer", "neuron"]].assign(kind="neurons", draw=0),
                     layer_matched_random(top, delta.shape[1], [RANDOM_SEED, role_index, k, draw]).assign(
                         kind="random", draw=draw)], ignore_index=True)
    d64, p64 = np.asarray(delta, np.float64), np.asarray(p, np.float64)
    out["delta"] = d64[out.layer, out.neuron]
    out["contribution"] = out.delta * p64[out.layer, out.neuron]
    return out.assign(role=role)[["role", "kind", "draw", "layer", "neuron", "delta", "contribution"]]


def top_components(contrib: np.ndarray, k: int = K) -> pd.DataFrame:
    """The k principal components (contrib: layers × components) with the largest |contribution| over all layers,
    largest first. Columns: layer, component, contribution."""
    c = np.asarray(contrib, np.float64)
    order = np.argsort(-np.abs(c).ravel(), kind="stable")[:k]
    layer, comp = np.divmod(order, c.shape[1])
    return pd.DataFrame({"layer": layer, "component": comp, "contribution": c.ravel()[order]})


def haar_rotation(n: int, seed) -> np.ndarray:
    """A Haar-random n × n orthogonal matrix (float32): the Q factor of a standard normal matrix drawn from
    np.random.default_rng(seed), each column's sign set so that R's diagonal is positive."""
    a = np.random.default_rng(seed).standard_normal((n, n), dtype=np.float32)
    q, r = np.linalg.qr(a)
    return q * np.where(np.diag(r) < 0, -1, 1).astype(np.float32)


def top_rotated(q: np.ndarray, delta: np.ndarray, p: np.ndarray, n: int) -> pd.DataFrame:
    """The n columns of q (one layer's rotated basis) with the largest |contribution| (q_j · Δa)(q_j · p), largest
    first (ties in index order). Columns: direction, coef (q_j · Δa), contribution."""
    x = np.asarray(delta, np.float64) @ q
    c = x * (np.asarray(p, np.float64) @ q)
    order = np.argsort(-np.abs(c), kind="stable")[:n]
    return pd.DataFrame({"direction": order, "coef": x[order], "contribution": c[order]})


# ---- Steering with them

def component_add_values(by_layer: dict) -> pd.DataFrame:
    """What a set of components adds to the neurons before scaling. by_layer maps l to (V_l, coef_l): the chosen
    components as columns (n_neurons × j) and coef_l = V_lᵀ Δa_l; the addition is V_l coef_l. Columns: layer, neuron, value."""
    parts = []
    for layer, (v, coef) in sorted(by_layer.items()):
        x = np.asarray(v, np.float64) @ np.asarray(coef, np.float64)
        parts.append(pd.DataFrame({"layer": layer, "neuron": np.arange(len(x)), "value": x}))
    return pd.concat(parts, ignore_index=True)


def write_vector(values: pd.DataFrame, w_down) -> torch.Tensor:
    """What adding `values` (layer, neuron, value) to the neurons outputs to the residual stream, Σ W_down_l[:, n] ×
    value, in float64 on the CPU. w_down(layer) is that layer's down_proj weight (d_model × n_neurons)."""
    total = None
    for layer, g in values.groupby("layer", sort=True):
        w = w_down(int(layer)).detach()
        idx = torch.tensor(g.neuron.to_numpy(), dtype=torch.long, device=w.device)
        x = (w[:, idx].float() @ torch.tensor(g.value.to_numpy(), dtype=torch.float32, device=w.device)).double().cpu()
        total = x if total is None else total + x
    return total


def matched_scale(values: pd.DataFrame, w_down, target_norm: float) -> float:
    """The c for which c × values outputs a vector of norm target_norm."""
    return float(target_norm / write_vector(values, w_down).norm().item())


class RowIntervention:
    """Per-row additions to neurons while active: rows[i] is None (row i untouched) or a DataFrame with columns layer,
    neuron, value, added to the input of that layer's mlp.down_proj at every position of row i."""

    def __init__(self, model, rows: list):
        self.model, self.rows, self.handles = model, rows, []

    def _tensors(self):
        n = self.model.model.layers[0].mlp.down_proj.in_features
        dev = next(self.model.parameters()).device
        out = {}
        for i, r in enumerate(self.rows):
            if r is None or len(r) == 0:
                continue
            for layer, g in r.groupby("layer"):
                add = out.setdefault(int(layer), torch.zeros(len(self.rows), n))
                add[i, torch.tensor(g.neuron.to_numpy(), dtype=torch.long)] += torch.tensor(g.value.to_numpy(), dtype=torch.float32)
        return {l: x.to(dev) for l, x in out.items()}

    def __enter__(self):
        for layer, add in self._tensors().items():

            def hook(module, args, add=add):
                x = args[0]
                if x.shape[0] != add.shape[0]:
                    raise ValueError(f"batch of {x.shape[0]} rows under a RowIntervention of {add.shape[0]}")
                return (x + add[:, None, :].to(x.dtype),) + tuple(args[1:])
            self.handles.append(self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def conditions(role: str) -> pd.DataFrame:
    """A role's five conditions, all at STRENGTH: the vector, the top K neurons, components and rotated directions, and
    the random neurons. Columns: condition_id, role, kind, draw."""
    return pd.DataFrame([{"condition_id": f"{role}|{kind}|d{DRAW}", "role": role, "kind": kind, "draw": DRAW}
                         for kind in KINDS])


def condition_values(cond: dict, sets: pd.DataFrame, comps: dict, w_down, target_norm: float):
    """What a condition does:
    - ("vector", None) for the vector (the caller adds it at layer 18), and ("none", None) for no steering;
    - ("rows", values): the neuron values a set adds, scaled so that its output has norm target_norm.
    sets: neuron_sets rows of this role; comps[(kind, draw)] for the direction sets (components and rotation_matched):
    {"rank": DataFrame with layer and coef (each direction's projection of Δa), "v": (n_neurons, n_directions) the
    directions as columns, in the same order}. A condition with a set size k takes the sets' rows of that size."""
    kind = cond["kind"]
    if kind in ("vector", "none"):
        return kind, None
    if kind in DIRECTION_SETS:
        d = comps[(kind, int(cond["draw"]))]
        k = int(cond.get("k", K))
        top = d["rank"].head(k)
        if kind == "rotation_matched" and len(d["rank"]) > k:   # per layer, as many as the top k neurons have there
            n = sets[(sets.kind == "neurons") & (sets.k == k)].layer.value_counts()
            r = d["rank"]
            top = r[r.groupby("layer").cumcount() < r.layer.map(n).fillna(0)]
        vals = component_add_values({int(l): (d["v"][:, g.index.to_numpy()], g.coef.to_numpy())
                                     for l, g in top.groupby("layer", sort=True)})
    else:
        s = sets[(sets.kind == kind) & (sets.draw == cond["draw"])]
        s = s[s.k == cond["k"]] if "k" in cond else s
        vals = s[["layer", "neuron"]].assign(value=s.delta.to_numpy().astype(np.float64))
    c = matched_scale(vals, w_down, target_norm)
    return "rows", vals.assign(value=c * vals.value.to_numpy()).reset_index(drop=True)


# ---- Sampling

def reply_seed(role_index: int, question_index: int, sample: int, base: int = REPLY_SEED) -> int:
    """The seed of one reply's generator, from np.random.default_rng([base, role, question, sample]); the same in
    every condition."""
    return int(np.random.default_rng([base, role_index, question_index, sample]).integers(2**63))


class RowSampler:
    """A logits processor that samples each row's next token with its own torch.Generator, with the authors' sampling
    settings (temperature, top-k, top-p, in transformers' order), and returns scores that are −inf everywhere but at
    that token, so greedy decoding takes it."""

    def __init__(self, seeds, temperature: float, top_k: int, top_p: float, device):
        from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

        self.gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
        self.warpers = LogitsProcessorList([TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k),
                                            TopPLogitsWarper(top_p)])

    def __call__(self, input_ids, scores):
        probs = torch.softmax(self.warpers(input_ids, scores).float(), dim=-1)
        picks = torch.stack([torch.multinomial(probs[i], 1, generator=g) for i, g in enumerate(self.gens)])
        out = torch.full_like(scores, float("-inf"))
        return out.scatter_(1, picks.to(out.device), 0.0)


def _chat_kwargs(tokenizer) -> dict:
    return {"enable_thinking": False} if "qwen" in getattr(tokenizer, "name_or_path", "").lower() else {}


def generate_rows(model, tokenizer, conversations: list[list[dict]], seeds, max_new_tokens: int = MAX_NEW_TOKENS,
                  temperature: float = 0.7, top_p: float = 0.9, rows: list | None = None) -> list[str]:
    """Sampled replies, row i from its own generator seeded seeds[i], with the authors' settings (temperature 0.7,
    top-p 0.9; top-k from the model's generation_config.json, 20 for Qwen3-8B), the chat template with thinking off
    and prompts padded on the left. rows: an optional RowIntervention list for the batch."""
    prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, **_chat_kwargs(tokenizer))
               for c in conversations]
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = side
    sampler = RowSampler(seeds, temperature, model.generation_config.top_k or 50, top_p, model.device)
    ctx = RowIntervention(model, rows) if rows is not None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, logits_processor=[sampler],
                                 pad_token_id=tokenizer.pad_token_id)
    n = inputs.input_ids.shape[1]
    return [tokenizer.decode(o[n:], skip_special_tokens=True) for o in outputs]


def steering_groups(conds: pd.DataFrame, sets: pd.DataFrame, comps: dict, model, vector: torch.Tensor,
                    target_norm: float, chunk: int, layer: int = TARGET_LAYER):
    """Yields (context, [(cond, rows payload or None)]) in run order: the vector condition alone under the authors'
    ActivationSteering (STRENGTH × v_r at `layer`); the rest in chunks of `chunk` conditions, each with its
    RowIntervention payload."""
    from assistant_axis.steering import ActivationSteering

    w_down = lambda l: model.model.layers[l].mlp.down_proj.weight
    batch = []
    for _, cond in conds.iterrows():
        cond = cond.to_dict()
        kind, payload = condition_values(cond, sets, comps, w_down, target_norm)
        if kind == "vector":
            v = vector.float()
            yield ActivationSteering(model, steering_vectors=[v / v.norm()], layer_indices=[layer],
                                     coefficients=[STRENGTH * v.norm().item()]), [(cond, None)]
        else:
            batch.append((cond, payload))
            if len(batch) == chunk:
                yield contextlib.nullcontext(), batch
                batch = []
    if batch:
        yield contextlib.nullcontext(), batch


# ---- Step 6b's conditions: more set sizes, more draws, the two halves of the top 100, and no steering

SET_SIZES = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
MORE_DRAWS = (1, 2, 3, 4)
HALF = 50


def more_conditions(role: str) -> pd.DataFrame:
    """A role's conditions in step 6b, all at STRENGTH but the unsteered one: no steering; for each size other than K,
    the top k neurons, random set 0, and draw 0 of the top k rotated directions and principal components; draws 1 to 4
    of the 100 rotated directions and 100 random neurons; and the two 50-neuron halves of the top 100 by role-specificity.
    Columns: condition_id, role, kind, k, draw."""
    rows = [("none", 0, 0)]
    rows += [(kind, k, 0) for kind in ("neurons", "random", "rotation_matched", "components") for k in SET_SIZES if k != K]
    rows += [(kind, K, d) for kind in ("rotation_matched", "random") for d in MORE_DRAWS]
    rows += [("specific", HALF, 0), ("shared", HALF, 0)]
    return pd.DataFrame([{"condition_id": f"{role}|{kind}|k{k}|d{d}", "role": role, "kind": kind, "k": k, "draw": d}
                         for kind, k, d in rows])


def more_neuron_sets(role: str, role_index: int, delta: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """The neuron sets of more_conditions but the halves, as neuron_sets draws them. Columns: neuron_sets's and k."""
    parts = []
    for k in SET_SIZES:
        if k != K:
            parts.append(neuron_sets(role, role_index, delta, p, k).assign(k=k))
    for d in MORE_DRAWS:
        s = neuron_sets(role, role_index, delta, p, K, d)
        parts.append(s[s.kind == "random"].assign(k=K))
    return pd.concat(parts, ignore_index=True)


def specificity_halves(top: pd.DataFrame, roles: list[str], delta: np.ndarray, p: np.ndarray, norm: np.ndarray) -> pd.DataFrame:
    """Each role's top 100 split by role-specificity: the neuron's |c| for the role as a share of its summed |c| over all
    roles, with c = Δa p / |v_r|. "specific" is the 50 with the highest share and "shared" the other 50; ties go to the
    neuron's rank in the role's own list. top: role, rank, layer, neuron, for every role; delta and p: (roles, layers, neurons);
    norm: (roles,). Adds kind, k, draw, specificity and n_lists, how many roles' lists hold the neuron."""
    n = top.groupby(["layer", "neuron"]).role.size().rename("n_lists").reset_index()
    layers, neurons = n.layer.to_numpy(), n.neuron.to_numpy()
    w = np.abs(delta[:, layers, neurons].astype(np.float64) * p[:, layers, neurons] / np.asarray(norm)[:, None])   # roles × neurons
    index = {r: i for i, r in enumerate(roles)}
    column = n.assign(column=np.arange(len(n)))[["layer", "neuron", "column", "n_lists"]]
    t = top.merge(column, on=["layer", "neuron"])
    t["specificity"] = w[t.role.map(index), t.column] / w.sum(0)[t.column]
    t = t.drop(columns="column").sort_values(["role", "specificity", "rank"], ascending=[True, False, True], kind="stable")
    half = (t.groupby("role").cumcount() < HALF).map({True: "specific", False: "shared"})
    return t.assign(kind=half, k=HALF, draw=0).reset_index(drop=True)
