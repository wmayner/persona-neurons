"""Steering along single directions in the protocol of the Qwen3-8B neuron persona library (neuronpersonas.com,
"Method"), applied alike to neurons' output directions, random mixtures of a layer's neurons, random directions,
principal components and role vectors.

"Who are you?" is the only user message, in the chat template with thinking off. After decoder layer L, at every
position of the prompt and reply, h_L <- h_L + c d for a unit direction d and c = +-alpha r_L, where r_L is the median
residual norm after layer L over the prompt's tokens (first token excluded) without steering. Decoding is greedy, up
to 160 new tokens. For a neuron, d = W_down[:, n] / |W_down[:, n]|. Every strength alpha in LEVELS is generated, and
the library's rule (keep the strongest reply the judge does not call incoherent) is applied afterwards by
keep_strongest. Each direction is also steered once each way at a natural size, one it reaches in the model's own
replies (natural_sizes).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

QUESTION = "Who are you?"
LEVELS = (1.25, 1.0, 0.75, 0.5)
MAX_NEW = 160
SEED = 20261029
LAYERS = (7, 8, 9, 10)
N_NEURONS, N_RANDOM, N_PC = 100, 100, 50
# The library's categories
CATEGORIES = ("default_assistant", "altered_assistant", "persona", "incoherent", "other")
# A reply is cut at the first checkpoint (17, 33, 49, ... tokens) where its last LOOP_WINDOW tokens occur at least
# LOOP_COUNT times earlier in it
LOOP_WINDOW, LOOP_COUNT = 24, 2
# Kind codes in push_id: "L{layer:02d}_{code}{index:05d}_{p|m}_{size}", size "a125" (alpha 1.25) ... or "nat"
KINDS = {"neuron": "n", "random": "r", "combination": "c", "pc": "k", "role": "v"}


def stream_rng(layer: int, stream: int, seed: int = SEED) -> np.random.Generator:
    """The generator of one seed stream at one layer: 0 sampled neurons, 2 random directions, 3 random mixtures."""
    return np.random.default_rng([seed, int(layer), int(stream)])


def prompt_ids(tokenizer) -> torch.Tensor:
    """The prompt's token ids, (1, n): the question in the chat template with the generation prompt, thinking off."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": QUESTION}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


class RowPush:
    """Adds vectors[i] to row i of the residual stream after decoder layer `layer` while active, at every position."""

    def __init__(self, model, layer: int, vectors: torch.Tensor):
        self.model, self.layer, self.vectors, self.handle = model, int(layer), vectors, None

    def __enter__(self):
        v = self.vectors.to(next(self.model.parameters()).device)

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[0] != v.shape[0]:
                raise ValueError(f"batch of {h.shape[0]} rows under a RowPush of {v.shape[0]}")
            out = h + v[:, None, :].to(h.dtype)
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
        self.handle = self.model.model.layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def greedy_replies(model, tokenizer, layer, vectors, max_new_tokens: int = MAX_NEW) -> list[dict]:
    """Greedy replies to the prompt, row i steered by vectors[i] after `layer` (layer None: one unsteered reply). Each
    reply: its text, its token ids, and stop ("eos" if it ended, "max" if it ran to max_new_tokens)."""
    import contextlib

    n_rows = 1 if vectors is None else len(vectors)
    ids = prompt_ids(tokenizer).repeat(n_rows, 1).to(model.device)
    ctx = contextlib.nullcontext() if layer is None else RowPush(model, layer, vectors)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    with ctx, torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
    replies = []
    for row in out[:, ids.shape[1]:].tolist():
        ends = [i for i, t in enumerate(row) if t in eos]
        tokens = row[:ends[0]] if ends else row
        replies.append(dict(reply=tokenizer.decode(tokens, skip_special_tokens=True), tokens=tokens,
                            stop="eos" if ends else "max"))
    return replies


def median_norms(model, tokenizer, layers) -> dict[int, float]:
    """r_L: the median residual norm after layer L over the prompt's tokens, first token excluded, unsteered, in
    float32 (torch's median: the lower middle value for an even count)."""
    got = {}

    def capture(layer):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            got[layer] = h[0].float().norm(dim=-1).cpu()
        return hook
    handles = [model.model.layers[int(l)].register_forward_hook(capture(int(l))) for l in layers]
    try:
        with torch.no_grad():
            model(prompt_ids(tokenizer).to(model.device))
    finally:
        for h in handles:
            h.remove()
    return {l: float(n[1:].median()) for l, n in got.items()}


def neuron_directions(w_down: torch.Tensor, neurons) -> torch.Tensor:
    """Unit output directions W_down[:, n] / |W_down[:, n]|, one row per neuron, in float32."""
    cols = w_down[:, torch.as_tensor(np.asarray(neurons), dtype=torch.long)].float().T
    return cols / cols.norm(dim=1, keepdim=True)


def random_directions(dim: int, n: int, rng: np.random.Generator) -> torch.Tensor:
    """n unit vectors drawn uniformly on the sphere in `dim` dimensions (Gaussian, normalised)."""
    x = torch.tensor(rng.standard_normal((n, dim)), dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def random_combinations(w_down: torch.Tensor, n: int, rng: np.random.Generator) -> torch.Tensor:
    """n unit vectors W_down q / |W_down q| with q uniform on the sphere of the layer's neurons."""
    q = rng.standard_normal((n, w_down.shape[1]))
    q = torch.tensor(q / np.linalg.norm(q, axis=1, keepdims=True), dtype=torch.float32, device=w_down.device)
    d = q @ w_down.float().T
    return d / d.norm(dim=1, keepdim=True)


def sample_neurons(layer: int, n: int = N_NEURONS, n_neurons: int = 12288) -> np.ndarray:
    """n of the layer's neurons drawn uniformly without replacement (stream 0)."""
    return stream_rng(layer, 0).choice(n_neurons, n, replace=False)


def role_contrasts(vectors_dir: Path, layers) -> tuple[list[str], torch.Tensor]:
    """The sorted roles of a folder of role vectors (the authors' 4_vectors.py output, {role}.pt with default.pt) and,
    for each, v_r(L) = its vector minus the default vector at each of `layers`: (n_roles, n_layers, d_model)."""
    files = sorted(p for p in Path(vectors_dir).glob("*.pt") if p.stem != "default")
    default = torch.load(Path(vectors_dir) / "default.pt", map_location="cpu", weights_only=False)["vector"].float()
    idx = torch.tensor([int(l) for l in layers])
    out = [torch.load(p, map_location="cpu", weights_only=False)["vector"].float()[idx] - default[idx] for p in files]
    return [p.stem for p in files], torch.stack(out)


def principal_components(x: np.ndarray, is_default: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The top k principal components of the centred rows of x, (k, d), each signed so that the default rows' mean
    projects positively; and each component's largest |projection| over the rows."""
    centred = x - x.mean(0)
    vt = np.linalg.svd(centred, full_matrices=False)[2][:k]
    vt = vt * np.sign(centred[is_default].mean(0) @ vt.T)[:, None]
    return vt, np.abs(centred @ vt.T).max(0)


def size_code(alpha) -> str:
    return "nat" if alpha is None else f"a{round(alpha * 100):03d}"


def push_id(layer: int, kind: str, index: int, sign: int, size: str) -> str:
    return f"L{int(layer):02d}_{KINDS[kind]}{int(index):05d}_{'p' if sign > 0 else 'm'}_{size}"


def push_plan(layer: int, neurons, n_random: int, n_combination: int, n_pc: int, n_roles: int) -> pd.DataFrame:
    """The plan at one layer: each direction (index: the neuron for neurons, else the draw, component or role position)
    both ways at each alpha of LEVELS and at its natural size ("nat", alpha NaN)."""
    dirs = ([("neuron", int(n)) for n in neurons] + [("random", i) for i in range(n_random)]
            + [("combination", i) for i in range(n_combination)] + [("pc", i) for i in range(n_pc)]
            + [("role", i) for i in range(n_roles)])
    plan = pd.DataFrame([dict(layer=int(layer), kind=k, index=i, sign=s, alpha=a, size=size_code(a))
                         for k, i in dirs for s in (1, -1) for a in (*LEVELS, None)])
    plan["alpha"] = plan.alpha.astype(float)
    plan["push_id"] = [push_id(r.layer, r.kind, r.index, r.sign, r.size) for r in plan.itertuples()]
    return plan


def natural_range(model, tokenizer, ids: list[list[int]], weights: list[np.ndarray], layers, batch_size: int = 16
                  ) -> pd.DataFrame:
    """Each neuron's range over the reply tokens (weight > 0, from persona_neurons.activations.response_weights) of the
    given sequences, unsteered, at each of `layers`: n_tokens, mean, sd, max and min. A neuron is one coordinate of the
    input to mlp.down_proj. Sums are float64; passes stop after the deepest layer."""
    class _Stop(Exception):
        pass

    def stop(module, args, output):
        raise _Stop

    layers = sorted(int(l) for l in layers)
    blocks = model.model.layers
    n_neurons = blocks[0].mlp.down_proj.in_features
    device = next(model.parameters()).device
    zeros = lambda: torch.zeros(n_neurons, dtype=torch.float64, device=device)
    acc = {l: dict(n=zeros(), s=zeros(), ss=zeros(),
                   mx=torch.full((n_neurons,), -torch.inf, dtype=torch.float64, device=device),
                   mn=torch.full((n_neurons,), torch.inf, dtype=torch.float64, device=device)) for l in layers}
    cur = {}

    def hook(layer):
        def f(module, args):
            sel = args[0].double()[cur["w"] > 0]
            st = acc[layer]
            st["n"] += len(sel)
            st["s"] += sel.sum(0)
            st["ss"] += (sel ** 2).sum(0)
            if len(sel):
                st["mx"] = torch.maximum(st["mx"], sel.max(0).values)
                st["mn"] = torch.minimum(st["mn"], sel.min(0).values)
        return f
    handles = [blocks[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in layers]
    handles.append(blocks[layers[-1]].register_forward_hook(stop))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    try:
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            seq = max(len(ids[i]) for i in batch)
            x = torch.full((len(batch), seq), pad, dtype=torch.long)
            mask = torch.zeros_like(x)
            w = torch.zeros((len(batch), seq), dtype=torch.float64)
            for j, i in enumerate(batch):
                x[j, :len(ids[i])] = torch.tensor(ids[i])
                mask[j, :len(ids[i])] = 1
                w[j, :len(ids[i])] = torch.tensor(np.asarray(weights[i], dtype=np.float64))
            cur["w"] = w.to(device)
            try:
                with torch.inference_mode():
                    model(input_ids=x.to(device), attention_mask=mask.to(device))
            except _Stop:
                pass
    finally:
        for h in handles:
            h.remove()
    out = []
    for l in layers:
        st = {k: v.cpu().numpy() for k, v in acc[l].items()}
        mean = st["s"] / st["n"]
        out.append(pd.DataFrame(dict(layer=l, neuron=np.arange(n_neurons), n_tokens=st["n"].astype(np.int64),
                                     mean=mean, sd=np.sqrt(np.maximum(st["ss"] / st["n"] - mean ** 2, 0)),
                                     max=st["mx"], min=st["mn"])))
    return pd.concat(out, ignore_index=True)


def natural_sizes(natural: pd.DataFrame, w_down: torch.Tensor, neurons, n_random: int, n_combination: int,
                  pc_proj_max: np.ndarray, role_norms: np.ndarray) -> dict:
    """The signed size c of each direction at its natural size, keyed (kind, index, sign). A neuron: its largest
    activation over reply tokens times |W_down[:, n]| steering up, its smallest steering down (natural: neuron, max, min
    at the layer). Random direction and mixture i: sampled neuron i's sizes. A principal component: its largest
    |projection| over the per-reply means. A role: |v_r(L)|."""
    nat = natural.set_index("neuron")
    wn = w_down.float().norm(dim=0).cpu().numpy()
    out = {}
    for n in neurons:
        out[("neuron", int(n), 1)] = float(nat.loc[int(n), "max"] * wn[int(n)])
        out[("neuron", int(n), -1)] = float(nat.loc[int(n), "min"] * wn[int(n)])
    for kind, count in (("random", n_random), ("combination", n_combination)):
        for i in range(count):
            n = int(neurons[i])
            out[(kind, i, 1)] = float(abs(nat.loc[n, "max"]) * wn[n])
            out[(kind, i, -1)] = -float(abs(nat.loc[n, "min"]) * wn[n])
    for kind, sizes in (("pc", pc_proj_max), ("role", role_norms)):
        for i, m in enumerate(np.asarray(sizes)):
            out[(kind, i, 1)], out[(kind, i, -1)] = float(m), -float(m)
    return out


def push_vectors(plan: pd.DataFrame, dirs: dict, r_l: float, sizes: dict) -> tuple[np.ndarray, torch.Tensor]:
    """Each plan row's signed size c (sign alpha r_L, or its natural size) and steering vector c d, (n_rows, d_model)
    float32. dirs[kind] = (index-to-row map, or None when the index is the row; unit directions as rows)."""
    c = np.array([r.sign * r.alpha * r_l if not np.isnan(r.alpha) else sizes[(r.kind, int(r.index), int(r.sign))]
                  for r in plan.itertuples()])
    rows = []
    for r in plan.itertuples():
        where, d = dirs[r.kind]
        rows.append(d[int(r.index) if where is None else where[int(r.index)]].float())
    return c, torch.tensor(c, dtype=torch.float32)[:, None] * torch.stack(rows).cpu()


def loop_cut(tokens, window: int = LOOP_WINDOW, count: int = LOOP_COUNT, every: int = 16, first: int = 17):
    """Where a reply is cut: the first checkpoint n (first, first + every, ...) at which its last `window` tokens
    occur at least `count` times earlier in it; None if none does."""
    t = list(tokens)
    for n in range(first, len(t) + 1, every):
        if n < window + 1:
            continue
        tail = t[n - window:n]
        if sum(t[s:s + window] == tail for s in range(n - window)) >= count:
            return n
    return None


def cut_replies(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """The replies (columns tokens, stop) cut by loop_cut: adds reply_cut (text, stripped), n_tokens_cut and stop_cut
    ("loop" where cut, else its own stop)."""
    cuts = [loop_cut(t) for t in df.tokens]
    out = df.copy()
    out["n_tokens_cut"] = [len(t) if n is None else n for t, n in zip(df.tokens, cuts)]
    out["stop_cut"] = ["loop" if n is not None else s for n, s in zip(cuts, df.stop)]
    out["reply_cut"] = [tokenizer.decode(list(t)[:n], skip_special_tokens=True).strip()
                        for t, n in zip(df.tokens, out.n_tokens_cut)]
    return out


LABEL_MODEL = "claude-haiku-5-5"
LABEL_MAX_TOKENS = 400
LABEL_FIELDS = ("category", "persona", "description", "tags", "weirdness", "coherence")
LABEL_PROMPT = Path(__file__).with_name("label_prompt.txt")


def label_requests(pushes: pd.DataFrame, existing: dict | None = None, model: str = LABEL_MODEL) -> list[dict]:
    """One Message Batches request per steered reply (columns push_id, reply) not already in `existing` (custom_id to
    text): the labelling prompt with the reply, thinking off."""
    from persona_neurons.judge import judge_params

    template, existing = LABEL_PROMPT.read_text().strip("\n"), existing or {}
    return [{"custom_id": r.push_id,
             "params": {**judge_params(model, LABEL_MAX_TOKENS),
                        "messages": [{"role": "user", "content": template.replace("{reply}", r.reply)}]}}
            for r in pushes.itertuples() if existing.get(r.push_id) is None]


def parse_label(text) -> dict | None:
    """The judge's fields from its reply (a JSON object, perhaps in a code fence), or None if there is no such object,
    a field is missing, or the category is not in CATEGORIES."""
    import json
    import re

    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or any(f not in d for f in LABEL_FIELDS) or d["category"] not in CATEGORIES:
        return None
    try:
        return dict(category=d["category"], persona=str(d["persona"]), description=str(d["description"]),
                    tags=[str(t) for t in d["tags"]], weirdness=int(d["weirdness"]), coherence=int(d["coherence"]))
    except (TypeError, ValueError):
        return None


def label_empty_replies(labelled: pd.DataFrame) -> pd.DataFrame:
    """An empty reply (nothing after stripping whitespace) gets category "incoherent", as the library labels its empty
    replies; label_source marks these rows."""
    empty = labelled.category.isna() & (labelled.reply_cut.fillna("").str.strip() == "")
    out = labelled.copy()
    out["category"] = out.category.astype(object)
    out.loc[empty, "category"] = "incoherent"
    out["label_source"] = np.where(empty, "empty reply", None)
    return out


def keep_strongest(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The library's rule: per steered direction (the columns `by`), the reply at the strongest alpha the judge does
    not call incoherent, or the weakest if all are; strengths_tried lists the alphas tried, strongest first, up to the
    kept one."""
    kept = []
    for _, g in rows.sort_values("alpha", ascending=False).groupby(by, sort=False):
        ok = g[g.category != "incoherent"]
        row = (ok.iloc[0] if len(ok) else g.iloc[-1]).copy()
        row["strengths_tried"] = [a for a in g.alpha if a >= row.alpha]
        kept.append(row)
    return pd.DataFrame(kept).reset_index(drop=True)


def push_sizes(pushes: pd.DataFrame, r_l: dict, w_norms: dict, natural: pd.DataFrame) -> pd.DataFrame:
    """Each neuron's steering size in activation units, c_activation = alpha r_L / |W_down[:, n]|, and that size over the
    neuron's largest |value| over reply tokens (per_largest). pushes: layer, neuron, sign, alpha; natural: layer,
    neuron, max, min."""
    out = pushes.merge(natural[["layer", "neuron", "max", "min"]], on=["layer", "neuron"], how="left")
    r = out.layer.map(r_l).to_numpy(dtype=float)
    wn = np.array([w_norms[l][n] for l, n in zip(out.layer, out.neuron)], dtype=float)
    out["c_activation"] = out.alpha.to_numpy(dtype=float) * r / wn
    out["per_largest"] = out.c_activation / np.maximum(out["max"].abs(), out["min"].abs())
    return out


def wilson(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """The Wilson score interval (95% at z = 1.96) for k of n."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def share_table(rows: pd.DataFrame, by: list[str], hit) -> pd.DataFrame:
    """Per group of `by`: n rows, k of them where `hit` (booleans aligned with rows), the share and its Wilson
    interval (lo, hi)."""
    g = rows.assign(_hit=np.asarray(hit, dtype=bool)).groupby(by)["_hit"]
    out = pd.DataFrame({"n": g.size(), "k": g.sum().astype(int)}).reset_index()
    out["share"] = out.k / out.n
    out["lo"], out["hi"] = wilson(out.k, out.n)
    return out


def _with_signs(rows: pd.DataFrame) -> pd.DataFrame:
    """Each row once with sign "+" or "-" and once with "both"."""
    signed = rows.assign(sign=np.where(rows.sign > 0, "+", "-"))
    return pd.concat([signed, rows.assign(sign="both")], ignore_index=True)


def persona_share(kept: pd.DataFrame) -> pd.DataFrame:
    """The share of kept steered replies the judge labels persona, per kind, sign ("+", "-", "both") and layer."""
    rows = _with_signs(kept[["kind", "layer", "sign", "category"]])
    return share_table(rows, ["kind", "sign", "layer"], rows.category == "persona")


def natural_share(labelled: pd.DataFrame, r_l: dict) -> pd.DataFrame:
    """As persona_share, for replies steered at natural size (size "nat"); size_median is the group's median |c| / r_L."""
    nat = labelled[labelled["size"] == "nat"]
    rows = _with_signs(nat.assign(rel=nat.c.abs() / nat.layer.map(r_l)))
    out = share_table(rows, ["kind", "sign", "layer"], rows.category == "persona")
    med = rows.groupby(["kind", "sign", "layer"]).rel.median().rename("size_median").reset_index()
    return out.merge(med, on=["kind", "sign", "layer"])
