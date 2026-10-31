"""Per-reply mean activations of every layer, over all reply tokens and with the massive-activation tokens left out.

The mean residual over all reply tokens is the authors': each batch goes through `extract_activations_batch` from the
checkout's `pipeline/2_activations.py`, so `activations/all/{role}.pt` is what their step 2 writes. Forward hooks around
that call record, over the same tokens with the same averaging (`SpanMapper.map_spans`, then the mean over assistant
turns), the mean of each MLP neuron's positive part max(a, 0) and negative part min(a, 0) (a is one coordinate of the
input to `mlp.down_proj`), each layer's `self_attn` output, the embedding output, and the residual again in float32.
Averaging is linear, so a reply's residual mean after layer L equals the embedding mean plus the attention means and the
`W_down`-mapped neuron means of layers 0 to L.

A reply token is a massive-activation token (Sun, Chen, Kolter and Liu 2024) when the largest |a| over a few neurons of
one layer is above a line: in Qwen3-8B, layer 16's neurons 2271, 1875 and 1756, with the line at 50. Each batch runs
forward twice: the first pass stops at that layer's `down_proj` input and gives those values at every token; the second
is `extract_activations_batch` with the hooks, which reduce every hooked tensor to per-reply means over all reply tokens
and with the massive-activation tokens left out.

The means are kept as float64 sums over replies per judge score bucket: with the tokens left out (plus sums of squares
of the neuron parts, and the neuron means of a seeded subsample of replies kept whole), and over all reply tokens.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

BUCKETS = ("0", "1", "2", "3", "missing")
MASSIVE_LAYER, MASSIVE_NEURONS, MASSIVE_LINE = 16, (2271, 1875, 1756), 50.0


@dataclass
class RoleActivations:
    residual_all: dict[str, torch.Tensor]
    """Key `f"{label}_p{prompt_index}_q{question_index}"` to the authors' per-reply mean residual over all reply
    tokens, (n_layers, d_model), in their dtype."""
    residual_without: dict[str, torch.Tensor]
    """Key to the reply's mean residual with the massive-activation tokens left out, in the same dtype. A reply whose
    every token is left out has none."""
    sums: dict
    """Bucket (`BUCKETS`) to float64 sums over replies of per-reply means with the massive-activation tokens left out:
    `pos`, `neg`, `pos_sq`, `neg_sq` (n_layers, n_neurons); `attn`, `resid` (n_layers, d_model); `embed` (d_model,);
    and `count`."""
    subsample: dict[str, torch.Tensor]
    """Key to the per-reply means of the positive and negative parts with the tokens left out, (2, n_layers,
    n_neurons), float16."""
    sums_all: dict
    """Bucket to the same sums over all reply tokens, without the squares."""
    replies: list[dict]
    """One row per reply: key, bucket, n_tokens, n_left_out, largest (the largest value), kept (any token left)."""
    dropped: list[str] = field(default_factory=list)
    """Keys of replies the authors' code gives no activation (no assistant span left after truncation)."""


def upstream_activations_module(checkout: Path):
    """The authors' `pipeline/2_activations.py`, loaded as a module (its file name starts with a digit)."""
    module_spec = importlib.util.spec_from_file_location("upstream_2_activations", Path(checkout) / "pipeline" / "2_activations.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def response_key(r: dict) -> str:
    return f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}"


def bucket_of(score) -> str:
    if score is None:
        return "missing"
    if score not in (0, 1, 2, 3):
        raise ValueError(f"score {score!r} is not one of 0, 1, 2, 3 or None")
    return str(int(score))


def empty_sums(n_layers: int, n_neurons: int, d_model: int, squares: bool = True) -> dict:
    def one():
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float64)
        s = {"pos": z(n_layers, n_neurons), "neg": z(n_layers, n_neurons),
             "attn": z(n_layers, d_model), "resid": z(n_layers, d_model), "embed": z(d_model), "count": 0}
        if squares:
            s |= {"pos_sq": z(n_layers, n_neurons), "neg_sq": z(n_layers, n_neurons)}
        return s

    return {b: one() for b in BUCKETS}


def accumulate_by_score(sums, keys: list[str], means: dict[str, torch.Tensor], scores: dict[str, int | None]) -> None:
    """Add each reply's means (row i of every array in `means` belongs to `keys[i]`) to its score bucket."""
    for i, k in enumerate(keys):
        s = sums[bucket_of(scores.get(k))]
        for name in ("pos", "neg", "attn", "resid", "embed"):
            s[name] += means[name][i].double()
        if "pos_sq" in s:
            s["pos_sq"] += means["pos"][i].double() ** 2
            s["neg_sq"] += means["neg"][i].double() ** 2
        s["count"] += 1


def select_subsample(keys: list[str], seed: int, n: int) -> list[str]:
    """`n` keys drawn without replacement with `np.random.default_rng(seed)` from the sorted keys, returned sorted.

    Drawing from the sorted keys makes the choice independent of file order. Roles share their keys' layout
    (`pos_p{0..4}_q{0..239}`), so one seed picks the same prompt and question indices in every role.
    """
    ordered = sorted(keys)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ordered), size=min(n, len(ordered)), replace=False)
    return [ordered[i] for i in sorted(idx)]


def response_weights(batch_spans: list[dict], truncated_lengths: list[int], seq_len: int,
                     keep: torch.Tensor | None = None) -> torch.Tensor:
    """Weights w (n_conversations, seq_len) such that w @ x is the authors' per-reply mean of x.

    Follows `SpanMapper.map_spans` (spans clipped to the truncated length, empty ones dropped, the rest in turn order)
    and `extract_activations_batch` (turns 1, 3, ... of those, each a mean over its tokens, averaged). A row of zeros
    means their code returns None for that conversation.

    With `keep` (n_conversations, seq_len, bool), each turn's mean is over its kept tokens only, and a turn with none
    kept is left out of the average over turns, as an empty span is.
    """
    n = len(truncated_lengths)
    w = torch.zeros(n, seq_len, dtype=torch.float64)
    by_conv: dict[int, list[dict]] = {}
    for span in batch_spans:
        by_conv.setdefault(span["conversation_id"], []).append(span)
    for conv_id in range(n):
        turns = []
        for span in sorted(by_conv.get(conv_id, []), key=lambda s: s["turn"]):
            start, end = span["start"], min(span["end"], truncated_lengths[conv_id])
            if start < end:
                turns.append((start, end))
        assistant = turns[1::2]
        if keep is not None:
            assistant = [(s, e, keep[conv_id, s:e]) for s, e in assistant if keep[conv_id, s:e].any()]
            for start, end, k in assistant:
                w[conv_id, start:end] += k.double() / (len(assistant) * int(k.sum()))
            continue
        for start, end in assistant:
            w[conv_id, start:end] += 1.0 / (len(assistant) * (end - start))
    return w


class _StopForward(Exception):
    pass


class _MeanHooks:
    """Forward hooks that reduce each hooked tensor (batch, seq, dim) to per-reply means under several sets of
    weights at once (n_variants, batch, seq): every mean comes out as (n_variants, batch, dim)."""

    def __init__(self, model, weights: torch.Tensor):
        self.weights = weights
        self._w_on: dict[torch.device, torch.Tensor] = {}
        self.out: dict = {}
        inner = model.model
        self.handles = [inner.embed_tokens.register_forward_hook(lambda m, i, o: self._store("embed", o))]
        for li, layer in enumerate(inner.layers):
            self.handles += [
                layer.register_forward_hook(lambda m, i, o, li=li: self._store(("resid", li), o[0] if isinstance(o, tuple) else o)),
                layer.self_attn.register_forward_hook(lambda m, i, o, li=li: self._store(("attn", li), o[0])),
                layer.mlp.down_proj.register_forward_pre_hook(lambda m, i, li=li: self._store_neurons(li, i[0])),
            ]

    def _w(self, device) -> torch.Tensor:
        if device not in self._w_on:
            self._w_on[device] = self.weights.to(device=device, dtype=torch.float32)
        return self._w_on[device]

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("vbt,btd->vbd", self._w(x.device), x.float()).cpu()

    def _store(self, name, x):
        self.out[name] = self._mean(x)

    def _store_neurons(self, li, a):
        self.out[("pos", li)] = self._mean(a.clamp(min=0))
        self.out[("neg", li)] = self._mean(a.clamp(max=0))

    def means(self, n_layers: int) -> dict[str, torch.Tensor]:
        stack = lambda name: torch.stack([self.out[(name, li)] for li in range(n_layers)], dim=2)
        return {"pos": stack("pos"), "neg": stack("neg"), "attn": stack("attn"), "resid": stack("resid"),
                "embed": self.out["embed"]}

    def remove(self):
        for h in self.handles:
            h.remove()


def massive_values(model, input_ids: torch.Tensor, mask: torch.Tensor, layer: int, neurons) -> torch.Tensor:
    """The largest |a| over `neurons` of `layer`'s `down_proj` input at every token, (batch, seq), float32 on CPU. The
    forward pass ends there."""
    store = {}

    def pre(m, i):
        store["v"] = i[0][..., list(neurons)].abs().amax(-1).float().cpu()
        raise _StopForward

    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device))
    except _StopForward:
        pass
    finally:
        h.remove()
    return store["v"]


def _length(r: dict) -> int:
    return sum(len(m["content"]) for m in r["conversation"])


def extract_role(
    model, tokenizer, responses: list[dict], scores: dict[str, int | None], batch_size: int, subsample_seed: int,
    subsample_n: int, layer: int = MASSIVE_LAYER, neurons=MASSIVE_NEURONS, line: float = MASSIVE_LINE,
    max_length: int = 2048, upstream_checkout: Path | None = None,
) -> RoleActivations:
    """One role's replies, batched in order of length (characters), as the module docstring describes. `scores` maps
    reply keys to judge scores; keys absent from it, or mapped to None, go to the bucket "missing". Tokens, padding and
    truncation are those of the authors' `ActivationExtractor.batch_conversations` (right padding with the pad token,
    an attention mask, at most `max_length` tokens). Requires the checkout on `sys.path`
    (`persona_neurons.upstream.use_upstream`); `upstream_checkout` defaults to the one `assistant_axis` came from."""
    import assistant_axis
    from assistant_axis.internals import ConversationEncoder, ProbingModel

    if upstream_checkout is None:
        upstream_checkout = Path(assistant_axis.__file__).resolve().parents[1]
    upstream = upstream_activations_module(upstream_checkout)
    pm = ProbingModel.from_existing(model, tokenizer)
    encoder = ConversationEncoder(tokenizer, pm.model_name)
    chat_kwargs = {"enable_thinking": False} if "qwen" in pm.model_name.lower() else {}
    n_layers, cfg = len(pm.get_layers()), model.config

    keys = [response_key(r) for r in responses]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate response keys")
    chosen = set(select_subsample(keys, subsample_seed, subsample_n))
    out = RoleActivations(residual_all={}, residual_without={}, subsample={}, replies=[],
                          sums=empty_sums(n_layers, cfg.intermediate_size, cfg.hidden_size),
                          sums_all=empty_sums(n_layers, cfg.intermediate_size, cfg.hidden_size, squares=False))
    order = sorted(range(len(responses)), key=lambda i: _length(responses[i]))

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch_keys = [keys[i] for i in idx]
        conversations = [responses[i]["conversation"] for i in idx]
        full_ids, spans, _ = encoder.build_batch_turn_spans(conversations, **chat_kwargs)
        seq_len = min(max_length, max(len(ids) for ids in full_ids))
        truncated = [ids[:seq_len] for ids in full_ids]
        lengths = [len(ids) for ids in truncated]
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (seq_len - len(ids)) for ids in truncated])
        mask = torch.tensor([[1] * n + [0] * (seq_len - n) for n in lengths])

        values = massive_values(model, input_ids, mask, layer, neurons)
        w_all = response_weights(spans, lengths, seq_len)
        weights = torch.stack([w_all, response_weights(spans, lengths, seq_len, keep=values <= line)])

        hooks = _MeanHooks(model, weights)
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the authors' code prints debug lines for every call
                residual = upstream.extract_activations_batch(
                    pm, conversations, layers=list(range(n_layers)), batch_size=len(conversations),
                    max_length=max_length)
        finally:
            hooks.remove()
        means = hooks.means(n_layers)
        m_all, m_without = ({name: m[v] for name, m in means.items()} for v in (0, 1))

        kept_all, kept_without = [], []
        for i, (k, act) in enumerate(zip(batch_keys, residual)):
            if (act is None) != bool((w_all[i] == 0).all()):
                raise RuntimeError(f"{k}: the authors' span selection and ours disagree")
            if act is None:
                out.dropped.append(k)
                continue
            rel = ((act.float() - m_all["resid"][i]).norm(dim=-1) / act.float().norm(dim=-1)).max()
            if rel > 0.05:  # the hooks' mean and the authors' differ only by the rounding to their dtype
                raise RuntimeError(f"{k}: the hooks' all-token mean differs from the authors' by {float(rel):.3g}")
            kept_all.append(i)
            out.residual_all[k] = act
            reply = w_all[i] > 0
            v = values[i][reply]
            kept = bool((weights[1, i] > 0).any())
            out.replies.append({"key": k, "bucket": bucket_of(scores.get(k)), "n_tokens": int(reply.sum()),
                                "n_left_out": int((v > line).sum()), "largest": float(v.max()), "kept": kept})
            if kept:
                kept_without.append(i)
                out.residual_without[k] = m_without["resid"][i].to(act.dtype)
                if k in chosen:
                    out.subsample[k] = torch.stack([m_without["pos"][i], m_without["neg"][i]]).to(torch.float16)
        pick = lambda m, rows: {n: t[rows] for n, t in m.items()}
        accumulate_by_score(out.sums_all, [batch_keys[i] for i in kept_all], pick(m_all, kept_all), scores)
        accumulate_by_score(out.sums, [batch_keys[i] for i in kept_without], pick(m_without, kept_without), scores)
    return out
