"""Per-reply mean activations over all reply tokens and with the massive-activation tokens left out
(`persona_neurons.activations`, `steps/3_activations.py`).

All tests run on the tiny random Qwen3 in float32 on CPU. The reference values come from a separate, unbatched forward
pass with hooks set up here, over the assistant span that the authors' `ConversationEncoder.build_turn_spans` finds:
a token is left out when the largest |a| over the chosen neurons of the chosen layer (the input to its `down_proj`)
is above the line, and every mean is over the remaining tokens of the span.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LAYER, NEURONS = 1, [3, 7, 11]

CONVERSATIONS = [
    [
        {"role": "system", "content": "You are a lighthouse keeper on a remote island."},
        {"role": "user", "content": "What do you do all day?"},
        {"role": "assistant", "content": "I trim the wick, polish the lens and log every ship that passes."},
    ],
    [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant."},
    ],
    [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Describe your favourite colour in two sentences, please."},
        {
            "role": "assistant",
            "content": "I don't see colours, but people often describe deep blue as calm. "
            "If I had to pick, I'd choose the blue of a clear evening sky.",
        },
    ],
]


def make_responses(conversations=CONVERSATIONS, label="pos"):
    return [
        {
            "system_prompt": conv[0]["content"] if conv[0]["role"] == "system" else "",
            "label": label,
            "prompt_index": i,
            "question_index": 10 + i,
            "question": next(m["content"] for m in conv if m["role"] == "user"),
            "conversation": conv,
        }
        for i, conv in enumerate(conversations)
    ]


def key(r):
    return f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}"


@pytest.fixture(scope="module")
def pm(tiny_qwen3_dir, upstream_checkout):
    from persona_neurons.upstream import use_upstream

    use_upstream(upstream_checkout)
    from assistant_axis.internals import ProbingModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ProbingModel(path) loads with device_map="auto", which would use any GPU the machine has; load on CPU.
    model = AutoModelForCausalLM.from_pretrained(tiny_qwen3_dir, dtype=torch.float32).eval()
    pm = ProbingModel.from_existing(model, AutoTokenizer.from_pretrained(tiny_qwen3_dir))
    assert "qwen" in pm.model_name.lower() and pm.device.type == "cpu"
    return pm


@pytest.fixture(scope="module")
def upstream_activations(upstream_checkout):
    from persona_neurons.upstream import use_upstream

    use_upstream(upstream_checkout)
    module_spec = importlib.util.spec_from_file_location("upstream_2_activations", upstream_checkout / "pipeline" / "2_activations.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def reference(pm, conversation):
    """Per-token values over the assistant span of one conversation, from an unbatched float32 forward pass."""
    from assistant_axis.internals import ConversationEncoder

    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    full_ids, spans = encoder.build_turn_spans(conversation, enable_thinking=False)
    (span,) = [s for s in spans if s["role"] == "assistant"]
    sl = slice(span["start"], span["end"])
    model = pm.model
    layers = model.model.layers
    cap = {}
    put = cap.__setitem__
    handles = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: put("embed", o))]
    for li, layer in enumerate(layers):
        handles += [
            layer.register_forward_hook(lambda m, i, o, li=li: put(("resid", li), o[0] if isinstance(o, tuple) else o)),
            layer.self_attn.register_forward_hook(lambda m, i, o, li=li: put(("attn", li), o[0])),
            layer.mlp.down_proj.register_forward_pre_hook(lambda m, i, li=li: put(("a", li), i[0])),
            layer.mlp.register_forward_hook(lambda m, i, o, li=li: put(("mlp", li), o)),
        ]
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([full_ids]))
    finally:
        for h in handles:
            h.remove()
    n = len(layers)
    tok = lambda t: t[0, sl].double()
    out = {
        "ids": full_ids[sl],
        "embed": tok(cap["embed"]),
        "resid": torch.stack([tok(cap[("resid", li)]) for li in range(n)], 1),
        "attn": torch.stack([tok(cap[("attn", li)]) for li in range(n)], 1),
        "a": torch.stack([tok(cap[("a", li)]) for li in range(n)], 1),
        "mlp": torch.stack([tok(cap[("mlp", li)]) for li in range(n)], 1),
    }
    out["value"] = out["a"][:, LAYER, NEURONS].abs().amax(-1)
    return out


def reference_means(ref, line):
    keep = ref["value"] <= line
    m = lambda t: t[keep].mean(0)
    return {"embed": m(ref["embed"]), "resid": m(ref["resid"]), "attn": m(ref["attn"]), "mlp": m(ref["mlp"]),
            "pos": m(ref["a"].clamp(min=0)), "neg": m(ref["a"].clamp(max=0)), "n_left_out": int((~keep).sum())}


@pytest.fixture(scope="module")
def refs(pm):
    return [reference(pm, c) for c in CONVERSATIONS]


@pytest.fixture(scope="module")
def line(refs):
    """A line that leaves out some tokens of every reply but not all of them."""
    v = torch.cat([r["value"] for r in refs])
    return float(v.quantile(0.7))


@pytest.fixture(scope="module")
def run(pm, line):
    """extract_role with each reply alone in its score bucket, so each bucket's sums are that reply's means."""
    from persona_neurons.activations import extract_role

    responses = make_responses()
    scores = {key(responses[0]): 3, key(responses[1]): 1}  # the third has no score: bucket "missing"
    out = extract_role(pm.model, pm.tokenizer, responses, scores, batch_size=2, subsample_seed=0, subsample_n=3,
                       layer=LAYER, neurons=NEURONS, line=line)
    buckets = {key(responses[0]): "3", key(responses[1]): "1", key(responses[2]): "missing"}
    return responses, out, buckets


def test_left_out_tokens_are_some_but_not_all(refs, line):
    for ref in refs:
        assert 0 < reference_means(ref, line)["n_left_out"] < len(ref["ids"])


def test_residual_all_is_the_authors(pm, upstream_activations, run):
    responses, out, _ = run
    n_layers = len(pm.model.model.layers)
    expected = upstream_activations.extract_activations_batch(
        pm, [r["conversation"] for r in responses], layers=list(range(n_layers)), batch_size=2)
    assert set(out.residual_all) == {key(r) for r in responses}
    for r, exp in zip(responses, expected):
        got = out.residual_all[key(r)]
        assert got.shape == exp.shape == (n_layers, pm.model.config.hidden_size)
        assert got.dtype == exp.dtype
        assert (got.float() - exp.float()).abs().max() < 1e-4


def test_residual_without(pm, upstream_activations, run, refs, line):
    responses, out, _ = run
    for r, ref in zip(responses, refs):
        got = out.residual_without[key(r)]
        assert got.dtype == out.residual_all[key(r)].dtype
        exp = reference_means(ref, line)["resid"]
        assert ((got.double() - exp).abs() <= exp.abs() / 128 + 1e-5).all()


def test_response_weights_keep():
    from persona_neurons.activations import response_weights

    spans = [{"conversation_id": 0, "turn": 0, "start": 0, "end": 3}, {"conversation_id": 0, "turn": 1, "start": 3, "end": 7},
             {"conversation_id": 1, "turn": 0, "start": 0, "end": 2}, {"conversation_id": 1, "turn": 1, "start": 2, "end": 4}]
    keep = torch.ones(2, 8, dtype=torch.bool)
    keep[0, 4] = False
    keep[1, 2:4] = False
    w = response_weights(spans, [8, 8], 8, keep=keep)
    assert torch.allclose(w[0], torch.tensor([0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0], dtype=torch.float64))
    assert (w[1] == 0).all()  # every reply token left out: no mean, as the authors' code gives none for an empty span
    assert torch.equal(response_weights(spans, [8, 8], 8), response_weights(spans, [8, 8], 8, keep=None))


@pytest.mark.parametrize("version", ["sums", "sums_all"])
def test_sums_match_reference(pm, run, refs, line, version):
    responses, out, buckets = run
    for r, ref in zip(responses, refs):
        exp = reference_means(ref, line if version == "sums" else float("inf"))
        s = getattr(out, version)[buckets[key(r)]]
        assert s["count"] == 1
        for name in ("embed", "resid", "attn", "pos", "neg"):
            assert (s[name] - exp[name]).abs().max() < 1e-4, name
        assert ("pos_sq" in s) == (version == "sums")
        if version == "sums":
            assert (s["pos_sq"] - exp["pos"] ** 2).abs().max() < 1e-4
        for li, layer in enumerate(pm.model.model.layers):  # the neuron means rebuild each MLP's mean output
            assert (s["pos"][li] >= 0).all() and (s["neg"][li] <= 0).all()
            mlp = layer.mlp.down_proj.weight.double() @ (s["pos"][li] + s["neg"][li])
            assert (mlp - exp["mlp"][li]).abs().max() < 1e-4


def test_residual_decomposition_exact(pm, run, refs, line):
    responses, out, buckets = run
    layers = pm.model.model.layers
    for r, ref in zip(responses, refs):
        s = out.sums[buckets[key(r)]]
        total = s["embed"].clone()
        for li in range(len(layers)):
            total = total + s["attn"][li] + layers[li].mlp.down_proj.weight.double() @ (s["pos"][li] + s["neg"][li])
            assert (total - s["resid"][li]).abs().max() < 1e-3


def test_sums_by_score():
    from persona_neurons.activations import BUCKETS, accumulate_by_score, empty_sums

    rng = np.random.default_rng(1)
    L, N, D = 3, 5, 4
    keys = ["pos_p0_q0", "pos_p0_q1", "pos_p1_q0", "pos_p1_q1"]
    scores = {"pos_p0_q0": 3, "pos_p0_q1": 3, "pos_p1_q0": 2, "pos_p1_q1": None}
    a = torch.tensor(rng.normal(size=(4, L, N)))
    means = {"pos": a.clamp(min=0), "neg": a.clamp(max=0), "attn": torch.tensor(rng.normal(size=(4, L, D))),
             "resid": torch.tensor(rng.normal(size=(4, L, D))), "embed": torch.tensor(rng.normal(size=(4, D)))}
    sums = empty_sums(L, N, D)
    accumulate_by_score(sums, keys, means, scores)
    assert set(sums) == set(BUCKETS)
    for bucket, idx in {"3": [0, 1], "2": [2], "missing": [3], "1": [], "0": []}.items():
        s = sums[bucket]
        assert s["count"] == len(idx)
        for name in ("pos", "neg", "attn", "resid", "embed"):
            assert s[name].dtype == torch.float64 and torch.equal(s[name], means[name][idx].sum(0))
        assert torch.equal(s["pos_sq"], (means["pos"][idx] ** 2).sum(0))


def test_subsample_deterministic():
    from persona_neurons.activations import select_subsample

    keys = [f"pos_p{p}_q{q}" for p in range(5) for q in range(240)]
    a = select_subsample(keys, seed=20261027, n=48)
    assert a == select_subsample(list(reversed(keys)), seed=20261027, n=48)  # independent of file order
    assert len(a) == len(set(a)) == 48 and set(a) <= set(keys)
    assert a != select_subsample(keys, seed=20261028, n=48)


def test_subsample_values(pm, run):
    responses, out, buckets = run
    assert sorted(out.subsample) == sorted(key(r) for r in responses)
    for k, v in out.subsample.items():
        assert v.shape == (2, len(pm.model.model.layers), pm.model.config.intermediate_size) and v.dtype == torch.float16
        s = out.sums[buckets[k]]
        assert torch.allclose(v[0].double(), s["pos"], atol=1e-3, rtol=1e-3)
        assert torch.allclose(v[1].double(), s["neg"], atol=1e-3, rtol=1e-3)


def test_reply_rows(run, refs, line):
    responses, out, buckets = run
    rows = {row["key"]: row for row in out.replies}
    assert set(rows) == {key(r) for r in responses}
    for r, ref in zip(responses, refs):
        row = rows[key(r)]
        assert row["bucket"] == buckets[key(r)] and row["kept"]
        assert row["n_tokens"] == len(ref["ids"])
        assert row["n_left_out"] == int((ref["value"] > line).sum())
        assert row["largest"] == pytest.approx(float(ref["value"].max()), rel=1e-5)


def test_first_pass_stops_after_layer(pm):
    from persona_neurons.activations import massive_values

    called = []
    h = pm.model.model.layers[1].register_forward_hook(lambda m, i, o: called.append(1))
    try:
        v = massive_values(pm.model, torch.tensor([[1, 2, 3, 4]]), torch.ones(1, 4, dtype=torch.long), 0, NEURONS)
    finally:
        h.remove()
    assert v.shape == (1, 4) and not called


def write_role(data, role, label="pos", scores=None):
    (data / "responses").mkdir(parents=True, exist_ok=True)
    (data / "scores").mkdir(exist_ok=True)
    responses = make_responses(label=label)
    with jsonlines.open(data / "responses" / f"{role}.jsonl", "w") as w:
        w.write_all(responses)
    if scores is not None:
        (data / "scores" / f"{role}.json").write_text(json.dumps({key(r): s for r, s in zip(responses, scores)}))
    return responses


def run_step(data, tiny_qwen3_dir, upstream_checkout, *extra):
    cmd = [sys.executable, str(ROOT / "steps" / "3_activations.py"), "--model-path", str(tiny_qwen3_dir),
           "--batch-size", "2", "--subsample-n", "2", "--dtype", "float32", "--layer", "1", "--neurons", "3", "7", "11",
           "--line", "1e9", *extra]
    # the step loads with device_map="auto", as the authors' code does
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PN_DATA_DIR": str(data), "PN_UPSTREAM_DIR": str(upstream_checkout)}
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)


def test_step(tmp_path, tiny_qwen3_dir, upstream_checkout):
    data = tmp_path / "data"
    responses = write_role(data, "pirate", scores=[3, 2, 1])
    write_role(data, "default", label="default")
    result = run_step(data, tiny_qwen3_dir, upstream_checkout)
    assert result.returncode == 0, result.stderr
    for version in ("all", "without"):  # the authors' 4_vectors.py reads every *.pt in these folders
        assert sorted(p.name for p in (data / "activations" / version).iterdir()) == ["default.pt", "pirate.pt"]
        residual = torch.load(data / "activations" / version / "pirate.pt", weights_only=False)
        assert sorted(residual) == sorted(key(r) for r in responses)
    stats = torch.load(data / "neuron_stats" / "pirate.pt", weights_only=False)
    assert stats["scored"] is True and stats["line"] == 1e9 and stats["neurons"] == [3, 7, 11]
    assert {b: stats["sums"][b]["count"] for b in ("1", "2", "3")} == {"1": 1, "2": 1, "3": 1}
    assert stats["sums_all"]["3"]["count"] == 1 and len(stats["subsample"]) == 2
    default = torch.load(data / "neuron_stats" / "default.pt", weights_only=False)
    assert default["scored"] is False and default["sums"]["missing"]["count"] == 3
    rows = pd.read_parquet(data / "replies" / "pirate.parquet")
    assert len(rows) == 3 and sorted(rows.score) == [1, 2, 3]

    mtime = (data / "activations" / "all" / "pirate.pt").stat().st_mtime_ns
    again = run_step(data, tiny_qwen3_dir, upstream_checkout)
    assert again.returncode == 0 and (data / "activations" / "all" / "pirate.pt").stat().st_mtime_ns == mtime


def test_step_shards_and_refuses_unscored(tmp_path, tiny_qwen3_dir, upstream_checkout):
    data = tmp_path / "data"
    write_role(data, "pirate")  # no scores
    write_role(data, "default", label="default")
    result = run_step(data, tiny_qwen3_dir, upstream_checkout, "--shard", "0/2")  # sorted: default, pirate
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (data / "neuron_stats").iterdir()) == ["default.pt"]
    result = run_step(data, tiny_qwen3_dir, upstream_checkout, "--shard", "1/2")
    assert result.returncode == 1 and "pirate" in result.stderr
