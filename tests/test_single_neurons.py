"""Steering along single directions, and their labels (persona_neurons.single_neurons)."""
import numpy as np
import pandas as pd
import pytest
import torch

from persona_neurons import single_neurons as sn


def _tiny(upstream_checkout, tiny_qwen3_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from persona_neurons.upstream import use_upstream

    use_upstream(upstream_checkout)
    model = AutoModelForCausalLM.from_pretrained(tiny_qwen3_dir, torch_dtype=torch.float32).eval()
    return model, AutoTokenizer.from_pretrained(tiny_qwen3_dir)


def _prompt(model, tok, n=1):
    return sn.prompt_ids(tok).repeat(n, 1).to(model.device)


class _AddToNeuron:
    """Adds value to neuron n's activation (the input of mlp.down_proj) of batch row 0 at layer `layer`."""

    def __init__(self, model, layer, n, value):
        self.mlp, self.n, self.value = model.model.layers[layer].mlp, n, value

    def __enter__(self):
        def pre(module, args):
            a = args[0].clone()
            a[0, :, self.n] += self.value
            return (a,)
        self.handle = self.mlp.down_proj.register_forward_pre_hook(pre)

    def __exit__(self, *exc):
        self.handle.remove()


def test_prompt_is_who_are_you_in_the_chat_template_with_thinking_off(upstream_checkout, tiny_qwen3_dir):
    _, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    text = tok.decode(sn.prompt_ids(tok)[0])
    assert text.startswith("<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n")
    assert text.endswith("<think>\n\n</think>\n\n")


def test_row_push_along_a_neuron_equals_adding_to_its_activation(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    w = model.model.layers[0].mlp.down_proj.weight.detach()
    n, c = 5, 3.0
    d = w[:, n] / w[:, n].norm()
    ids = _prompt(model, tok, 2)
    with torch.no_grad(), sn.RowPush(model, 0, torch.stack([c * d, torch.zeros_like(d)])):
        steered = model(ids).logits
    with torch.no_grad(), _AddToNeuron(model, 0, n, c / w[:, n].norm().item()):
        want = model(ids).logits
    with torch.no_grad():
        plain = model(ids).logits
    torch.testing.assert_close(steered, want, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(steered[1], plain[1], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(steered[0], plain[0], atol=1e-3)


def test_row_push_acts_at_every_position(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    ids = _prompt(model, tok, 1)
    v = torch.randn(1, model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    def layer_output(push):
        got = {}

        def cap(module, args, output):
            got["h"] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        with torch.no_grad(), push:
            h = model.model.layers[0].register_forward_hook(cap)  # after the steering hook, so it sees the steered output
            try:
                model(ids)
            finally:
                h.remove()
        return got["h"]
    plain = layer_output(sn.RowPush(model, 0, torch.zeros_like(v)))
    steered = layer_output(sn.RowPush(model, 0, v))
    torch.testing.assert_close(steered - plain, v[:, None, :].expand_as(plain), rtol=1e-4, atol=1e-4)


def test_greedy_replies_row_by_row_equal_the_batch(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    vs = torch.randn(3, model.config.hidden_size, generator=torch.Generator().manual_seed(1)) * 2.0
    vs[1] = 0
    batch = sn.greedy_replies(model, tok, 0, vs, max_new_tokens=12)
    alone = [sn.greedy_replies(model, tok, 0, vs[i:i + 1], max_new_tokens=12)[0] for i in range(3)]
    assert [b["tokens"] for b in batch] == [a["tokens"] for a in alone]
    assert batch[1]["tokens"] == sn.greedy_replies(model, tok, None, None, max_new_tokens=12)[0]["tokens"]
    for b in batch:
        assert set(b) == {"reply", "tokens", "stop"} and b["stop"] in ("eos", "max") and len(b["tokens"]) <= 12
        assert b["reply"] == tok.decode(b["tokens"], skip_special_tokens=True)


def test_median_norm_is_over_the_prompt_without_its_first_token(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    with torch.no_grad():
        hs = model(_prompt(model, tok, 1), output_hidden_states=True).hidden_states
    want = hs[1][0, 1:].float().norm(dim=-1).median().item()
    assert sn.median_norms(model, tok, [0])[0] == pytest.approx(want, rel=1e-5)


def test_directions_are_unit_seeded_and_of_the_right_kind():
    w = torch.tensor(np.random.default_rng(0).standard_normal((16, 40)), dtype=torch.float32)
    nd = sn.neuron_directions(w, [3, 7])
    torch.testing.assert_close(nd[0], w[:, 3] / w[:, 3].norm())
    rd = sn.random_directions(16, 5, sn.stream_rng(layer=4, stream=2))
    torch.testing.assert_close(rd, sn.random_directions(16, 5, sn.stream_rng(layer=4, stream=2)))
    assert not torch.allclose(rd, sn.random_directions(16, 5, sn.stream_rng(layer=5, stream=2)))
    rc = sn.random_combinations(w, 5, sn.stream_rng(layer=4, stream=3))
    for d in (nd, rd, rc):
        torch.testing.assert_close(d.norm(dim=1), torch.ones(len(d)))
    # a random mixture is w q / |w q| for q uniform on the sphere of the neurons
    q = sn.stream_rng(layer=4, stream=3).standard_normal((5, 40))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    want = torch.tensor(q, dtype=torch.float32) @ w.T
    torch.testing.assert_close(rc, want / want.norm(dim=1, keepdim=True), rtol=1e-5, atol=1e-5)


def test_sample_neurons_is_seeded_per_layer_and_without_repeats():
    a = sn.sample_neurons(7, n=10, n_neurons=40)
    assert len(set(a)) == 10 and all(0 <= n < 40 for n in a)
    assert np.array_equal(a, sn.sample_neurons(7, n=10, n_neurons=40))
    assert not np.array_equal(a, sn.sample_neurons(8, n=10, n_neurons=40))


def test_keep_strongest_keeps_the_strongest_coherent_reply():
    rows = pd.DataFrame(dict(push=["a"] * 4 + ["b"] * 4, alpha=[1.25, 1.0, 0.75, 0.5] * 2,
                             category=["incoherent", "persona", "default_assistant", "persona"] + ["incoherent"] * 4))
    kept = sn.keep_strongest(rows, by=["push"])
    a = kept[kept.push == "a"].iloc[0]
    assert a.alpha == 1.0 and a.category == "persona" and list(a.strengths_tried) == [1.25, 1.0]
    b = kept[kept.push == "b"].iloc[0]
    assert b.alpha == 0.5 and b.category == "incoherent" and len(kept) == 2


def test_loop_cut_at_the_first_checkpoint_where_the_tail_was_seen_before():
    # 5 distinct tokens, then a 6-token block repeated: the last 8 tokens first occur earlier at length 5 + 6 + 8
    t = list(range(100, 105)) + [1, 2, 3, 4, 5, 6] * 10
    assert sn.loop_cut(t, window=8, count=1, every=1, first=1) == 19
    assert sn.loop_cut(t, window=8, count=1) == 33  # checked only at 17, 33, ...
    assert sn.loop_cut(t, window=8, count=2, every=1, first=1) == 25
    assert sn.loop_cut(list(range(200)), window=8, count=1) is None
    assert sn.loop_cut(t[:18], window=8, count=1, every=1, first=1) is None


class _WordTok:
    def decode(self, ids, skip_special_tokens=True):
        return " " + " ".join(map(str, ids)) + "\n"


def test_cut_replies_applies_the_loop_rule_marks_the_stop_and_strips():
    looping = list(range(100, 140)) + list(range(1, 25)) * 4  # the last 24 tokens seen twice before by token 137
    df = pd.DataFrame(dict(push_id=["a", "b"], tokens=[looping, [7, 8]], stop=["max", "eos"]))
    out = sn.cut_replies(df, _WordTok())
    n = sn.loop_cut(looping)
    assert n is not None and list(out.n_tokens_cut) == [n, 2]
    assert list(out.stop_cut) == ["loop", "eos"]
    assert out.reply_cut.iloc[1] == "7 8"


def test_label_requests_fill_the_prompt_with_the_reply_alone():
    pushes = pd.DataFrame(dict(push_id=["L08_n00012_p_a125", "L08_r00003_m_a050"], reply=["I am a pirate.", "Arr"]))
    reqs = sn.label_requests(pushes)
    assert [r["custom_id"] for r in reqs] == list(pushes.push_id)
    p = reqs[0]["params"]
    assert p["model"] == "claude-haiku-5-5" and p["thinking"] == {"type": "disabled"}
    text = p["messages"][0]["content"]
    assert "<reply>\nI am a pirate.\n</reply>" in text and "{reply}" not in text
    assert text.startswith('A language model was asked "Who are you?"')
    for c in sn.CATEGORIES:
        assert f"- {c}:" in text
    assert len(sn.label_requests(pushes, existing={"L08_n00012_p_a125": "{}"})) == 1


def test_parse_label_reads_the_json_and_refuses_anything_else():
    good = ('```json\n{"category": "persona", "persona": "Pirate", "description": "A pirate.", "tags": ["sea"], '
            '"weirdness": 7, "coherence": 9}\n```')
    assert sn.parse_label(good) == dict(category="persona", persona="Pirate", description="A pirate.", tags=["sea"],
                                        weirdness=7, coherence=9)
    assert sn.parse_label('{"category": "pirate", "persona": "x", "description": "", "tags": [], "weirdness": 1, '
                          '"coherence": 1}') is None
    assert sn.parse_label("I cannot help with that.") is None
    assert sn.parse_label(None) is None
    assert sn.parse_label('{"category": "persona", "weirdness": 3}') is None


def test_empty_replies_are_labelled_incoherent():
    df = pd.DataFrame(dict(reply_cut=["", "  \n", "Hello", ""], category=[None, None, None, "persona"]))
    out = sn.label_empty_replies(df)
    assert out.category.fillna("-").tolist() == ["incoherent", "incoherent", "-", "persona"]
    assert out.label_source.fillna("-").tolist() == ["empty reply", "empty reply", "-", "-"]


def test_push_ids_are_valid_batch_ids():
    assert sn.push_id(layer=8, kind="role", index=274, sign=-1, size="a075") == "L08_v00274_m_a075"


def test_natural_range_matches_the_activations_on_reply_tokens(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    g = np.random.default_rng(0)
    ids = [list(g.integers(0, 1000, size=n)) for n in (9, 14, 6)]
    weights = [np.r_[np.zeros(3), np.full(n - 3, 1 / (n - 3))] for n in (9, 14, 6)]
    stats = sn.natural_range(model, tok, ids, weights, layers=[0, 1], batch_size=2)
    n_neurons = model.config.intermediate_size
    assert len(stats) == 2 * n_neurons
    got = []
    h = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda m, a: got.append(a[0][0].double()))
    try:
        with torch.no_grad():
            for x in ids:
                model(torch.tensor([x]))
    finally:
        h.remove()
    reply = np.concatenate([a.numpy()[w > 0] for a, w in zip(got, weights)])
    s = stats[stats.layer == 1].set_index("neuron").sort_index()
    np.testing.assert_allclose(s["max"], reply.max(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(s["min"], reply.min(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(s["sd"], reply.std(0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(s["mean"], reply.mean(0), rtol=1e-4, atol=1e-6)
    assert (s.n_tokens == len(reply)).all()


def test_role_contrasts_are_role_minus_default_at_each_layer(tmp_path):
    g = torch.Generator().manual_seed(0)
    vecs = {r: torch.randn(36, 8, generator=g) for r in ["default", "pirate", "angel"]}
    for r, v in vecs.items():
        torch.save({"vector": v, "type": "mean" if r == "default" else "pos_3", "role": r}, tmp_path / f"{r}.pt")
    roles, v = sn.role_contrasts(tmp_path, layers=[3, 18])
    assert roles == ["angel", "pirate"] and v.shape == (2, 2, 8)
    torch.testing.assert_close(v[1, 0], vecs["pirate"][3] - vecs["default"][3])
    torch.testing.assert_close(v[0, 1], vecs["angel"][18] - vecs["default"][18])


def test_principal_components_point_towards_the_default_replies():
    x = np.random.default_rng(0).standard_normal((200, 6)) * np.array([5, 3, 1, 1, 1, 1])
    is_default = np.zeros(200, bool)
    is_default[:20] = True
    x[is_default, 0] -= 4  # the default replies sit on the negative side of the first axis
    pcs, proj_max = sn.principal_components(x, is_default, k=3)
    np.testing.assert_allclose(pcs @ pcs.T, np.eye(3), atol=1e-6)
    centred = x - x.mean(0)
    assert (centred[is_default].mean(0) @ pcs.T > 0).all()
    assert abs(pcs[0, 0]) > 0.9 and pcs[0, 0] < 0
    np.testing.assert_allclose(proj_max, np.abs(centred @ pcs.T).max(0))
    want = np.linalg.svd(centred, full_matrices=False)[2][:3]
    np.testing.assert_allclose(np.abs(pcs @ want.T), np.eye(3), atol=1e-6)


def test_push_plan_has_every_direction_both_ways_at_every_size():
    plan = sn.push_plan(layer=8, neurons=[1, 2, 3], n_random=2, n_combination=2, n_pc=1, n_roles=3)
    assert len(plan) == (3 + 2 + 2 + 1 + 3) * 2 * 5 and plan.push_id.is_unique
    assert set(plan["size"]) == {"a125", "a100", "a075", "a050", "nat"}
    assert plan[plan["size"] == "nat"].alpha.isna().all()
    assert set(plan.kind) == set(sn.KINDS) and (plan.layer == 8).all()
    assert sorted(plan[plan.kind == "neuron"]["index"].unique()) == [1, 2, 3]


def test_push_vectors_scale_each_direction_by_its_size():
    plan = sn.push_plan(layer=3, neurons=[0, 2], n_random=2, n_combination=2, n_pc=1, n_roles=1)
    g = torch.Generator().manual_seed(0)
    w = torch.randn(6, 5, generator=g)
    unit = lambda x: x / x.norm(dim=-1, keepdim=True)
    role = torch.randn(1, 6, generator=g)
    dirs = {"neuron": ({0: 0, 2: 1}, sn.neuron_directions(w, [0, 2])),
            "random": (None, unit(torch.randn(2, 6, generator=g))),
            "combination": (None, unit(torch.randn(2, 6, generator=g))),
            "pc": (None, unit(torch.randn(1, 6, generator=g))), "role": (None, unit(role))}
    natural = pd.DataFrame(dict(neuron=[0, 2], max=[2.0, 0.5], min=[-1.0, -3.0]))
    sizes = sn.natural_sizes(natural, w, neurons=[0, 2], n_random=2, n_combination=2, pc_proj_max=np.array([7.0]),
                             role_norms=role.norm(dim=1).numpy())
    c, v = sn.push_vectors(plan, dirs, r_l=10.0, sizes=sizes)
    c = pd.Series(c, index=plan.push_id)
    wn = w.norm(dim=0)
    assert c["L03_n00002_m_a125"] == pytest.approx(-12.5)
    assert c["L03_n00000_p_nat"] == pytest.approx(2.0 * wn[0].item(), rel=1e-6)
    assert c["L03_n00000_m_nat"] == pytest.approx(-1.0 * wn[0].item(), rel=1e-6)
    # random draw i is paired with sampled neuron i's natural sizes, signed by the row
    assert c["L03_r00001_p_nat"] == pytest.approx(0.5 * wn[2].item(), rel=1e-6)
    assert c["L03_c00001_m_nat"] == pytest.approx(-3.0 * wn[2].item(), rel=1e-6)
    assert c["L03_k00000_m_nat"] == pytest.approx(-7.0)
    assert c["L03_v00000_p_nat"] == pytest.approx(role.norm().item(), rel=1e-6)
    i = list(plan.push_id).index("L03_n00000_p_nat")
    torch.testing.assert_close(v[i], 2.0 * w[:, 0], rtol=1e-5, atol=1e-6)
    j = list(plan.push_id).index("L03_v00000_m_a050")
    torch.testing.assert_close(v[j], -5.0 * unit(role)[0], rtol=1e-5, atol=1e-6)


def test_push_sizes_in_units_of_the_neurons_largest_value():
    pushes = pd.DataFrame(dict(layer=[8, 8], neuron=[0, 1], sign=[1, -1], alpha=[1.0, 0.5]))
    natural = pd.DataFrame(dict(layer=[8, 8], neuron=[0, 1], max=[3.0, 1.0], min=[-1.0, -4.0]))
    a = sn.push_sizes(pushes, r_l={8: 100.0}, w_norms={8: np.array([2.0, 5.0])}, natural=natural).set_index("neuron")
    assert a.loc[0, "c_activation"] == pytest.approx(50.0) and a.loc[0, "per_largest"] == pytest.approx(50 / 3)
    assert a.loc[1, "c_activation"] == pytest.approx(10.0) and a.loc[1, "per_largest"] == pytest.approx(2.5)


def test_wilson_interval_matches_the_formula():
    lo, hi = sn.wilson(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert lo[0] == pytest.approx(0.0) and hi[0] == pytest.approx(0.2775, abs=1e-4)
    assert lo[1] == pytest.approx(0.2366, abs=1e-4) and hi[1] == pytest.approx(0.7634, abs=1e-4)
    assert lo[2] == pytest.approx(0.7225, abs=1e-4) and hi[2] == pytest.approx(1.0)


def test_persona_share_per_kind_layer_and_sign():
    kept = pd.DataFrame(dict(layer=[8] * 5, kind=["neuron"] * 4 + ["random"], sign=[1, -1, 1, -1, 1],
                             category=["persona", "other", "persona", "persona", "other"]))
    s = sn.persona_share(kept).set_index(["kind", "sign", "layer"])
    assert (s.loc[("neuron", "both", 8), "n"], s.loc[("neuron", "both", 8), "k"]) == (4, 3)
    assert (s.loc[("neuron", "-", 8), "n"], s.loc[("neuron", "-", 8), "k"]) == (2, 1)
    assert s.loc[("random", "both", 8), "share"] == 0
    assert (s.lo <= s.share).all() and (s.share <= s.hi).all()


def test_natural_share_counts_only_the_natural_size_pushes_with_their_size_against_r_L():
    rows = pd.DataFrame(dict(layer=8, kind=["neuron"] * 4 + ["role"], sign=[1, -1, 1, 1, 1],
                             size=["nat", "nat", "nat", "a125", "nat"], c=[2.0, -4.0, 6.0, 50.0, 10.0],
                             category=["persona", "default_assistant", "default_assistant", "persona", "persona"]))
    s = sn.natural_share(rows, {8: 40.0}).set_index(["kind", "sign", "layer"])
    n = s.loc[("neuron", "both", 8)]
    assert (n.n, n.k) == (3, 1) and n.size_median == 0.1
    assert s.loc[("role", "+", 8), "share"] == 1


def test_step_loads_the_model_through_the_authors_code(monkeypatch, upstream_checkout, tiny_qwen3_dir):
    import importlib.util
    from pathlib import Path

    import persona_neurons.upstream as up

    path = Path(__file__).resolve().parents[1] / "steps" / "7_single_neurons.py"
    spec = importlib.util.spec_from_file_location("step7", path)
    step7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step7)
    monkeypatch.setenv("PN_UPSTREAM_DIR", str(upstream_checkout))
    monkeypatch.setattr(up, "model_snapshot", lambda *a, **k: tiny_qwen3_dir)
    model, tok = step7.load_model("float32")
    assert model.config.num_hidden_layers == 2 and tok("Who").input_ids


def _step7():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "steps" / "7_single_neurons.py"
    spec = importlib.util.spec_from_file_location("step7", path)
    step7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step7)
    return step7


def test_step_natural_push_and_collect_end_to_end_on_the_tiny_model(monkeypatch, tmp_path, upstream_checkout,
                                                                    tiny_qwen3_dir):
    import json

    import persona_neurons.upstream as up

    step7 = _step7()
    monkeypatch.setenv("PN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PN_UPSTREAM_DIR", str(upstream_checkout))
    monkeypatch.setattr(up, "model_snapshot", lambda *a, **k: tiny_qwen3_dir)
    model, tok = step7.load_model("float32")
    d, n_neurons = model.config.hidden_size, model.config.intermediate_size
    inputs = tmp_path / "single_neurons" / "inputs"
    inputs.mkdir(parents=True)
    layers, neurons = [0, 1], {"0": [3, 5], "1": [7, 2]}
    g = torch.Generator().manual_seed(0)
    torch.save({"layers": layers, "contrasts": torch.randn(2, 2, d, generator=g)}, inputs / "role_contrasts.pt")
    pcs = np.linalg.qr(np.random.default_rng(0).standard_normal((d, 2)))[0].T[None].repeat(2, 0)
    np.savez(inputs / "pcs.npz", layers=np.array(layers), pcs=pcs.astype(np.float32), proj_max=np.ones((2, 2)))
    ids = [list(np.random.default_rng(i).integers(0, 1000, size=n)) for i, n in enumerate((9, 12))]
    pd.DataFrame(dict(corpus=["default", "role"], token_ids=ids,
                      weights=[np.r_[np.zeros(3), np.full(len(x) - 3, 1 / (len(x) - 3))] for x in ids])).to_parquet(
        inputs / "natural_corpus.parquet")
    plan = pd.concat([sn.push_plan(l, neurons[str(l)], 2, 2, 2, 2) for l in layers], ignore_index=True)
    plan.to_parquet(inputs / "plan.parquet")
    (inputs / "prepare.json").write_text(json.dumps(dict(layers=layers, neurons=neurons, n_random=2, n_pc=2)))

    step7.main(["natural"])
    nat = tmp_path / "single_neurons" / "natural"
    for l in layers:
        assert json.loads((nat / f"r_L{l}.json").read_text())["r_L"] > 0
        assert len(pd.read_parquet(nat / f"natural_range_L{l}.parquet")) == n_neurons
        assert np.load(nat / f"w_norms_L{l}.npy").shape == (n_neurons,)
    step7.main(["push", "--shard", "0/2", "--batch-size", "16"])
    step7.main(["push", "--shard", "1/2", "--batch-size", "16"])
    step7.main(["collect"])
    pushes = pd.read_parquet(tmp_path / "single_neurons" / "pushes.parquet")
    assert sorted(pushes.push_id) == sorted(plan.push_id)
    assert {"reply", "tokens", "stop", "c", "reply_cut", "stop_cut"} <= set(pushes.columns)
    r = {l: json.loads((nat / f"r_L{l}.json").read_text())["r_L"] for l in layers}
    a = pushes[pushes["size"] != "nat"]
    np.testing.assert_allclose(a.c.abs(), a.alpha * a.layer.map(r), rtol=1e-6)

    # the labels, with the judge's batches stubbed: every reply labelled persona except the strongest of each direction
    import persona_neurons.judge as judge

    label = '{{"category": "{}", "persona": "x", "description": "", "tags": [], "weirdness": 1, "coherence": 5}}'
    texts = {p: label.format("incoherent" if s == "a125" else "persona") for p, s in zip(pushes.push_id, pushes["size"])}
    monkeypatch.setattr(judge, "collect", lambda ids, run_dir: texts)
    (tmp_path / "single_neurons" / "labels").mkdir()
    (tmp_path / "single_neurons" / "labels" / "batches.json").write_text("[]")
    step7.main(["label", "collect"])
    kept = pd.read_parquet(tmp_path / "single_neurons" / "kept.parquet")
    assert (kept.alpha == 1.0).all() and len(kept) == len(plan[plan["size"] == "a100"])
    step7.main(["tables"])
    share = pd.read_parquet(tmp_path / "single_neurons" / "tables" / "share.parquet")
    assert (share.share == 1).all() and set(share.kind) == set(sn.KINDS)
    assert len(pd.read_parquet(tmp_path / "single_neurons" / "tables" / "push_size.parquet")) == 2 * 2 * 2
