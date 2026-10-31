"""Patching under role prompts (step 8): the conditions, what each patch does, the analysis, and the step on the tiny model."""
import contextlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from persona_neurons import necessity as nc, role_steering as rs
from persona_neurons.upstream import use_upstream
from tests.test_role_steering import _fake_inputs, _step6


def _inputs():
    g = np.random.default_rng(9)
    delta = g.standard_normal((2, 128))
    q = np.linalg.qr(g.standard_normal((128, 128)))[0][:, :12]
    layer = [0] * 7 + [1] * 5
    rank = pd.DataFrame({"layer": layer, "coef": [q[:, i] @ delta[l] for i, l in enumerate(layer)]})
    comps = {("rotation_matched", 0): {"rank": rank, "v": q}, ("components", 0): {"rank": rank, "v": q}}
    sets = pd.DataFrame({"kind": ["neurons"] * 3 + ["random"] * 2, "draw": 0, "layer": [0, 0, 1, 0, 1],
                         "neuron": [1, 2, 3, 4, 5]})
    sets["delta"] = delta[sets.layer, sets.neuron]
    return sets, comps, delta, q


def _cond(kind, how):
    c = nc.conditions("pirate")
    return c[(c.set == kind) & (c.patch == how)].iloc[0].to_dict()


def test_conditions_cover_each_set():
    c = nc.conditions("pirate")
    assert len(c) == 6 and set(c[c.patch == "subtract"].set) == set(nc.SETS) and c.iloc[0].patch == "prompt"


def test_prompt_index_uses_each_system_prompt_twice():
    used = pd.Series([nc.prompt_index(q, s) for q, s in nc.ITEMS]).value_counts()
    assert sorted(used.index) == [0, 1, 2, 3, 4] and used.eq(2).all()


@pytest.mark.parametrize("kind", ["neurons", "random"])
def test_patch_neuron_sets(kind):
    sets, comps, delta, q = _inputs()
    how, vals = nc.patch_values(_cond(kind, "subtract"), sets, comps)
    assert how == "rows"
    np.testing.assert_allclose(vals.value, -sets[sets.kind == kind].delta.to_numpy())


@pytest.mark.parametrize("kind", ["components", "rotation_matched"])
def test_patch_directions(kind):
    sets, comps, delta, q = _inputs()
    how, vals = nc.patch_values(_cond(kind, "subtract"), sets, comps)
    assert how == "rows"
    rank = comps[(kind, 0)]["rank"]
    for layer in (0, 1):
        v = q[:, np.flatnonzero(rank.layer.to_numpy() == layer)]
        np.testing.assert_allclose(vals[vals.layer == layer].value.to_numpy(), -(v @ (v.T @ delta[layer])), atol=1e-5)


def test_patch_prompt_and_vector():
    sets, comps, delta, q = _inputs()
    assert nc.patch_values(nc.conditions("pirate").iloc[0].to_dict(), sets, comps) == ("none", None)
    assert nc.patch_values(_cond("vector", "subtract"), sets, comps) == ("vector", None)


def test_lower_ranked_reaches_the_top_sum():
    delta = np.array([[4.0, 3.0, 2.0, 1.0, -0.5, 0.5, 0.5, 0.25]])
    p = np.ones_like(delta)
    s, x = nc.lower_ranked(delta, p, k=1)
    assert x["target"] == 4.0 and x["matched"] and x["n"] == 2 and s.neuron.tolist() == [1, 2]
    s, x = nc.lower_ranked(delta, p, k=2)
    assert x["target"] == 7.0 and not x["matched"] and x["sum"] == pytest.approx(3.75)
    s, x = nc.lower_ranked(delta, p, k=2, same_sign=True)
    assert (s.contribution > 0).all() and x["sum"] == pytest.approx(4.25)


def _tiny(upstream_checkout, tiny_qwen3_dir):
    use_upstream(upstream_checkout)
    from assistant_axis.internals import ProbingModel

    pm = ProbingModel(str(tiny_qwen3_dir), dtype=torch.float32, device={"": "cpu"})
    return pm.model, pm.tokenizer


def test_vector_patch_on_the_residual(upstream_checkout, tiny_qwen3_dir):
    model, _ = _tiny(upstream_checkout, tiny_qwen3_dir)
    ids = torch.tensor([[1, 5, 9, 2, 7]])
    v = torch.from_numpy(np.random.default_rng(3).standard_normal(64)).float()
    base = nc.residual_at(model, ids, 1)
    with nc.vector_patch(model, v, 1):
        patched = nc.residual_at(model, ids, 1)
    torch.testing.assert_close(patched - base, -v.double().expand(5, 64), rtol=1e-3, atol=1e-3)


def test_row_intervention_adds_to_its_row(upstream_checkout, tiny_qwen3_dir):
    model, _ = _tiny(upstream_checkout, tiny_qwen3_dir)
    ids = torch.tensor([[1, 5, 9, 2, 7]] * 2)
    got = {}
    g = model.model.layers[0].mlp.down_proj.register_forward_hook(lambda m, args, o: got.__setitem__("x", args[0].detach().clone()))
    with torch.no_grad():
        model(ids)
        clean = got["x"].clone()
        with rs.RowIntervention(model, [pd.DataFrame({"layer": [0], "neuron": [4], "value": [1.0]}), None]):
            model(ids)
    g.remove()
    torch.testing.assert_close(got["x"][0, :, 4], clean[0, :, 4] + 1.0)
    torch.testing.assert_close(got["x"][1], clean[1])


def _judged(extra=None):
    """Thirty roles' replies: the role prompt alone has 8 of 10 with the traits; the vector's patch 2 to 6 fewer by role, the
    top 100's 1, 2 or 3 fewer, the components' twice the top 100's drop, the others no drop. One reply is unscored."""
    rows = []
    for i in range(30):
        conds = nc.conditions(f"r{i}")
        for c in conds.itertuples():
            n_in = 8 - {"vector": 2 + i % 5, "neurons": 1 + i % 3, "components": 2 * (1 + i % 3)}.get(c.set, 0)
            for s in range(10):
                rows.append({"role": c.role, "patch": c.patch, "set": c.set, "response": "x",
                             "score": 3.0 if s < n_in - 1 else (2.0 if s < n_in else 0.0)})
    j = pd.DataFrame(rows)
    first = j.index[(j.role == "r0") & (j.patch == "subtract") & (j.set == "random")][0]
    j.loc[first, "score"] = np.nan
    return j


def test_analysis_on_a_small_table():
    per = nc.per_role_table(_judged())
    per = nc.add_drops(per, per[per.patch == "prompt"].set_index("role"))
    assert len(per) == 30 * 6
    r = per.set_index(["role", "patch", "set"])
    u = r.loc[("r0", "subtract", "random")]   # the unscored reply is not counted: 7 of the 9 scored have the traits, 6 fully
    assert u.n == 10 and u.n_labelled == 9 and u.in_role == pytest.approx(7 / 9) and u.fully == pytest.approx(6 / 9)
    assert r.loc[("r1", "subtract", "neurons"), "drop_in_role"] == pytest.approx(0.2)
    both = nc.paired_table(per)
    both = both[(both.patch == "subtract") & (both.measure == "in_role")].set_index("comparison")
    drop = r.xs("subtract", level="patch").drop_in_role.unstack("set")
    assert both.loc["prompt alone minus neurons", "n"] == 30
    assert both.loc["components minus neurons", "mean_diff"] == pytest.approx((drop.neurons - drop.components).mean())
    assert both.loc["rotation_matched minus neurons", "mean_diff"] == pytest.approx(drop.neurons.mean() - drop.rotation_matched.mean())
    # the components' drop is twice the top 100's at every role, so each end of the interval is twice the top 100's
    s = nc.share_of_vector_drop(per)
    s = s.set_index("set")
    assert s.loc["neurons", "share_of_vector_drop"] == pytest.approx(drop.neurons.mean() / drop.vector.mean())
    assert s.loc["components", ["lo", "hi"]].to_numpy() == pytest.approx(2 * s.loc["neurons", ["lo", "hi"]].to_numpy())


def test_scaled_tables_on_a_small_table():
    first = nc.per_role_table(_judged())
    first = nc.add_drops(first, first[first.patch == "prompt"].set_index("role"))
    rows = []
    for i in range(30):
        for c in nc.scaled_conditions(f"r{i}").itertuples():   # 3 fewer than the prompt alone for the top 100, 1 for random
            n_in = 8 - (3 if c.set == "neurons" else 1)
            rows += [{"role": c.role, "patch": c.patch, "set": c.set, "response": "x", "score": 3.0 if s < n_in else 0.0} for s in range(10)]
    per = nc.add_drops(nc.per_role_table(pd.DataFrame(rows)), first[first.patch == "prompt"].set_index("role"))
    assert per[(per.patch == "scaled") & (per.set == "neurons")].drop_in_role.round(6).eq(0.3).all()
    paired, ratios = nc.scaled_tables(per, first)
    vec = first[(first.patch == "subtract") & (first.set == "vector")].drop_in_role.mean()
    assert ratios.set_index(["patch", "set"]).share_of_vector_drop[("scaled", "neurons")] == pytest.approx(0.3 / vec)
    p = paired[(paired.patch == "push") & (paired.measure == "in_role")].set_index("comparison")
    assert p.loc["random minus neurons", "mean_diff"] == pytest.approx(0.2)
    assert p.loc["the vector minus neurons", "mean_diff"] == pytest.approx(0.3 - vec)


def test_paired_summary_with_one_role_has_no_bayes_factor():
    s = nc.paired_summary([0.5], [0.2])
    assert s["n"] == 1 and s["mean_diff"] == pytest.approx(0.3) and np.isnan(s["bf10_r1.0"])


def test_scaled_patch():
    """The top 100's change is scaled by the strength, with the random neurons' write matched to it; at the steering runs'
    strength each set's write has norm 4 |v_r|, subtracted."""
    g = np.random.default_rng(2)
    weights = {l: torch.tensor(g.standard_normal((64, 128)), dtype=torch.float32) for l in (0, 1)}
    w_down = lambda l: weights[l]
    sets = pd.DataFrame({"kind": ["neurons"] * 3 + ["random"] * 3, "draw": 0, "layer": [0, 0, 1, 0, 1, 1], "neuron": [1, 2, 3, 4, 5, 6],
                         "delta": g.standard_normal(6)})
    c = nc.scaled_conditions("pirate").set_index(["patch", "set"])
    how, top = nc.scaled_patch(c.loc[("scaled", "neurons")].to_dict() | {"patch": "scaled", "set": "neurons"}, sets, w_down, 5.0)
    assert how == "rows"
    np.testing.assert_allclose(top.value, -nc.SCALE_ALONG_V * sets[sets.kind == "neurons"].delta.to_numpy(), rtol=1e-5)
    _, rnd = nc.scaled_patch(c.loc[("scaled", "random")].to_dict() | {"patch": "scaled", "set": "random"}, sets, w_down, 5.0)
    assert rs.write_vector(rnd, w_down).norm().item() == pytest.approx(rs.write_vector(top, w_down).norm().item(), rel=1e-4)
    for kind in ("neurons", "random"):
        _, vals = nc.scaled_patch(c.loc[("push", kind)].to_dict() | {"patch": "push", "set": kind}, sets, w_down, 5.0)
        assert (vals.value.to_numpy() / sets[sets.kind == kind].delta.to_numpy()).max() < 0
        assert rs.write_vector(vals, w_down).norm().item() == pytest.approx(20.0, rel=1e-4)


def test_usual_range_table_and_reply_sds():
    pos, neg = torch.ones(1, 3) * 2.0, torch.zeros(1, 3)
    stats = {"sums": {"missing": {"count": 2, "pos": pos * 2, "neg": neg, "pos_sq": pos ** 2 * 2 + 1.0, "neg_sq": neg}},
             "subsample": {"a": torch.stack([torch.ones(1, 3), torch.zeros(1, 3)]), "b": torch.stack([torch.ones(1, 3) * 3, torch.zeros(1, 3)])}}
    bound, sub = nc.reply_sds(stats)
    assert bound.shape == (1, 3) and sub == pytest.approx(np.full((1, 3), np.sqrt(2.0)))
    sets = pd.DataFrame({"role": "a", "kind": "neurons", "draw": 0, "layer": [0], "neuron": [1], "delta": [2.0]})
    sizes = pd.DataFrame({"role": ["a"], "patch": ["scaled"], "set": ["neurons"], "scale": [-3.0]})
    t = nc.usual_range_table(sets, sizes, np.full((1, 3), 2.0), np.zeros((1, 3))).iloc[0]
    assert t.write == 6 and t.own_size == 2 and t.in_sd_bound == 3 and np.isnan(t.in_sd_subsample)


def test_step8_on_the_tiny_model(tmp_path, tiny_qwen3_dir, upstream_checkout, monkeypatch):
    s6 = _step6()
    spec = importlib.util.spec_from_file_location("step8", Path(__file__).parent.parent / "steps/8_necessity.py")
    s8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s8)
    roles = ["alien", "gamer"]
    monkeypatch.setenv("PN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PN_UPSTREAM_DIR", str(upstream_checkout))
    bases = {l: np.linalg.qr(np.random.default_rng(l).standard_normal((128, 128)))[0] for l in range(2)}
    _fake_inputs(tmp_path, roles, bases)
    monkeypatch.setattr(s6.sp, "layer_basis", lambda stats, rl, l, t: (None, bases[l]))
    monkeypatch.setattr(s6.sp, "roles_with_vectors", lambda d: roles)
    base = ["--target", "1"]
    assert s6.main(["prepare"] + base) == 0
    for part in (["components", "--layers", "0-1"], ["rotate", "--layers", "0-1"], ["finish"]):
        assert s6.main(part + base) == 0
    (tmp_path / "neuron_stats").mkdir()
    g = np.random.default_rng(4)
    x = lambda *shape: torch.from_numpy(g.standard_normal(shape))
    torch.save({"scored": False, "sums": {"missing": {"count": 1, "pos": x(2, 128), "neg": x(2, 128), "pos_sq": x(2, 128) ** 2,
                                                      "neg_sq": x(2, 128) ** 2, "attn": x(2, 64), "embed": x(64), "resid": x(2, 64)}},
                "subsample": {"k0": x(2, 2, 128), "k1": x(2, 2, 128)}}, tmp_path / "neuron_stats" / "default.pt")
    assert s8.main(["prepare"] + base) == 0
    out = tmp_path / "necessity"
    conv = pd.read_parquet(out / "conversations.parquet")
    assert len(conv) == 20 and (conv.prompt_index == (2 * conv.question_index + conv["sample"]) % 5).all()
    prompts = json.loads((upstream_checkout / "data/roles/instructions/alien.json").read_text())["instruction"]
    first = conv[(conv.role == "alien") & (conv.question_index == 2) & (conv["sample"] == 1)].iloc[0]
    assert first.system == prompts[0]["pos"]   # (2 × 2 + 1) mod 5
    gen = ["--model-path", str(tiny_qwen3_dir), "--dtype", "float32", "--device", "cpu", "--max-new-tokens", "4", "--chunk", "4",
           "--changes"] + base
    assert s8.main(["generate"] + gen) == 0
    rep = pd.read_parquet(out / "replies" / "alien.parquet")
    assert len(rep) == 6 * 10 and rep.groupby(["question_index", "sample"]).seed.nunique().eq(1).all()
    # the role prompt alone: each reply is the one its seed gives under its own system prompt
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    r0 = rep[(rep.patch == "prompt") & (rep.question_index == 2) & (rep["sample"] == 1)].iloc[0]
    msgs = [{"role": "system", "content": first.system}, {"role": "user", "content": rs.QUESTIONS[2]}]
    assert r0.response == rs.generate_rows(model, tok, [msgs], [rs.reply_seed(0, 2, 1)], max_new_tokens=4)[0]
    sizes = pd.read_parquet(out / "sizes" / "alien.parquet")
    assert len(sizes) == 5 and sizes.write_norm.gt(0).all()
    v = sizes[sizes.set == "vector"].iloc[0]
    assert v.write_along_v == pytest.approx(-v.write_norm)   # the vector's subtraction runs along the vector
    ch = pd.read_parquet(out / "changes" / "alien.parquet")
    assert len(ch) == 5 * 10 * 2 and set(ch.part) == {"prompt", "reply"} and "prompt" not in set(ch.patch)
    sub = ch[ch.set == "vector"]
    # subtracting the vector adds −v at every position: the same norm at every token, all of it along v
    np.testing.assert_allclose(sub.along_v, -sub.mean_norm, rtol=1e-4)
    from persona_neurons import judge

    def judged(flags, n):
        sent = []
        monkeypatch.setattr(judge, "submit", lambda reqs, run_dir: sent.extend(reqs) or ["batch_0"])
        assert s8.main(["judge", "submit"] + flags + base) == 0
        assert len(sent) == n and all(len(r["custom_id"]) <= 64 for r in sent)
        monkeypatch.setattr(judge, "collect", lambda ids, run_dir: {r["custom_id"]: "Score: 3" for r in sent})
        folder = tmp_path / ("necessity_scaled" if flags else "necessity")
        (folder / "judge").mkdir(exist_ok=True)
        (folder / "judge" / "batches.json").write_text(json.dumps(["batch_0"]))
        assert s8.main(["judge", "collect"] + flags + base) == 0
        assert s8.main(["build"] + flags + base) == 0
        assert s8.main(["analyze"] + flags + base) == 0

    judged([], 2 * len(rep))
    per = pd.read_parquet(out / "per_role.parquet")
    assert len(per) == 12 and (per.in_role == 1).all()
    assert per[(per.patch == "subtract") & (per.set == "neurons")].write_norm.notna().all()
    assert per[(per.patch == "subtract") & (per.set == "neurons")].reply_mean_norm.notna().all()
    assert pd.read_parquet(out / "paired.parquet").comparison.str.contains("prompt alone minus vector").any()
    with pytest.raises(SystemExit):
        s8.main(["analyze"] + base)   # never overwrites
    assert s8.main(["analyze", "--to", "necessity/again"] + base) == 0   # another folder, beside the released outputs
    # the scaled run
    assert s8.main(["prepare-scaled"] + base) == 0
    assert s8.main(["generate", "--scaled"] + gen) == 0
    scaled = tmp_path / "necessity_scaled"
    rep2 = pd.read_parquet(scaled / "replies" / "alien.parquet")
    assert len(rep2) == 4 * 10 and set(rep2.seed) == set(rep.seed)
    z = pd.read_parquet(scaled / "sizes" / "alien.parquet").set_index(["patch", "set"])
    assert z.loc[("scaled", "neurons"), "scale"] == pytest.approx(-nc.SCALE_ALONG_V, rel=1e-4)
    assert z.loc[("scaled", "random"), "write_norm"] == pytest.approx(z.loc[("scaled", "neurons"), "write_norm"], rel=1e-3)
    assert len(pd.read_parquet(scaled / "changes" / "alien.parquet")) == 4 * 10 * 2
    judged(["--scaled"], 2 * len(rep2))
    assert len(pd.read_parquet(scaled / "per_role.parquet")) == 8
    assert s8.main(["usual-range"] + base) == 0
    assert (pd.read_parquet(scaled / "usual_range" / "per_neuron.parquet").patch.isin(nc.SCALED_PATCHES)).all()
    # removals of the same size
    spec = importlib.util.spec_from_file_location("step8b", Path(__file__).parent.parent / "steps/8b_necessity_same_size.py")
    s8b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s8b)
    more = tmp_path / "role_steering_more"
    (more / "replies").mkdir(parents=True)
    pd.read_parquet(tmp_path / "role_steering" / "roles.parquet").to_parquet(more / "roles.parquet", index=False)
    for r in roles:
        a = pd.read_parquet(out / "replies" / f"{r}.parquet")
        a[a.patch == "prompt"].assign(kind="none").to_parquet(more / "replies" / f"{r}.parquet", index=False)
    assert s8b.main(["prepare"] + base) == 0
    same = tmp_path / "necessity_same_size"
    assert len(pd.read_parquet(same / "lower_sets.parquet")) == 2 * len(roles) and len(pd.read_parquet(same / "assistant.parquet")) == 20
    model_flags = ["--model-path", str(tiny_qwen3_dir), "--dtype", "float32", "--device", "cpu"] + base
    assert s8b.main(["score", "--batch-size", "7"] + model_flags) == 0
    forced = pd.read_parquet(same / "forced" / "alien.parquet")
    assert len(forced) == len(nc.FORCED) * 20 and forced.logp_mean.notna().all()
    added = pd.read_parquet(same / "sizes_forced" / "alien.parquet").set_index("condition")
    alpha = pd.read_parquet(same / "alpha.parquet").set_index("role").alpha["alien"]
    assert added.loc[nc.SLICE, "norm"] == pytest.approx(abs(alpha) * added.loc[nc.VECTOR, "norm"], rel=1e-6)
    assert added.loc[nc.OFF, "along_v"] == pytest.approx(0, abs=1e-5)
    assert added.loc[nc.DIR_TOP, "norm"] == pytest.approx(added.loc[nc.TOP, "norm"], rel=1e-6)
    assert s8b.main(["generate", "--max-new-tokens", "4", "--chunk", "4"] + model_flags) == 0
    assert len(pd.read_parquet(same / "replies" / "alien.parquet")) == 3 * 10
    assert len(pd.read_parquet(same / "changes" / "alien.parquet")) == 3 * 10 * 2
    sent = []
    monkeypatch.setattr(judge, "submit", lambda reqs, run_dir: sent.extend(reqs) or ["batch_0"])
    assert s8b.main(["judge", "submit"] + base) == 0 and len(sent) == 60
    monkeypatch.setattr(judge, "collect", lambda ids, run_dir: {r["custom_id"]: "Score: 3" for r in sent})
    (same / "judge").mkdir(exist_ok=True)
    (same / "judge" / "batches.json").write_text(json.dumps(["batch_0"]))
    assert s8b.main(["judge", "collect"] + base) == 0 and s8b.main(["build"] + base) == 0
    assert s8b.main(["analyze"] + base) == 0
    summary = pd.read_parquet(same / "sizes_summary.parquet").set_index("condition")
    assert len(summary) == 9 and summary.loc[nc.SLICE, "written_norm"] > 0 and summary.loc[nc.TOP, "measured_norm"] > 0
    assert len(pd.read_parquet(same / "judged_per_role.parquet")) == 6
    with pytest.raises(SystemExit):
        s8b.main(["analyze"] + base)
