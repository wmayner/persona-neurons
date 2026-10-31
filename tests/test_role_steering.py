"""persona_neurons.role_steering, and step 6 on the tiny model."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from persona_neurons import role_steering as rs


def _tiny(upstream_checkout, tiny_qwen3_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from persona_neurons.upstream import use_upstream

    use_upstream(upstream_checkout)
    model = AutoModelForCausalLM.from_pretrained(tiny_qwen3_dir, torch_dtype=torch.float32).eval()
    return model, AutoTokenizer.from_pretrained(tiny_qwen3_dir)


def _arrays(seed=0, n_layers=3, n_neurons=40):
    g = np.random.default_rng(seed)
    return (g.standard_normal((n_layers, n_neurons)).astype(np.float32),
            g.standard_normal((n_layers, n_neurons)).astype(np.float32))


def test_role_indices_are_positions_among_all_roles_sorted():
    idx = rs.role_indices(["default", "zebra", "gamer", "alien", "aardvark", "analyst"])
    assert idx == {"aardvark": 0, "alien": 1, "analyst": 2, "gamer": 3, "zebra": 4}


def test_top_neurons_rank_by_absolute_contribution():
    delta, p = _arrays()
    top = rs.top_neurons(delta, p, k=7)
    want = np.sort(np.abs(delta * p).ravel())[::-1][:7]
    np.testing.assert_allclose(top.contribution.abs().to_numpy(), want, rtol=1e-6)
    np.testing.assert_allclose(top.delta.to_numpy(), delta[top.layer, top.neuron])


def test_neuron_sets_random_set_matches_layers_and_is_seeded():
    delta, p = _arrays()
    sets = rs.neuron_sets("pirate", 4, delta, p, k=10, draw=3)
    top = sets[sets.kind == "neurons"]
    assert len(top) == 10 and set(sets.kind) == {"neurons", "random"}
    want = top.layer.value_counts().sort_index()
    r = sets[sets.kind == "random"]
    assert set(r.draw) == {3}
    pd.testing.assert_series_equal(r.layer.value_counts().sort_index(), want)
    np.testing.assert_allclose(r.delta.to_numpy(), delta[r.layer, r.neuron])
    # each layer's draw in increasing layer order, from [RANDOM_SEED, role index, k, draw]
    rng = np.random.default_rng([rs.RANDOM_SEED, 4, 10, 3])
    first = sorted(want.index)[0]
    assert sorted(r[r.layer == first].neuron) == sorted(rng.choice(40, size=int(want[first]), replace=False))
    np.testing.assert_allclose(sets.contribution, delta[sets.layer, sets.neuron].astype(np.float64)
                               * p[sets.layer, sets.neuron], rtol=1e-6)


def test_top_components_rank_over_layers():
    c = np.array([[0.1, -3.0, 0.2], [2.0, 0.0, -0.5]])
    top = rs.top_components(c, k=3)
    assert list(zip(top.layer, top.component)) == [(0, 1), (1, 0), (1, 2)]


def test_component_add_values_project_delta_on_the_chosen_components():
    g = np.random.default_rng(1)
    v, _ = np.linalg.qr(g.standard_normal((6, 6)))
    delta = g.standard_normal(6)
    vals = rs.component_add_values({2: (v[:, [0, 3]], v[:, [0, 3]].T @ delta)})
    np.testing.assert_allclose(vals[vals.layer == 2].sort_values("neuron").value, v[:, [0, 3]] @ (v[:, [0, 3]].T @ delta))


def test_write_vector_and_matched_scale():
    g = np.random.default_rng(2)
    w = {0: torch.from_numpy(g.standard_normal((8, 5))), 1: torch.from_numpy(g.standard_normal((8, 5)))}
    vals = pd.DataFrame(dict(layer=[0, 0, 1], neuron=[1, 4, 2], value=[0.5, -1.0, 2.0]))
    want = 0.5 * w[0][:, 1] - 1.0 * w[0][:, 4] + 2.0 * w[1][:, 2]
    torch.testing.assert_close(rs.write_vector(vals, lambda l: w[l]), want.double(), rtol=1e-5, atol=1e-6)
    c = rs.matched_scale(vals, lambda l: w[l], target_norm=7.0)
    assert (c * want).norm().item() == pytest.approx(7.0)


def test_condition_values_match_the_vectors_size():
    g = np.random.default_rng(5)
    delta, p = g.standard_normal((2, 128)), g.standard_normal((2, 128))
    sets = rs.neuron_sets("pirate", 0, delta, p, k=10)
    v = {l: np.linalg.qr(g.standard_normal((128, 128)))[0] for l in range(2)}
    rank = rs.top_components(g.standard_normal((2, 128)), k=20)
    cols = np.stack([v[l][:, j] for l, j in zip(rank.layer, rank.component)], axis=1)
    rank["coef"] = [cols[:, i] @ delta[l] for i, l in enumerate(rank.layer)]
    w = {l: torch.from_numpy(g.standard_normal((64, 128))) for l in range(2)}
    w_down = lambda l: w[l]
    conds = rs.conditions("pirate")
    cond = lambda k: conds[conds.kind == k].iloc[0].to_dict()
    assert rs.condition_values(cond("vector"), sets, {}, w_down, 9.0) == ("vector", None)
    for kind in ("neurons", "random", "components"):
        _, vals = rs.condition_values(cond(kind), sets, {("components", 0): {"rank": rank, "v": cols}}, w_down, 9.0)
        assert rs.write_vector(vals, w_down).norm().item() == pytest.approx(9.0)
    top = sets[sets.kind == "neurons"]
    _, vals = rs.condition_values(cond("neurons"), sets, {}, w_down, 9.0)
    ratio = vals.value.to_numpy() / top.delta.to_numpy()
    np.testing.assert_allclose(ratio, ratio[0])
    assert ratio[0] > 0



def _down_inputs(model, ids, layer):
    seen = []
    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        h.remove()
    return seen[0]


def test_row_intervention_acts_on_each_row_alone(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    ids = tok(["Who are you?"] * 3, return_tensors="pt").input_ids
    before = _down_inputs(model, ids, 1)
    rows = [pd.DataFrame(dict(layer=[1, 1], neuron=[3, 40], value=[2.5, -1.0])), None,
            pd.DataFrame(dict(layer=[0], neuron=[7], value=[9.0]))]
    with rs.RowIntervention(model, rows):
        after = _down_inputs(model, ids, 1)
    torch.testing.assert_close(after[1], before[1])
    want0 = before[0].clone()
    want0[:, [3, 40]] += torch.tensor([2.5, -1.0])
    torch.testing.assert_close(after[0], want0)
    assert not torch.allclose(after[2], before[2])   # layer 0's change reaches layer 1 through the residual stream
    assert not model.model.layers[1].mlp.down_proj._forward_pre_hooks


def test_reply_seeds_are_per_role_question_and_sample():
    s = rs.reply_seed(3, 2, 1)
    assert s == rs.reply_seed(3, 2, 1) and len({s, rs.reply_seed(3, 2, 0), rs.reply_seed(4, 2, 1)}) == 3


def test_generate_rows_pairs_each_reply_with_its_seed_whatever_the_batch(upstream_checkout, tiny_qwen3_dir):
    model, tok = _tiny(upstream_checkout, tiny_qwen3_dir)
    q = [[{"role": "user", "content": "Who are you?"}], [{"role": "user", "content": "What is your name?"}]]
    alone = rs.generate_rows(model, tok, [q[0]], seeds=[11], max_new_tokens=12)
    batch = rs.generate_rows(model, tok, [q[1], q[0], q[0]], seeds=[5, 11, 12], max_new_tokens=12)
    assert batch[1] == alone[0] and batch[2] != batch[1]
    vals = pd.DataFrame(dict(layer=[0], neuron=[3], value=[50.0]))
    mixed = rs.generate_rows(model, tok, [q[1], q[0], q[0]], seeds=[5, 11, 12], max_new_tokens=12,
                             rows=[vals, None, None])
    assert mixed[1:] == batch[1:] and mixed[0] != batch[0]


def test_row_sampler_draws_from_the_authors_distribution():
    scores = torch.tensor([[2.0, 1.9, -5.0, -6.0]])
    sampler = rs.RowSampler([7], temperature=0.7, top_k=2, top_p=0.9, device="cpu")
    picks = {int(sampler(torch.zeros((1, 1), dtype=torch.long), scores.clone()).argmax()) for _ in range(50)}
    assert picks == {0, 1}


# ---- step 6 on the tiny model (2 layers of 128 neurons, d_model 64), target layer 1

def _step6():
    path = Path(__file__).parent.parent / "steps/6_role_steering.py"
    spec = importlib.util.spec_from_file_location("step6", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_inputs(data: Path, roles: list[str], bases: dict):
    """vectors/{all,without}/, split/arrays.npz and split/pca/ for the tiny model, the components' contributions in
    the given bases; neuron_stats/ stands in through the bases (the test replaces split.layer_basis)."""
    g = np.random.default_rng(9)
    for version in ("all", "without"):
        (data / "vectors" / version).mkdir(parents=True)
        for r in roles + ["default"]:
            torch.save({"vector": torch.from_numpy(g.standard_normal((2, 64))).float(), "role": r},
                       data / "vectors" / version / f"{r}.pt")
    (data / "split" / "pca").mkdir(parents=True)
    delta = g.standard_normal((len(roles), 2, 128)).astype(np.float32)
    p = g.standard_normal((len(roles), 2, 128)).astype(np.float32)
    np.savez(data / "split" / "arrays.npz", roles=np.array(roles), norm=np.full(len(roles), 3.0), delta_a=delta, p=p)
    for l, v in bases.items():
        contrib = ((delta[:, l].astype(np.float64) @ v) * (p[:, l].astype(np.float64) @ v)).astype(np.float32)
        np.savez(data / "split" / "pca" / f"layer_{l:02d}.npz", roles=np.array(roles), contributions=contrib)


def test_step6_on_the_tiny_model(tmp_path, tiny_qwen3_dir, upstream_checkout, monkeypatch):
    s6 = _step6()
    roles = ["alien", "gamer", "witch"]
    monkeypatch.setenv("PN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PN_UPSTREAM_DIR", str(upstream_checkout))
    bases = {l: np.linalg.qr(np.random.default_rng(l).standard_normal((128, 128)))[0] for l in range(2)}
    _fake_inputs(tmp_path, roles, bases)
    (tmp_path / "vectors" / "without" / "witch.pt").unlink()   # a role without a vector is skipped
    monkeypatch.setattr(s6.sp, "layer_basis", lambda stats, rl, l, t: (None, bases[l]))
    monkeypatch.setattr(s6.sp, "roles_with_vectors", lambda d: roles)
    base = ["--target", "1"]
    assert s6.main(["prepare"] + base) == 0
    out = tmp_path / "role_steering"
    assert pd.read_parquet(out / "roles.parquet").to_dict("list") == {"role": ["alien", "gamer"], "role_index": [0, 1]}
    assert pd.read_parquet(out / "skipped.parquet").to_dict("records") == [{"role": "witch", "reason": "no vector"}]
    assert s6.main(["components", "--layers", "0"] + base) == 0
    assert s6.main(["components", "--layers", "1"] + base) == 0
    assert s6.main(["rotate", "--layers", "0-1"] + base) == 0
    assert s6.main(["finish"] + base) == 0
    z = np.load(out / "components" / "alien.npz")
    arr = np.load(tmp_path / "split" / "arrays.npz")
    want = np.concatenate([(bases[l].T @ arr["delta_a"][0, l]) * (bases[l].T @ arr["p"][0, l]) for l in range(2)])
    np.testing.assert_allclose(np.abs(z["contribution"]), np.sort(np.abs(want))[::-1][:100], rtol=1e-5)
    for i in (0, 57, 99):   # each column is the chosen component, with its projection of Δa
        np.testing.assert_allclose(z["cols"][:, i], bases[z["layer"][i]][:, z["component"][i]], rtol=1e-6)
        assert z["coef"][i] == pytest.approx(z["cols"][:, i] @ arr["delta_a"][0, z["layer"][i]], rel=1e-5)
    shares = pd.read_parquet(out / "shares.parquet")
    assert sorted(shares.kind) == sorted(["neurons", "random", "components", "rotation_matched"] * 2)
    # the rotated basis's directions: 100, with the top 100 neurons' layer counts, the rotation's own columns
    sets = pd.read_parquet(out / "sets.parquet")
    counts = sets[(sets.role == "alien") & (sets.kind == "neurons")].layer.value_counts().sort_index()
    dirs = np.load(out / "directions" / "alien.npz")
    assert dirs["rotation_matched|d0|cols"].shape == (128, 100)
    assert pd.Series(dirs["rotation_matched|d0|layer"]).value_counts().sort_index().to_dict() == counts.to_dict()
    q = rs.haar_rotation(128, [rs.ROTATION_SEED, 0, 1])
    rows = dirs["rotation_matched|d0|layer"] == 1
    np.testing.assert_allclose(dirs["rotation_matched|d0|cols"][:, rows],
                               q[:, dirs["rotation_matched|d0|direction"][rows]])
    top = rs.top_rotated(q, arr["delta_a"][0, 1], arr["p"][0, 1], int(counts[1]))
    assert dirs["rotation_matched|d0|direction"][rows].tolist() == top.direction.tolist()
    assert s6.main(["generate", "--model-path", str(tiny_qwen3_dir), "--dtype", "float32", "--device", "cpu",
                    "--max-new-tokens", "6", "--shard", "1/2"] + base) == 0
    assert [f.stem for f in (out / "replies").glob("*.parquet")] == ["gamer"]   # every 2nd role from the 2nd
    assert s6.main(["generate", "--model-path", str(tiny_qwen3_dir), "--dtype", "float32", "--device", "cpu",
                    "--max-new-tokens", "6"] + base) == 0
    rep = pd.read_parquet(out / "replies" / "alien.parquet")
    assert len(rep) == 5 * 10 and rep.response.str.len().gt(0).all()
    # judge: requests built; collect from saved results, as the API returns them
    from persona_neurons import judge

    trial = []
    monkeypatch.setattr(judge, "submit", lambda reqs, run_dir: trial.extend(reqs) or ["batch_0"])
    assert s6.main(["judge", "submit", "--roles", "gamer", "--limit", "5"] + base) == 0
    assert len(trial) == 5 and all(r["custom_id"].startswith("gamer__") for r in trial)
    sent = []
    monkeypatch.setattr(judge, "submit", lambda reqs, run_dir: sent.extend(reqs) or ["batch_1"])
    assert s6.main(["judge", "submit"] + base) == 0
    assert len(sent) == 2 * len(rep)
    monkeypatch.setattr(judge, "collect", lambda ids, run_dir: {r["custom_id"]: "Score: 3" for r in sent})
    (out / "judge").mkdir(exist_ok=True)
    (out / "judge" / "batches.json").write_text(json.dumps(["batch_1"]))
    assert s6.main(["judge", "collect"] + base) == 0
    assert s6.main(["build"] + base) == 0
    j = pd.read_parquet(out / "judged_replies.parquet")
    assert len(j) == 2 * len(rep) and j.score.notna().all()


def test_build_summary_counts_only_scored_replies():
    s6 = _step6()
    j = pd.DataFrame({"kind": ["vector", "vector", "neurons", "neurons"], "score": [3, 1, np.nan, 2]})
    s = s6.in_role_summary(j)
    assert s.loc["vector", "scored"] == 2 and s.loc["vector", "in_role"] == 0.5
    assert s.loc["neurons", "scored"] == 1 and s.loc["neurons", "in_role"] == 1.0


# ---- the rotated basis

def test_haar_rotation_is_orthogonal_seeded_and_sign_fixed():
    q = rs.haar_rotation(24, [rs.ROTATION_SEED, 0, 3])
    assert q.shape == (24, 24) and q.dtype == np.float32
    np.testing.assert_allclose(q.T @ q, np.eye(24), atol=1e-5)
    np.testing.assert_array_equal(q, rs.haar_rotation(24, [rs.ROTATION_SEED, 0, 3]))
    assert not np.allclose(q, rs.haar_rotation(24, [rs.ROTATION_SEED, 1, 3]))
    # the Q factor of the seeded Gaussian, each column's sign set so that R's diagonal is positive
    a = np.random.default_rng([rs.ROTATION_SEED, 0, 3]).standard_normal((24, 24), dtype=np.float32)
    np.testing.assert_allclose(np.diag(q.T @ a), np.abs(np.diag(np.linalg.qr(a)[1])), rtol=1e-4)


def test_top_rotated_takes_the_largest_contributions():
    g = np.random.default_rng(1)
    q = rs.haar_rotation(30, [rs.ROTATION_SEED, 0, 0])
    delta, p = g.standard_normal(30), g.standard_normal(30)
    top = rs.top_rotated(q, delta, p, 4)
    c = (q.T @ delta) * (q.T @ p)
    want = np.argsort(-np.abs(c), kind="stable")[:4]
    assert top.direction.tolist() == want.tolist()
    np.testing.assert_allclose(top.coef, (q.T @ delta)[want], rtol=1e-5)
    np.testing.assert_allclose(top.contribution, c[want], rtol=1e-5)
    assert c.sum() == pytest.approx(delta @ p, rel=1e-4)   # the directions' contributions sum to Δa · p


def test_the_five_conditions():
    c = rs.conditions("pirate")
    assert c.kind.tolist() == ["vector", "neurons", "components", "rotation_matched", "random"]
    assert c.condition_id.tolist() == [f"pirate|{k}|d0" for k in c.kind] and set(c.draw) == {0}


def _directions(g, n_neurons=128, layers=(0, 0, 1)):
    """A direction set as finish saves it: rank (layer, coef, contribution) and its columns."""
    v = {l: np.linalg.qr(g.standard_normal((n_neurons, n_neurons)))[0] for l in set(layers)}
    cols = np.stack([v[l][:, i] for i, l in enumerate(layers)], axis=1)
    return {"rank": pd.DataFrame({"layer": list(layers), "coef": g.standard_normal(len(layers)),
                                  "contribution": g.standard_normal(len(layers))}), "v": cols}


def test_rotated_condition_values():
    g = np.random.default_rng(6)
    delta, p = g.standard_normal((2, 128)), g.standard_normal((2, 128))
    sets = rs.neuron_sets("pirate", 0, delta, p, k=10)
    w = {l: torch.from_numpy(g.standard_normal((64, 128))) for l in range(2)}
    w_down = lambda l: w[l]
    conds = rs.conditions("pirate")
    cond = conds[conds.kind == "rotation_matched"].iloc[0].to_dict()
    d = _directions(g)
    # a set of directions is steered along as components are: c × V coef, scaled to the target norm
    kind, vals = rs.condition_values(cond, sets, {("rotation_matched", 0): d}, w_down, 9.0)
    want = rs.component_add_values({l: (d["v"][:, g_.index.to_numpy()], d["rank"].coef.to_numpy()[g_.index])
                                    for l, g_ in d["rank"].groupby("layer")})
    ratio = vals.value.to_numpy() / want.value.to_numpy()
    assert kind == "rows" and rs.write_vector(vals, w_down).norm().item() == pytest.approx(9.0)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)
    assert ratio[0] > 0


# ---- step 6b

def test_more_conditions_list_every_set_once():
    c = rs.more_conditions("alien")
    assert not c.duplicated(["kind", "k", "draw"]).any() and c.condition_id.is_unique
    for kind in ("neurons", "random", "rotation_matched", "components"):
        assert sorted(c[(c.kind == kind) & (c.draw == 0)].k) == [k for k in rs.SET_SIZES if k != 100]
    assert sorted(c[(c.kind == "random") & (c.k == 100)].draw) == [1, 2, 3, 4]
    assert sorted(c[(c.kind == "rotation_matched") & (c.k == 100)].draw) == [1, 2, 3, 4]
    assert c[c.kind.isin(["specific", "shared"])].k.tolist() == [50, 50] and (c.kind == "none").sum() == 1


def test_more_neuron_sets_draw_as_step_6_does():
    g = np.random.default_rng(0)
    delta, p = g.standard_normal((3, 200)), g.standard_normal((3, 200))
    s = rs.more_neuron_sets("alien", 4, delta, p)
    top = rs.neuron_sets("alien", 4, delta, p, 10)
    got = s[(s.k == 10) & (s.kind == "random")].reset_index(drop=True)
    pd.testing.assert_frame_equal(got.drop(columns="k"), top[top.kind == "random"].reset_index(drop=True))
    d3 = rs.neuron_sets("alien", 4, delta, p, 100, 3)
    got = s[(s.k == 100) & (s.draw == 3)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got.drop(columns="k"), d3[d3.kind == "random"].reset_index(drop=True))
    assert not s[(s.k == 100)].kind.eq("neurons").any()


def test_specificity_halves_split_by_the_roles_share_of_each_neuron():
    roles = ["a", "b"]
    delta, p = np.ones((2, 1, 120)), np.ones((2, 1, 120))
    delta[1, 0, :50] = 3.0   # role b uses neurons 0 to 49 three times as much as role a
    delta[1, 0, 50:100] = 1 / 3
    top = pd.DataFrame([{"role": r, "rank": i, "layer": 0, "neuron": i} for r in roles for i in range(100)])
    h = rs.specificity_halves(top, roles, delta, p, np.ones(2))
    assert h.groupby(["role", "kind"]).size().eq(50).all() and h.n_lists.eq(2).all()
    assert set(h[(h.role == "b") & (h.kind == "specific")].neuron) == set(range(50))
    assert set(h[(h.role == "a") & (h.kind == "specific")].neuron) == set(range(50, 100))
    assert h[(h.role == "b") & (h.neuron == 0)].specificity.item() == pytest.approx(0.75)


def test_step6b_on_the_tiny_model(tmp_path, tiny_qwen3_dir, upstream_checkout, monkeypatch):
    s6 = _step6()
    spec = importlib.util.spec_from_file_location("step6b", Path(__file__).parent.parent / "steps/6b_more_conditions.py")
    s6b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s6b)
    roles = ["alien", "sommelier"]
    monkeypatch.setenv("PN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PN_UPSTREAM_DIR", str(upstream_checkout))
    bases = {l: np.linalg.qr(np.random.default_rng(l).standard_normal((128, 128)))[0] for l in range(2)}
    _fake_inputs(tmp_path, roles, bases)
    monkeypatch.setattr(s6.sp, "layer_basis", lambda stats, rl, l, t: (None, bases[l]))
    monkeypatch.setattr(s6.sp, "roles_with_vectors", lambda d: roles)
    base = ["--target", "1"]
    instructions = upstream_checkout / "data" / "roles" / "instructions"
    assert (instructions / "sommelier.json").exists()
    assert s6.main(["prepare"] + base) == 0
    for part in (["components", "--layers", "0-1"], ["rotate", "--layers", "0-1"], ["finish"]):
        assert s6.main(part + base) == 0
    assert s6b.main(["prepare"] + base) == 0
    assert s6b.main(["components", "--layers", "0-1"] + base) == 0
    for d in (0,) + rs.MORE_DRAWS:
        assert s6b.main(["rotate", "--draw", str(d), "--layers", "0-1"] + base) == 0
    assert s6b.main(["finish"] + base) == 0
    out = tmp_path / "role_steering_more"
    sets = pd.read_parquet(out / "sets.parquet")
    assert sets[(sets.role == "alien") & (sets.kind == "specific")].shape[0] == 50
    z = np.load(out / "directions" / "alien.npz")
    assert z["rotation_matched|d3|cols"].shape == (128, 100)
    assert z["rotation_matched|d0|cols"].shape == (128, 256)   # the tiny model's 256 neurons, the most a set takes
    # draw 0's first directions in a layer, for a size's layer counts, are that layer's largest, as the top k rotate takes them
    q = rs.haar_rotation(128, [rs.ROTATION_SEED, 0, 1])
    arr = np.load(tmp_path / "split" / "arrays.npz")
    top = rs.top_rotated(q, arr["delta_a"][0, 1], arr["p"][0, 1], 128)
    rows = z["rotation_matched|d0|layer"] == 1
    assert z["rotation_matched|d0|direction"][rows].tolist() == top.direction.tolist()
    comps = np.load(out / "components" / "alien.npz")
    assert comps["cols"].shape == (128, 256) and np.all(np.diff(np.abs(comps["contribution"])) <= 1e-6)
    assert s6b.main(["generate", "--model-path", str(tiny_qwen3_dir), "--dtype", "float32", "--device", "cpu",
                     "--max-new-tokens", "4", "--roles", *roles] + base) == 0
    rep = pd.read_parquet(out / "replies" / "sommelier.parquet")
    n_conditions = len(rs.more_conditions("sommelier"))
    assert len(rep) == 10 * n_conditions
    # every condition's seeds are the role's, so that replies are paired across conditions
    assert rep.groupby(["question_index", "sample"]).seed.nunique().eq(1).all()
    from persona_neurons import judge

    sent = []
    monkeypatch.setattr(judge, "submit", lambda reqs, run_dir: sent.extend(reqs) or ["batch_0"])
    assert s6b.main(["judge", "submit"] + base) == 0
    assert len(sent) == 2 * len(rep)
    monkeypatch.setattr(judge, "collect", lambda ids, run_dir: {r["custom_id"]: "Score: 2" for r in sent})
    (out / "judge").mkdir(exist_ok=True)
    (out / "judge" / "batches.json").write_text(json.dumps(["batch_0"]))
    assert s6b.main(["judge", "collect"] + base) == 0
    # the shares of step 6's draw 0 are read by build
    assert s6b.main(["build"] + base) == 0
    sizes = pd.read_parquet(tmp_path / "role_steering_sizes" / "judged_replies.parquet")
    controls = pd.read_parquet(tmp_path / "role_steering_control_sizes" / "judged_replies.parquet")
    assert set(controls.kind) == {"rotation_matched", "components"} and (controls.draw == 0).all() and 100 not in set(controls.k)
    draws = pd.read_parquet(tmp_path / "role_steering_draws" / "judged_replies.parquet")
    none = pd.read_parquet(tmp_path / "role_steering" / "unsteered_sommelier.parquet")
    assert (sizes.draw == 0).all() and set(draws.draw) == set(rs.MORE_DRAWS) and len(none) == 10
    assert set(none.role) == {"sommelier"} and sizes.score.eq(2).all()
    assert len(sizes) + len(controls) + len(draws) + 2 * len(none) == 2 * len(rep)
    shares = pd.read_parquet(tmp_path / "role_steering_draws" / "shares.parquet")
    assert shares.groupby("kind").draw.nunique().eq(5).all() and shares.role.nunique() == 2


def test_a_size_of_the_control_directions():
    """A rotated set stored for the top 1,000 takes each layer's first, as many as the top k neurons have there; components
    take the k with the largest contribution, whichever layer they are in."""
    v = np.eye(64)[:, :6]
    rank = pd.DataFrame({"layer": [0, 0, 0, 1, 1, 1], "coef": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    sets = pd.DataFrame({"kind": "neurons", "k": 2, "layer": [0, 1], "neuron": [0, 0]})
    weights = {0: torch.eye(64), 1: torch.eye(64)}
    w_down = lambda l: weights[l]
    cond = {"kind": "rotation_matched", "draw": 0, "k": 2}
    _, vals = rs.condition_values(cond, sets, {("rotation_matched", 0): {"rank": rank, "v": v}}, w_down, 1.0)
    # the first of layer 0 and the first of layer 1
    vals = vals[vals.value != 0]
    assert set(zip(vals.layer, vals.neuron)) == {(0, 0), (1, 3)}
    cond = {"kind": "components", "draw": 0, "k": 2}
    _, vals = rs.condition_values(cond, sets, {("components", 0): {"rank": rank, "v": v}}, w_down, 1.0)
    vals = vals[vals.value != 0]
    assert set(zip(vals.layer, vals.neuron)) == {(0, 0), (0, 1)}   # the rank's first two
