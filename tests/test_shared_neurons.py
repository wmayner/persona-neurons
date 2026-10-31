"""The two tables of the shared-neuron figures."""
import numpy as np
import pandas as pd
import pytest
import torch

from persona_neurons import shared_neurons as sn


def test_pooled_sd_is_around_each_groups_own_mean():
    x = np.array([[1.0], [3.0], [10.0], [14.0]])
    got = sn.pooled_sd(x, np.array(["a", "a", "b", "b"]))
    assert got[0] == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))
    # the groups need not be in order
    assert sn.pooled_sd(x[[0, 2, 1, 3]], np.array(["a", "b", "a", "b"]))[0] == pytest.approx(got[0])


def test_away_is_the_cosine_with_the_mean_role():
    v = np.array([[1.0, 0.0], [1.0, 2.0]])
    mean = v.mean(0)
    assert sn.away_from_assistant(v)[0] == pytest.approx(mean[0] / np.linalg.norm(mean))


def test_overlap_counts_neurons_in_common_with_the_average_other_role():
    top = pd.DataFrame({"role": ["a", "a", "b", "b", "c", "c"], "layer": 0, "neuron": [1, 2, 1, 3, 4, 5]})
    t = sn.role_table(["a", "b", "c"], np.zeros(3), top).set_index("role")
    assert t.mean_others["a"] == 0.5 and t.mean_others["c"] == 0
    assert t.overlap["a"] == pytest.approx(100 / 2 * 0.5)


def test_neuron_cells():
    delta = np.zeros((2, 1, 3))
    p = np.zeros((2, 1, 3))
    delta[:, 0, 2], p[:, 0, 2] = [2.0, -4.0], [0.5, 0.25]
    top = pd.DataFrame({"role": ["b"], "rank": [7], "layer": [0], "neuron": [2]})
    t = sn.neuron_cells(["a", "b"], delta, p, np.array([10.0, 20.0]), np.full((1, 3), 2.0), top, [(0, 2)]).set_index("role")
    assert t.x["a"] == 1 and t.x["b"] == -2 and t.y["a"] == 10 and t.y["b"] == -5
    assert t["rank"]["a"] == 0 and t["rank"]["b"] == 7


def test_step_5b(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import json

    import torch

    spec = importlib.util.spec_from_file_location("step5b", Path(__file__).parent.parent / "steps/5b_shared_neurons.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    monkeypatch.setenv("PN_DATA_DIR", str(tmp_path))
    roles, shape = ["alien", "sommelier"], (19, 12288)
    g = np.random.default_rng(0)
    (tmp_path / "split").mkdir()
    (tmp_path / "neuron_stats").mkdir()
    (tmp_path / "scores").mkdir()
    np.savez(tmp_path / "split" / "arrays.npz", roles=np.array(roles), norm=np.array([2.0, 4.0]),
             delta_a=np.ones((2, *shape), np.float32), p=np.full((2, *shape), 0.5, np.float32))
    pd.DataFrame({"role": ["alien", "sommelier"], "rank": [1, 1], "layer": [9, 9], "neuron": [10358, 10358]}).to_parquet(
        tmp_path / "split" / "top_neurons.parquet")
    bucket = lambda resid: {"count": 1, "pos": torch.zeros(shape), "neg": torch.zeros(shape), "attn": torch.zeros(19, 8),
                            "embed": torch.zeros(8), "resid": torch.full((36, 8), resid)}
    part = lambda value: torch.stack([torch.full(shape, value, dtype=torch.float32), torch.zeros(shape)])
    for role, resid in (("default", 1.0), ("alien", 2.0), ("sommelier", 4.0)):
        sums = {"missing": bucket(resid)} if role == "default" else {"2": bucket(resid), "3": bucket(resid)}
        torch.save({"scored": role != "default", "sums": sums, "subsample": {"k0": part(0.0), "k1": part(2.0), "k2": part(50.0)}},
                   tmp_path / "neuron_stats" / f"{role}.pt")
    for role in roles:   # the third reply scores 1, so it is left out of the SD
        (tmp_path / "scores" / f"{role}.json").write_text(json.dumps({"k0": 3, "k1": 2, "k2": 1}))
    assert s.main(["inputs"]) == 0
    with np.load(tmp_path / "split" / "shared_inputs.npz") as z:
        assert z["roles"].tolist() == roles and z["v"][1][0] == 3.0
        # the default's three replies around their mean 52 / 3, and each role's two around 1
        want = np.sqrt((((np.array([0, 2, 50]) - 52 / 3) ** 2).sum() + 2 * 2.0) / 7)
        assert z["sd"][9, 10358] == pytest.approx(want)
    assert s.main(["tables"]) == 0
    t = pd.read_parquet(tmp_path / "shared_neurons" / "roles.parquet").set_index("role")
    assert t.mean_others["alien"] == 1 and t.overlap["alien"] == 100
    c = pd.read_parquet(tmp_path / "shared_neurons" / "sommelier_neurons.parquet")
    assert len(c) == 6 and c[(c.neuron == 10358) & (c.role == "sommelier")]["rank"].iloc[0] == 1
    assert c[c.neuron == 288]["rank"].eq(0).all()
    with pytest.raises(SystemExit):
        s.main(["tables"])


def test_on_axis_tables():
    g = np.random.default_rng(1)
    roles = [f"r{i}" for i in range(6)]
    weights = {l: torch.from_numpy(g.standard_normal((16, 40))).float() for l in range(2)}
    delta = g.standard_normal((6, 2, 40)).astype(np.float32)
    v = g.standard_normal((6, 16)) + 3.0
    rows = [{"role": r, "kind": "shared" if n < 3 else "specific", "layer": n % 2, "neuron": n} for r in roles for n in range(6)]
    halves = pd.DataFrame(rows)
    t = sn.on_axis_table(roles, v, delta, halves, weights.__getitem__, 2, seed=7)
    assert len(t) == 6 * (2 + 2 * 2) and t[list(sn.COSINES)].abs().le(1 + 1e-9).all().all()
    # cos_axis is the cosine of the write with the unit mean of the persona vectors
    s = halves[(halves.role == "r2") & (halves.kind == "shared")]
    w = sum(float(delta[2, l, n]) * weights[l][:, n].double() for l, n in zip(s.layer, s.neuron))
    want = float(w @ torch.tensor(sn.unit_mean(v)) / w.norm())
    assert t[(t.role == "r2") & (t.set == "shared")].cos_axis.item() == pytest.approx(want, rel=1e-5)
    summary, paired = sn.on_axis_summaries(t)
    assert len(summary) == 4 * len(sn.COSINES) and len(paired) == 4 * len(sn.COSINES)
    top = halves.assign(delta=delta[halves.role.map({r: i for i, r in enumerate(roles)}), halves.layer, halves.neuron], n_lists=halves.neuron + 1, specificity=(halves.neuron + 1) / 10)
    neurons, rho = sn.on_axis_neurons(roles, v, top, weights.__getitem__)
    assert len(neurons) == len(top) and len(rho) == 6 and neurons.cos_axis.abs().le(1).all()
