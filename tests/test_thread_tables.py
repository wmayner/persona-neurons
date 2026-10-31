"""The per-role tables (persona_neurons.thread_tables), on small made-up replies."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from persona_neurons import thread_tables as tt


def replies(role, kind, draw, scores, k=None):
    d = {"role": role, "kind": kind, "draw": draw, "question_index": range(len(scores)), "sample": 0, "score": scores}
    return pd.DataFrame(d if k is None else {**d, "k": k})


def load_numbers():
    spec = importlib.util.spec_from_file_location("print_numbers", Path(__file__).parent.parent / "figures/print_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shares_leave_out_unscored_replies():
    t = tt.in_role_shares(replies("a", "neurons", 0, [3, 2, 1, np.nan]), ["role"]).iloc[0]
    assert t.n == 3 and t.in_role == pytest.approx(200 / 3) and t.fully == pytest.approx(100 / 3)


def test_mean_ci_is_seeded_and_brackets_the_mean():
    v = np.arange(20.0)
    assert tt.mean_ci(v) == tt.mean_ci(v)
    mean, lo, hi = tt.mean_ci(v)
    assert lo < mean < hi and mean == v.mean()


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("PN_DATA_DIR", str(tmp_path))
    (tmp_path / "role_steering").mkdir()
    (tmp_path / "role_steering_draws").mkdir()
    (tmp_path / "role_steering_sizes").mkdir()
    pd.concat([replies("a", "vector", 0, [3, 3]), replies("a", "neurons", 0, [3, 0]), replies("a", "random", 0, [3, 3])]
              ).to_parquet(tmp_path / "role_steering" / "judged_replies.parquet")
    replies("a", "random", 1, [0, 0]).to_parquet(tmp_path / "role_steering_draws" / "judged_replies.parquet")
    pd.DataFrame({"role": "a", "kind": ["neurons", "random"], "draw": 0, "share": [6.0, 90.0]}
                 ).to_parquet(tmp_path / "role_steering" / "shares.parquet")
    pd.DataFrame({"role": "a", "kind": "random", "draw": [0, 1], "share": [1.0, 3.0]}
                 ).to_parquet(tmp_path / "role_steering_draws" / "shares.parquet")
    replies("a", "neurons", 0, [3, 3], k=10).to_parquet(tmp_path / "role_steering_sizes" / "judged_replies.parquet")
    return tmp_path


def test_five_draws_are_averaged_within_a_role(data):
    t = tt.five_draw_per_role().set_index("kind")
    assert t.loc["random", "in_role"] == 50 and t.loc["random", "draws"] == 2 and t.loc["random", "n"] == 4
    assert t.loc["neurons", "in_role"] == 50 and t.loc["neurons", "draws"] == 1


def test_contribution_takes_the_drawn_sets_from_the_draws_folder(data):
    t = tt.contribution_per_role().set_index("kind")
    assert t.loc["random", "share"] == 2.0 and t.loc["neurons", "share"] == 6.0


def test_set_sizes_use_draw_0_and_give_step_6_its_k(data):
    t = tt.set_size_per_role().set_index(["kind", "k"]).in_role
    assert t[("vector", 0)] == 100 and t[("neurons", 100)] == 50 and t[("neurons", 10)] == 100
    assert t[("random", 100)] == 100   # draw 0 alone: the draws folder is not read here


def test_set_size_folders_may_be_missing(data):
    (data / "role_steering_sizes" / "judged_replies.parquet").unlink()
    assert set(tt.set_size_per_role().k) == {0, 100}


def test_numbers_tables():
    nb = load_numbers()
    per_role = pd.DataFrame({"role": ["a", "b"] * 5, "in_role": [50.0, 70.0, 40.0, 60.0, 10.0, 12.0, 8.0, 8.0, 4.0, 6.0],
                             "fully": 1.0, "n": 10, "draws": 1,
                             "kind": sorted(nb.CONDITIONS * 2, key=nb.CONDITIONS.index)})
    s = nb.steering(per_role)
    assert s.kind.tolist() == nb.CONDITIONS
    assert s.set_index("kind").in_role["vector"] == 60 and s.roles.eq(2).all()
    p = nb.paired(per_role, [("vector", "neurons")]).iloc[0]
    assert p["mean"] == 10 and p.ahead == 2 and p.tied == 0 and p.behind == 0
    c = nb.sommelier_counts(replies("sommelier", "neurons", 0, [3, 2, 1, np.nan])).iloc[0]
    assert (c.has_traits, c.fully, c.judged) == (2, 1, 3)


def test_single_neuron_shares_pool_layers_in_the_drawn_sign():
    nb = load_numbers()
    share = pd.DataFrame({"kind": ["neuron", "neuron", "neuron", "role", "role"], "sign": ["both", "both", "both", "+", "-"],
                          "layer": [7, 8, 20, 7, 7], "n": [10, 10, 10, 10, 10], "k": [1, 3, 10, 5, 0]})
    natural = share.assign(k=0)
    t = nb.single_neuron_shares(share, natural).set_index("kind")
    assert t.loc["neuron", "percent"] == 20 and t.loc["neuron", "directions"] == 20
    assert t.loc["role", "percent"] == 50 and t.loc["neuron", "percent_natural"] == 0


def test_set_size_with_five_draws_replaces_only_the_drawn_sets_at_100(monkeypatch):
    rows = [{"kind": k, "k": n, "measure": "in_role", "mean": 1.0, "lo": 0.0, "hi": 2.0} for k in ("neurons", "random") for n in (50, 100)]
    monkeypatch.setattr(tt, "set_size_summary", lambda: pd.DataFrame(rows))
    per_role = pd.DataFrame({"role": ["a", "b", "a", "b"], "kind": ["random", "random", "rotation_matched", "rotation_matched"], "in_role": [10.0, 20.0, 30.0, 40.0]})
    monkeypatch.setattr(tt, "five_draw_per_role", lambda: per_role)
    monkeypatch.setattr(tt, "DRAWN", ["random"])
    t = tt.set_size_with_five_draws().set_index(["kind", "k"])
    assert t.loc[("random", 100), "mean"] == 15 and t.loc[("random", 100), "draws"] == 5
    assert t.loc[("random", 50), "mean"] == 1 and t.loc[("neurons", 100), "mean"] == 1 and t.loc[("neurons", 100), "draws"] == 1
