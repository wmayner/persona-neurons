"""Step 5: the split of each role's persona vector into neuron writes and principal-component writes
(`persona_neurons.split`, `steps/5_split.py`), on step 3's output for the tiny random Qwen3."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from test_activations import run_step, write_role

ROOT = Path(__file__).resolve().parents[1]


def test_component_contributions_add_up_to_the_neurons():
    from persona_neurons import split as sp

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 12))  # more replies than neurons: a complete basis
    vals, v = sp.pca_basis(x)
    assert np.all(np.diff(vals) <= 1e-12) and np.allclose(v.T @ v, np.eye(12))
    delta, proj = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    c = sp.pc_contributions(delta, proj, v)
    assert np.allclose(c.sum(0), (delta * proj).sum(0))


def test_top_curve_and_top_neurons():
    from persona_neurons import split as sp

    c = np.array([[0.5, -2.0, 0.1], [1.0, 0.0, -0.3]])
    assert np.allclose(sp.top_curve(c, 4.0, 3), [100 * -2 / 4, 100 * -1 / 4, 100 * -0.5 / 4])
    top = sp.top_neurons(c, 4.0, 2)
    assert top[["layer", "neuron"]].values.tolist() == [[0, 1], [1, 0]]
    assert np.allclose(top.share, [-50, 25])


@pytest.fixture(scope="module")
def data(tmp_path_factory, tiny_qwen3_dir, upstream_checkout):
    """Step 3's output for two roles and the default, and a vectors/all/ folder naming the two roles."""
    data = tmp_path_factory.mktemp("split") / "data"
    write_role(data, "pirate", scores=[3, 2, 1])
    write_role(data, "monk", scores=[1, 3, 2])  # the replies are the default's: a role must not keep all three
    write_role(data, "default", label="default")
    result = run_step(data, tiny_qwen3_dir, upstream_checkout)
    assert result.returncode == 0, result.stderr
    (data / "vectors" / "all").mkdir(parents=True)
    for role in ("default", "monk", "pirate"):
        torch.save({"role": role, "vector": torch.zeros(2, 64)}, data / "vectors" / "all" / f"{role}.pt")
    return data


def run_part(data, tiny_qwen3_dir, *args):
    cmd = [sys.executable, str(ROOT / "steps" / "5_split.py"), *args, "--target", "1", "--model-path", str(tiny_qwen3_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PN_DATA_DIR": str(data)})
    assert result.returncode == 0, result.stderr
    return result


def test_split_steps(data, tiny_qwen3_dir):
    from persona_neurons import split as sp

    run_part(data, tiny_qwen3_dir, "split")
    out = data / "split"
    summary = json.loads((out / "split.json").read_text())
    assert summary["n_roles"] == 2 and summary["n_replies_scored_2_or_3"] == {"monk": 2, "pirate": 2}
    assert summary["largest_gap_relative"] < 1e-4  # the parts rebuild the vector
    arr = np.load(out / "arrays.npz")
    assert arr["roles"].tolist() == ["monk", "pirate"] and arr["delta_a"].shape == (2, 2, 128)

    # the persona vector is the role's mean residual over replies scored 2 or 3 minus the default's
    stats = {r: torch.load(data / "neuron_stats" / f"{r}.pt", weights_only=False) for r in ("pirate", "default")}
    s, d = stats["pirate"]["sums"], stats["default"]["sums"]["missing"]
    v = ((s["2"]["resid"][1] + s["3"]["resid"][1]) / 2 - d["resid"][1] / d["count"]).numpy()
    assert np.isclose(arr["norm"][1], np.linalg.norm(v))

    # what all neurons write, plus the attention layers and the embedding, is the vector's whole length
    loaded = sp.load(data / "neuron_stats", ["monk", "pirate"], target=1)
    unit = loaded.v / arr["norm"][:, None]
    rest = np.einsum("rld,rd->r", loaded.delta_attn, unit) + np.einsum("rd,rd->r", loaded.delta_embed, unit)
    neurons = (arr["delta_a"].astype(np.float64) * arr["p"]).sum((1, 2))
    assert np.allclose(neurons + rest, arr["norm"], rtol=1e-4)

    tops = pd.read_parquet(out / "top_neurons.parquet")
    assert set(tops.role) == {"monk", "pirate"} and tops.groupby("role").size().tolist() == [256, 256]

    run_part(data, tiny_qwen3_dir, "pca", "--layers", "0-1")
    for layer in (0, 1):
        saved = np.load(out / "pca" / f"layer_{layer:02d}.npz")
        vals, _ = sp.layer_basis(data / "neuron_stats", ["monk", "pirate"], layer, target=1)
        assert np.allclose(saved["eigenvalues"], vals) and saved["contributions"].shape == (2, len(vals))

    run_part(data, tiny_qwen3_dir, "finish")
    conc = pd.read_parquet(out / "concentration.parquet")
    assert set(conc.basis) == {"neurons", "principal components"}
    first = conc[(conc.role == "pirate") & (conc.basis == "neurons") & (conc.k == 1)].share.item()
    assert np.isclose(first, tops[(tops.role == "pirate") & (tops["rank"] == 1)].share.item())
