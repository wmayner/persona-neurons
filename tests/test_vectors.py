"""Step 4: the authors' 4_vectors.py on scores with every 2 made a 3, for both versions of the activations."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_vectors_are_means_over_scores_2_and_3(tmp_path, upstream_checkout):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    keys = [f"pos_p{p}_q{q}" for p in range(5) for q in range(14)]  # 70 replies
    scores = {k: int(s) for k, s in zip(keys, rng.choice([0, 1, 2, 2, 3, 3, 3, 3, 3, 3], len(keys)))}  # 50 or more scored 2 or 3
    scores[keys[0]], scores[keys[1]] = 2, 3
    (data / "scores").mkdir(parents=True)
    (data / "scores" / "pirate.json").write_text(json.dumps(scores))
    (data / "scores" / "pirate.missing.json").write_text(json.dumps({"pos_p9_q9": None}))
    acts = {}
    for version in ("all", "without"):
        d = data / "activations" / version
        d.mkdir(parents=True)
        acts[version] = {k: torch.tensor(rng.normal(size=(3, 8)), dtype=torch.float32) for k in keys}
        torch.save(acts[version], d / "pirate.pt")
        torch.save({f"default_p0_q{q}": torch.tensor(rng.normal(size=(3, 8)), dtype=torch.float32) for q in range(4)},
                   d / "default.pt")
    env = {**os.environ, "PN_DATA_DIR": str(data), "PN_UPSTREAM_DIR": str(upstream_checkout)}
    result = subprocess.run([sys.executable, str(ROOT / "steps" / "4_vectors.py")], capture_output=True, text=True,
                            cwd=ROOT, env=env)
    assert result.returncode == 0, result.stderr
    remapped = json.loads((data / "scores_2or3" / "pirate.json").read_text())
    assert remapped == {k: 3 if s == 2 else s for k, s in scores.items()}
    assert not (data / "scores_2or3" / "pirate.missing.json").exists()
    kept = [k for k, s in scores.items() if s in (2, 3)]
    assert len(kept) >= 50  # the authors' --min_count
    for version in ("all", "without"):
        v = torch.load(data / "vectors" / version / "pirate.pt", weights_only=False)["vector"]
        expected = torch.stack([acts[version][k] for k in kept]).mean(0)
        assert torch.allclose(v.float(), expected, atol=1e-5)
        assert (data / "vectors" / version / "default.pt").exists()
