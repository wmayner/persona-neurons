#!/usr/bin/env python3
"""Step 4 (CPU): the role vectors and the default vector, by the authors' pipeline/4_vectors.py, unchanged.

A role's vector is the mean, over its replies scored 2 or 3, of each reply's mean residual; a role with fewer than 50
such replies gets none (the script's `--min_count` default). The authors' script keeps replies scored 3, so it reads
a copy of the scores with every 2 made a 3 (scores_2or3/). The default vector is the mean over the default replies.
It runs on the activations over all reply tokens and on those with massive-activation tokens left out:

    activations/all/      -> vectors/all/{role}.pt
    activations/without/  -> vectors/without/{role}.pt

The authors' script skips roles whose vector exists.

    uv run steps/4_vectors.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config
from persona_neurons.upstream import fetch_upstream


def scores_2_or_3(scores_dir: Path, out_dir: Path) -> None:
    """Each {role}.json with every score 2 made a 3; .missing.json lists are skipped, as are files that exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(scores_dir.glob("*.json")):
        if f.name.endswith(".missing.json") or (out_dir / f.name).exists():
            continue
        scores = json.loads(f.read_text())
        (out_dir / f.name).write_text(json.dumps({k: 3 if s == 2 else s for k, s in scores.items()}, indent=2))


def main() -> int:
    data = config.data_dir()
    checkout = fetch_upstream(config.upstream_dir())
    scores_2_or_3(data / "scores", data / "scores_2or3")
    for version in ("all", "without"):
        subprocess.run(
            [sys.executable, str(checkout / "pipeline" / "4_vectors.py"),
             "--activations_dir", str(data / "activations" / version), "--scores_dir", str(data / "scores_2or3"),
             "--output_dir", str(data / "vectors" / version)],
            check=True, cwd=checkout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
