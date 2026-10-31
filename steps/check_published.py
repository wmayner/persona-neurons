#!/usr/bin/env python3
"""Check (CPU): each role's loading on PC1 of persona space in Qwen3-8B (step 4's vectors over all reply tokens, layer 18)
beside its loading in Lu et al.'s published Qwen3-32B vectors (layer 32, downloaded). Writes check/pc1_loadings.parquet.

    uv run steps/check_published.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config
from persona_neurons import persona_space as ps
from persona_neurons.upstream import fetch_upstream, use_upstream


def main() -> int:
    data = config.data_dir()
    use_upstream(fetch_upstream(config.upstream_dir()))
    rows = []
    for model, vectors, layer in (
        ("Qwen3-8B", ps.load_vectors(data / "vectors" / "all"), config.TARGET_LAYER),
        ("Qwen3-32B", ps.download_published(data / "published"), ps.PUBLISHED_LAYER),
    ):
        pcs = ps.pca_roles(vectors, layer)
        rows.append(pd.DataFrame({"model": model, "role": pcs.roles, "loading": pcs.loadings[:, 0]}))
    out = data / "check" / "pc1_loadings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
