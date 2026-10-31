#!/usr/bin/env python3
"""Step 5 (CPU): each role's persona vector at layer 18 (role minus default, with massive-activation tokens left out)
split into the contributions of the MLP neurons of layers 0 to 18, and of each layer's principal components
(`persona_neurons.split`). Three parts, writing into split/ in the data directory:

  split   the neuron split: arrays.npz (roles, delta_a, p, norm), top_neurons.parquet (each role's top 1,000
          neurons), the neurons' concentration curves and split.json (the largest gap between a vector and its
          parts, relative to its length)
  pca     the principal-component bases of some layers (--layers 0-4, or 5,6,7) and every role's contributions in
          them: pca/layer_LL.npz (roles, eigenvalues, contributions (n_roles, n_components))
  finish  concentration.parquet (role, basis, k from 1 to 1,000, share in %), once every layer has its bases

The roles are those with a vector in vectors/all/ (step 4). The neuron output weights are read from Qwen3-8B's
safetensors files at the fixed revision. Each part refuses to overwrite its outputs. Memory: about 16 GB for split;
for pca, about 20 GB and several CPU cores for 20 to 50 minutes per 5 layers.

    uv run steps/5_split.py split
    uv run steps/5_split.py pca --layers 0-4      (then 5-9, 10-14, 15-18)
    uv run steps/5_split.py finish
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config, split as sp
from persona_neurons.upstream import model_snapshot

K_MAX = 1000


def refuse(paths):
    clash = [str(p) for p in paths if Path(p).exists()]
    if clash:
        raise SystemExit(f"already exist: {clash}")


def parse_layers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run_split(args, data: Path, out: Path) -> None:
    refuse([out / "split.json"])
    out.mkdir(parents=True, exist_ok=True)
    t, roles = args.target, sp.roles_with_vectors(data / "vectors" / "all")
    d = sp.load(data / "neuron_stats", roles, target=t)
    model_dir = args.model_path or model_snapshot(config.MODEL_ID, config.MODEL_REVISION,
                                                  allow_patterns=["*.safetensors", "*.json"])
    w_down = [w for _, w in sp.down_proj_tensors(model_dir, t)]
    norm = np.linalg.norm(d.v, axis=1)
    p = sp.projections(w_down, d.v)
    gap = np.linalg.norm(d.v - sp.rebuild(d, w_down), axis=1) / norm
    c = sp.contributions(d.delta_a, p)
    np.savez(out / "arrays.npz", roles=np.array(roles), delta_a=d.delta_a, p=p, norm=norm)
    k_max = min(K_MAX, c[0].size)
    curves, tops = [], []
    for i, role in enumerate(roles):
        curves.append(pd.DataFrame({"role": role, "basis": "neurons", "k": np.arange(1, k_max + 1),
                                    "share": sp.top_curve(c[i], norm[i], k_max)}))
        top = sp.top_neurons(c[i], norm[i], k_max)
        tops.append(top.assign(role=role, delta=d.delta_a[i, top.layer, top.neuron]))
    pd.concat(curves, ignore_index=True).to_parquet(out / "concentration_neurons.parquet", index=False)
    pd.concat(tops, ignore_index=True).to_parquet(out / "top_neurons.parquet", index=False)
    (out / "split.json").write_text(json.dumps({
        "n_roles": len(roles), "n_default_replies": d.n_default, "n_replies_scored_2_or_3": d.n_roleplay,
        "target_layer": t, "k_max": k_max, "largest_gap_relative": float(gap.max())}, indent=2) + "\n")
    print(f"split: {len(roles)} roles; largest gap between a vector and its parts {gap.max():.2g} of its length")


def pca_file(out: Path, layer: int) -> Path:
    return out / "pca" / f"layer_{layer:02d}.npz"


def run_pca(args, data: Path, out: Path) -> None:
    layers = parse_layers(args.layers)
    refuse([pca_file(out, l) for l in layers])
    (out / "pca").mkdir(parents=True, exist_ok=True)
    arr = np.load(out / "arrays.npz")
    roles = sp.roles_with_vectors(data / "vectors" / "all")
    assert roles == arr["roles"].tolist(), "the role list differs from the split's"
    d = sp.load(data / "neuron_stats", roles, target=args.target, layers=layers, subsample=True)
    for i, l in enumerate(layers):
        vals, v = sp.pca_basis(d.sub_a[:, i].astype(np.float32))
        contrib = sp.pc_contributions(arr["delta_a"][:, l].T, arr["p"][:, l].T, v).T.astype(np.float32)
        np.savez(pca_file(out, l), roles=np.array(roles), eigenvalues=vals, contributions=contrib)
        print(f"pca: layer {l} done", flush=True)


def run_finish(args, data: Path, out: Path) -> None:
    refuse([out / "concentration.parquet"])
    arr = np.load(out / "arrays.npz")
    roles, norm = arr["roles"].tolist(), arr["norm"]
    k_max = json.loads((out / "split.json").read_text())["k_max"]
    files = [pca_file(out, l) for l in range(args.target + 1)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"principal components missing for {missing}")
    pcs = np.concatenate([np.load(f)["contributions"] for f in files], axis=1)  # (n_roles, all components)
    k_max = min(k_max, pcs.shape[1])
    comps = pd.concat([pd.DataFrame({"role": r, "basis": "principal components", "k": np.arange(1, k_max + 1),
                                     "share": sp.top_curve(pcs[i], norm[i], k_max)}) for i, r in enumerate(roles)])
    conc = pd.concat([pd.read_parquet(out / "concentration_neurons.parquet"), comps], ignore_index=True)
    conc.to_parquet(out / "concentration.parquet", index=False)
    print(f"finish: {out / 'concentration.parquet'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["split", "pca", "finish"])
    ap.add_argument("--layers", help="pca: the layers, e.g. 0-4 or 5,6,7")
    ap.add_argument("--target", type=int, default=config.TARGET_LAYER)
    ap.add_argument("--model-path", type=Path, help="a local model folder, in place of Qwen3-8B at the fixed revision")
    args = ap.parse_args(argv)
    data = config.data_dir()
    out = data / "split"
    if args.part == "pca" and not args.layers:
        ap.error("pca needs --layers")
    {"split": run_split, "pca": run_pca, "finish": run_finish}[args.part](args, data, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
