#!/usr/bin/env python3
"""Step 5b: how roles share neurons (`persona_neurons.shared_neurons`), the tables of the shared-neuron figures. Parts:

  inputs  (CPU) split/shared_inputs.npz: each role's vector as the split has it (without massive-activation tokens),
          and each neuron's SD of its per-reply mean activation around the reply's role, pooled over the roles and the
          default, over the subsample of replies step 3 keeps whole (role replies scored 2 or 3 only). Reads
          neuron_stats/ (step 3) and scores/ (step 2); about 16 GB of memory
  tables  (CPU) shared_neurons/roles.parquet and shared_neurons/sommelier_neurons.parquet. Reads split/
  axis    (CPU) shared_neurons/on_axis*.parquet: how the two halves' direct writes line up with the Assistant Axis,
          against random neurons in the same layers, and each top 100 neuron's write against how many lists hold it.
          Reads split/, vectors/without/ and the model's down_proj weights

    uv run steps/5b_shared_neurons.py inputs
    uv run steps/5b_shared_neurons.py tables
    uv run steps/5b_shared_neurons.py axis
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config, role_steering as rs, shared_neurons as sn, split as sp
from persona_neurons.upstream import model_snapshot

SOMMELIER_NEURONS = [(10, 3032), (10, 288), (9, 10358)]   # (layer, neuron): from one in 147 roles' top 100 to one in the sommelier's alone
K = 100
N_DRAWS = 5   # random sets per role and half


def refuse(paths):
    clash = [str(p) for p in paths if Path(p).exists()]
    if clash:
        raise SystemExit(f"already exist: {clash}")


def run_inputs(args, data: Path) -> None:
    out = data / "split" / "shared_inputs.npz"
    refuse([out])
    layers = list(range(args.target + 1))
    with np.load(data / "split" / "arrays.npz") as z:
        roles = z["roles"].tolist()
    rows, groups, vectors = [], [], []
    default = None
    for role in ["default"] + roles:
        s = torch.load(data / "neuron_stats" / f"{role}.pt", map_location="cpu", weights_only=False)
        scores = None if role == "default" else json.loads((data / "scores" / f"{role}.json").read_text())
        for key, x in zip(sorted(s["subsample"]), sp._subsample(s, layers)):
            if scores is None or scores.get(key) in (2, 3):
                rows.append(x)
                groups.append(role)
        if scores is None:
            default = sp._means(s["sums"]["missing"])["resid"][args.target]
        else:
            vectors.append(sp._means(sp._merge([s["sums"][k] for k in sp.ROLEPLAY_SCORES]))["resid"][args.target] - default)
    sub, groups = np.stack(rows), np.array(groups)
    sd = np.stack([sn.pooled_sd(sub[:, i], groups) for i in range(len(layers))]).astype(np.float32)
    np.savez(out, roles=np.array(roles), sd=sd, v=np.stack(vectors).astype(np.float64))
    print(f"inputs: {len(rows)} replies of {len(set(groups))} groups", flush=True)


def run_tables(args, data: Path) -> None:
    out = data / "shared_neurons"
    refuse([out / "roles.parquet", out / "sommelier_neurons.parquet"])
    with np.load(data / "split" / "arrays.npz") as z:
        arr = {k: z[k] for k in ("roles", "delta_a", "p", "norm")}
    roles = arr["roles"].tolist()
    top = pd.read_parquet(data / "split" / "top_neurons.parquet", columns=["role", "rank", "layer", "neuron"])
    top = top[top["rank"] <= K]
    with np.load(data / "split" / "shared_inputs.npz") as z:
        assert z["roles"].tolist() == roles, "the role list differs from the split's"
        sd, v = z["sd"], z["v"]
    out.mkdir(exist_ok=True)
    sn.role_table(roles, sn.away_from_assistant(v), top).to_parquet(out / "roles.parquet", index=False)
    sn.neuron_cells(roles, arr["delta_a"], arr["p"], arr["norm"], sd, top, SOMMELIER_NEURONS).to_parquet(
        out / "sommelier_neurons.parquet", index=False)
    print(f"tables: {len(roles)} roles", flush=True)


def run_axis(args, data: Path) -> None:
    out = data / "shared_neurons"
    refuse([out / f"on_axis{x}.parquet" for x in ("", "_summary", "_paired", "_neurons", "_spearman")])
    with np.load(data / "split" / "arrays.npz") as z:
        roles, delta, p, norm = z["roles"].tolist(), z["delta_a"], z["p"], z["norm"]
    default = torch.load(data / "vectors" / "without" / "default.pt", map_location="cpu", weights_only=False)["vector"].double()[args.target]
    v = np.stack([(torch.load(data / "vectors" / "without" / f"{r}.pt", map_location="cpu", weights_only=False)["vector"].double()[args.target]
                   - default).numpy() for r in roles])
    model_dir = args.model_path or model_snapshot(config.MODEL_ID, config.MODEL_REVISION, allow_patterns=["*.safetensors", "*.json"])
    weights = {l: torch.from_numpy(w).float() for l, w in sp.down_proj_tensors(model_dir, args.target)}
    top = pd.read_parquet(data / "split" / "top_neurons.parquet", columns=["role", "rank", "layer", "neuron"])
    halves = rs.specificity_halves(top[top["rank"] <= K], roles, delta, p, norm)
    t = sn.on_axis_table(roles, v, delta, halves, weights.__getitem__, N_DRAWS, rs.RANDOM_SEED)
    summary, paired = sn.on_axis_summaries(t)
    index = {r: i for i, r in enumerate(roles)}
    halves = halves.assign(delta=delta[halves.role.map(index), halves.layer, halves.neuron])
    neurons, rho = sn.on_axis_neurons(roles, v, halves[["role", "kind", "layer", "neuron", "delta", "n_lists", "specificity"]], weights.__getitem__)
    out.mkdir(exist_ok=True)
    for name, table in (("", t), ("_summary", summary), ("_paired", paired), ("_neurons", neurons), ("_spearman", rho)):
        table.to_parquet(out / f"on_axis{name}.parquet", index=False)
    print(summary[summary.metric == "cos_axis"].round(3).to_string(index=False), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["inputs", "tables", "axis"])
    ap.add_argument("--model-path", type=Path, help="axis: a local folder with the model's safetensors")
    ap.add_argument("--target", type=int, default=config.TARGET_LAYER)
    args = ap.parse_args(argv)
    {"inputs": run_inputs, "tables": run_tables, "axis": run_axis}[args.part](args, config.data_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
