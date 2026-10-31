#!/usr/bin/env python3
"""Step 3 (GPU): each reply's mean activations, over all reply tokens and with the massive-activation tokens left out
(`persona_neurons.activations`). For each role in responses/, writes into the data directory:

- activations/all/{role}.pt: each reply's mean residual at every layer, as the authors' `pipeline/2_activations.py`
  writes it;
- activations/without/{role}.pt: the same with the massive-activation tokens left out;
- neuron_stats/{role}.pt: sums by score bucket of the neuron, attention, embedding and residual means (both versions)
  and the subsample;
- replies/{role}.parquet: one row per reply (role, key, score, n_tokens, n_left_out, largest, kept).

A role whose four outputs exist is skipped, and no output is overwritten. Roles other than default need their
scores (step 2).

    uv run steps/3_activations.py [--shard i/n] [--roles ROLE ...]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import jsonlines
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config
from persona_neurons.activations import MASSIVE_LAYER, MASSIVE_LINE, MASSIVE_NEURONS, extract_role
from persona_neurons.shards import shard
from persona_neurons.upstream import fetch_upstream, model_snapshot, use_upstream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("activations")


def save_new(obj, path: Path) -> None:
    """Write `obj` to `path` through a temporary file, refusing to replace an existing file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        obj.to_parquet(tmp, index=False)
    else:
        torch.save(obj, tmp)
    try:
        os.link(tmp, path)  # fails if `path` exists
    finally:
        tmp.unlink()


def outputs(data: Path, role: str) -> list[Path]:
    return [data / "activations" / "all" / f"{role}.pt", data / "activations" / "without" / f"{role}.pt",
            data / "neuron_stats" / f"{role}.pt", data / "replies" / f"{role}.parquet"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", help="i/n: the i-th of n parts of the roles (i from 0)")
    p.add_argument("--roles", nargs="+", help="only these roles")
    p.add_argument("--model-path", type=Path, help="a local model folder, in place of Qwen3-8B at the fixed revision")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--subsample-seed", type=int, default=20261027)
    p.add_argument("--subsample-n", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--layer", type=int, default=MASSIVE_LAYER, help=argparse.SUPPRESS)
    p.add_argument("--neurons", type=int, nargs="+", default=list(MASSIVE_NEURONS), help=argparse.SUPPRESS)
    p.add_argument("--line", type=float, default=MASSIVE_LINE, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    data = config.data_dir()
    responses_dir, scores_dir = data / "responses", data / "scores"
    for d in ("activations/all", "activations/without", "neuron_stats", "replies"):
        (data / d).mkdir(parents=True, exist_ok=True)
    roles = [f.stem for f in responses_dir.glob("*.jsonl") if not args.roles or f.stem in args.roles]
    todo = [r for r in shard(roles, args.shard) if not all(o.exists() for o in outputs(data, r))]
    log.info("%d roles in this shard, %d to do", len(shard(roles, args.shard)), len(todo))
    unscored = [r for r in todo if "default" not in r and not (scores_dir / f"{r}.json").exists()]
    if unscored:
        log.error("no scores for %d roles: %s", len(unscored), " ".join(unscored))
        return 1
    if not todo:
        return 0

    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    from assistant_axis.internals import ProbingModel

    model_path = args.model_path or model_snapshot(config.MODEL_ID, config.MODEL_REVISION)
    pm = ProbingModel(str(model_path), dtype=getattr(torch, args.dtype))  # as the authors load it
    log.info("loaded %s (%d layers) on %s", model_path, len(pm.get_layers()), pm.device)
    meta = {"model": str(args.model_path) if args.model_path else f"{config.MODEL_ID}@{config.MODEL_REVISION}",
            "dtype": args.dtype, "batch_size": args.batch_size, "max_length": args.max_length,
            "upstream_commit": config.UPSTREAM_COMMIT, "layer": args.layer, "neurons": args.neurons, "line": args.line,
            "subsample_seed": args.subsample_seed, "subsample_n": args.subsample_n}

    for role in todo:
        with jsonlines.open(responses_dir / f"{role}.jsonl") as reader:
            responses = list(reader)
        scored = "default" not in role
        scores = json.loads((scores_dir / f"{role}.json").read_text()) if scored else {}
        s = extract_role(pm.model, pm.tokenizer, responses, scores, batch_size=args.batch_size,
                         subsample_seed=args.subsample_seed, subsample_n=args.subsample_n, layer=args.layer,
                         neurons=args.neurons, line=args.line, max_length=args.max_length, upstream_checkout=checkout)
        rows = pd.DataFrame(s.replies)
        rows = rows.assign(role=role, score=rows.key.map(lambda k: scores.get(k)))
        act_all, act_without, stats, replies = outputs(data, role)
        new = {
            act_all: s.residual_all or None,  # the authors' step writes nothing for a role with no activations
            act_without: s.residual_without or None,
            stats: {"role": role, "scored": scored, "sums": s.sums, "sums_all": s.sums_all, "subsample": s.subsample,
                    "dropped": s.dropped, "n_responses": len(responses), **meta},
            replies: rows,
        }
        for path, obj in new.items():
            if obj is not None and not path.exists():
                save_new(obj, path)
        log.info("%s: %d replies, %d with a token left out, %d with every token left out", role, len(rows),
                 int((rows.n_left_out > 0).sum()), int((~rows.kept).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
