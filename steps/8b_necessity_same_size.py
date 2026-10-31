#!/usr/bin/env python3
"""Step 8b: under a role's own system prompt, the top 100 neurons' removal against removals of the same size. Writes into
necessity_same_size/ in the data directory.

  prepare  (CPU) alpha.parquet (the top 100's measured movement along v̂_r as a fraction of |v_r|), sets.parquet and
           lower_sets.parquet (the lower-ranked neurons), alone.parquet (step 8's replies to the role prompt alone) and
           assistant.parquet (step 6b's unsteered replies)
  score    (GPU) every condition's teacher-forced log-probability of the role's 10 replies and 10 Assistant replies, under
           the role's prompt: forced/, sizes_forced/
  generate (GPU) the slice and the lower-ranked sets' replies, as step 8 does, with --changes: replies/, sizes/, changes/
  judge    (API) `judge submit`, later `judge collect`: judge/, labels.parquet
  build    (CPU) judged_replies.parquet
  analyze  (CPU) per_reply, per_role, paired, share_of_vector_drop, sizes, judged_per_role, judged_paired, sizes_measured and
           sizes_summary (parquet), in --to if given

`--roles ROLE ...` limits the parts that take a model or the judge to those roles.
"""
import argparse
import contextlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config, necessity as nc, role_steering as rs
from persona_neurons.shards import shard
from persona_neurons.upstream import fetch_upstream, use_upstream

_spec = importlib.util.spec_from_file_location("step8", Path(__file__).with_name("8_necessity.py"))
s8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s8)
s6 = s8.s6

DIRECTION_SEED = 20261112   # the random directions: np.random.default_rng([DIRECTION_SEED, role index])
TABLES = ("per_reply", "per_role", "paired", "share_of_vector_drop", "sizes", "judged_per_role", "judged_paired",
          "sizes_measured", "sizes_summary")


def run_prepare(args, data: Path, first: Path, out: Path) -> None:
    names = ("alpha.parquet", "sets.parquet", "lower_sets.parquet", "alone.parquet", "assistant.parquet")
    s6.refuse([out / n for n in names])
    out.mkdir(parents=True, exist_ok=True)
    roles = s6.steered_roles(data / "role_steering")
    per = pd.read_parquet(first / "per_role.parquet")
    top = per[(per.patch == "subtract") & (per.set == "neurons")].set_index("role")
    vec = np.load(data / "role_steering" / "vectors.npz")
    norm = pd.Series(np.linalg.norm(vec["v_without"].astype(np.float64), axis=1), index=vec["roles"].tolist())
    alpha = pd.DataFrame({"role": roles, "along_v": top.reply_along_v.loc[roles].to_numpy(), "v_norm": norm.loc[roles].to_numpy()})
    alpha["alpha"] = -alpha.along_v / alpha.v_norm
    alpha.to_parquet(out / "alpha.parquet", index=False)

    split = s6.load_split(data)
    index = {r: i for i, r in enumerate(split["roles"].tolist())}
    parts, info = [], []
    for role in roles:
        delta, p = split["delta_a"][index[role]], split["p"][index[role]]
        for kind, same in (("lower", False), ("lower_same_sign", True)):
            s, x = nc.lower_ranked(delta, p, same_sign=same)
            parts.append(s[["layer", "neuron", "delta"]].assign(role=role, kind=kind, draw=0))
            info.append({"role": role, "kind": kind} | x)
    pd.concat(parts, ignore_index=True).to_parquet(out / "sets.parquet", index=False)
    pd.DataFrame(info).to_parquet(out / "lower_sets.parquet", index=False)

    j = pd.read_parquet(first / "judged_replies.parquet")
    j[j.patch == "prompt"][["role", "question_index", "sample", "seed", "response", "score"]].to_parquet(out / "alone.parquet", index=False)
    r = s6.read_replies(data / "role_steering_more", None)
    r[(r.kind == "none") & r.role.isin(roles)][["role", "question_index", "sample", "seed", "response"]].to_parquet(
        out / "assistant.parquet", index=False)
    print(f"prepare: {len(roles)} roles, mean alpha {alpha.alpha.mean():.3f}", flush=True)


def role_inputs(data: Path, out: Path, role: str) -> dict:
    inp = s8.role_inputs(data, role)
    lower = pd.read_parquet(out / "sets.parquet", filters=[("role", "==", role)])
    inp["sets"] = pd.concat([inp["sets"], lower], ignore_index=True)
    return inp


def alphas(out: Path) -> pd.Series:
    return pd.read_parquet(out / "alpha.parquet").set_index("role").alpha


def random_direction(role_index: int, d_model: int) -> torch.Tensor:
    x = np.random.default_rng([DIRECTION_SEED, role_index]).standard_normal(d_model)
    return torch.tensor(x / np.linalg.norm(x), dtype=torch.float64)


def score_role(model, tok, role: str, inp: dict, alpha: float, convs: dict, replies: pd.DataFrame, batch_size: int, layer: int):
    """(one row per condition and reply, one row per condition with what it adds at `layer`), the additions in sign as added."""
    w_down = lambda l: model.model.layers[l].mlp.down_proj.weight
    v = inp["vector"].double()
    vhat = v / v.norm()
    payload = {c: nc.patch_values({"patch": "subtract", "set": k}, inp["sets"], inp["comps"])[1] for c, k in nc.SET_KINDS.items()}
    written = {c: rs.write_vector(p, w_down) for c, p in payload.items()}
    w = written[nc.TOP].double()
    u = random_direction(inp["role_index"], len(v))
    added = {nc.SLICE: -alpha * v, nc.VECTOR: -v, nc.OFF: w - (w @ vhat) * vhat, nc.DIR_TOP: -float(w.norm()) * u,
             nc.DIR_SLICE: -float(alpha * v.norm()) * u}
    rows_c = [convs[(q, s)] for q, s in zip(replies.question_index, replies["sample"])]
    parts = []
    for c in nc.FORCED:
        if c in added:
            ctx, rows = nc.vector_patch(model, -added[c].float(), layer), None
        else:
            ctx, rows = contextlib.nullcontext(), None if c == nc.BASE else [payload[c]] * len(replies)
        with ctx:
            scored = nc.forced_scores(model, tok, rows_c, replies.response.tolist(), rows=rows, batch_size=batch_size)
        parts.append(replies[["reply_kind", "question_index", "sample", "seed"]].assign(
            condition=c, n_tokens=[x["n_tokens"] for x in scored], logp_sum=[x["logp"] for x in scored],
            logp_mean=[x["logp_mean"] for x in scored]))
    sizes = [{"condition": c, "norm": float(x.double().norm()), "along_v": float(x.double().cpu() @ vhat.cpu())}
             for c, x in (written | added).items()]
    return pd.concat(parts, ignore_index=True).assign(role=role), pd.DataFrame(sizes).assign(role=role)


def run_score(args, data: Path, first: Path, out: Path) -> None:
    use_upstream(fetch_upstream(config.upstream_dir()))
    pm = s8.load_model(args)
    for d in ("forced", "sizes_forced"):
        (out / d).mkdir(parents=True, exist_ok=True)
    convs_all = pd.read_parquet(first / "conversations.parquet")
    alone, assistant = pd.read_parquet(out / "alone.parquet"), pd.read_parquet(out / "assistant.parquet")
    alpha = alphas(out)
    for role in shard(args.roles or s6.steered_roles(data / "role_steering"), args.shard):
        path = out / "forced" / f"{role}.parquet"
        if path.exists():
            continue
        replies = pd.concat([alone[alone.role == role].assign(reply_kind="role"),
                             assistant[assistant.role == role].assign(reply_kind="assistant")], ignore_index=True)
        replies = replies.sort_values(["reply_kind", "question_index", "sample"]).reset_index(drop=True)
        forced, sizes = score_role(pm.model, pm.tokenizer, role, role_inputs(data, out, role), float(alpha[role]),
                                   s8.role_conversations(convs_all, role), replies, args.batch_size, args.target)
        s8.write_parquet(sizes, out / "sizes_forced" / f"{role}.parquet")
        s8.write_parquet(forced, path)
        print(f"scored {role}", flush=True)


def run_generate(args, data: Path, first: Path, out: Path) -> None:
    alpha = alphas(out)
    args.changes = True
    s8.generate(args, data, first, out, lambda r: nc.same_size_conditions(r, float(alpha[r])), lambda r: role_inputs(data, out, r))


def run_analyze(args, data: Path, first: Path, out: Path) -> None:
    to = data / args.to if args.to else out
    s6.refuse([to / f"{n}.parquet" for n in TABLES])
    forced, alone = s8.read_folder(out / "forced"), pd.read_parquet(out / "alone.parquet")
    to.mkdir(parents=True, exist_ok=True)
    forced.merge(alone[["role", "question_index", "sample", "score"]], on=["role", "question_index", "sample"], how="left").to_parquet(
        to / "per_reply.parquet", index=False)
    per = nc.role_means(nc.conversation_table(forced, alone))
    per.to_parquet(to / "per_role.parquet", index=False)
    size = nc.size_table(s8.read_folder(out / "sizes_forced"), per)
    size.to_parquet(to / "sizes.parquet", index=False)
    paired, shares = nc.same_size_tables(per, pd.read_parquet(out / "lower_sets.parquet"), alphas(out))
    paired.to_parquet(to / "paired.parquet", index=False)
    shares.to_parquet(to / "share_of_vector_drop.parquet", index=False)
    first_per = pd.read_parquet(first / "per_role.parquet")
    judged_per, judged_paired = nc.judged_tables(pd.read_parquet(out / "judged_replies.parquet"), first_per,
                                                 s8.read_folder(out / "sizes"), s8.read_folder(out / "changes"))
    judged_per.to_parquet(to / "judged_per_role.parquet", index=False)
    judged_paired.to_parquet(to / "judged_paired.parquet", index=False)
    measured, summary = nc.sizes_measured(size, first_per, judged_per)
    measured.to_parquet(to / "sizes_measured.parquet", index=False)
    summary.to_parquet(to / "sizes_summary.parquet", index=False)
    print(shares.round(3).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["prepare", "score", "generate", "judge", "build", "analyze"])
    s8.add_generate_arguments(ap)
    ap.add_argument("--batch-size", type=int, default=32, help="score: rows per teacher-forced pass")
    ap.add_argument("--to", help="analyze: another folder in the data directory, to run it again beside released outputs")
    args = ap.parse_args(argv)
    if args.part == "judge" and not args.action:
        ap.error("judge needs submit or collect")
    data = config.data_dir()
    run = {"prepare": run_prepare, "score": run_score, "generate": run_generate, "judge": s8.run_judge, "build": s8.run_build,
           "analyze": run_analyze}[args.part]
    return run(args, data, data / "necessity", data / "necessity_same_size") or 0


if __name__ == "__main__":
    sys.exit(main())
