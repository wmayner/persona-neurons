#!/usr/bin/env python3
"""Step 8: patching under role prompts (`persona_neurons.necessity`). Writes into necessity/ in the data directory, or with
--scaled into necessity_scaled/.

  prepare        (CPU) conversations.parquet, each role's 10: question q, sample s, system prompt (2q + s) mod 5
  generate       (GPU) every condition's 10 replies (replies/) and each condition's size at the target layer (sizes/); with
                 --changes, each patch's measured change to the residual on the role prompt alone's replies (changes/)
  judge          (API) `judge submit`, later `judge collect`: judge/, labels.parquet
  build          (CPU) judged_replies.parquet
  analyze        (CPU) per_role.parquet, paired.parquet and share_of_vector_drop.parquet, in --to if given
  prepare-scaled (CPU) alone.parquet in necessity_scaled/, the first run's replies to the role prompt alone
  usual-range    (CPU) how far the scaled conditions put each neuron from its usual range: necessity_scaled/usual_range/

Reads role_steering/ (step 6) and neuron_stats/default.pt (step 3). Parts refuse to overwrite their outputs, and generate
skips roles whose file exists. `--roles ROLE ...` limits generate, judge, build and analyze to those roles.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config, necessity as nc, role_steering as rs
from persona_neurons.shards import shard
from persona_neurons.upstream import fetch_upstream, model_snapshot, use_upstream

_spec = importlib.util.spec_from_file_location("step6", Path(__file__).with_name("6_role_steering.py"))
s6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s6)


def run_prepare(args, data: Path, first: Path, out: Path) -> None:
    s6.refuse([first / "conversations.parquet"])
    first.mkdir(parents=True, exist_ok=True)
    checkout = fetch_upstream(config.upstream_dir())
    rows = []
    for r in s6.steered_roles(data / "role_steering"):
        prompts = [x["pos"] for x in json.loads((checkout / "data/roles/instructions" / f"{r}.json").read_text())["instruction"]]
        if len(prompts) != 5:
            raise SystemExit(f"{r}: {len(prompts)} system prompts")
        rows += [{"role": r, "question_index": q, "sample": s, "prompt_index": nc.prompt_index(q, s),
                  "system": prompts[nc.prompt_index(q, s)], "question": rs.QUESTIONS[q]} for q, s in nc.ITEMS]
    pd.DataFrame(rows).to_parquet(first / "conversations.parquet", index=False)
    print(f"prepare: {len(rows) // len(nc.ITEMS)} roles", flush=True)


def run_prepare_scaled(args, data: Path, first: Path, out: Path) -> None:
    s6.refuse([out / "alone.parquet"])
    out.mkdir(parents=True, exist_ok=True)
    j = pd.read_parquet(first / "judged_replies.parquet")
    alone = j[j.patch == "prompt"][["role", "question_index", "sample", "response"]]
    alone.to_parquet(out / "alone.parquet", index=False)
    print(f"prepare-scaled: {len(alone)} replies to the role prompt alone", flush=True)


def role_inputs(data: Path, role: str) -> dict:
    inp = s6.role_inputs(data / "role_steering", role)
    vec = np.load(data / "role_steering" / "vectors.npz")
    inp["vector"] = torch.tensor(vec["v_without"][vec["roles"].tolist().index(role)])
    return inp


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").rename(path)


def load_model(args):
    from assistant_axis.internals import ProbingModel

    model_path = args.model_path or model_snapshot(config.MODEL_ID, config.MODEL_REVISION)
    return ProbingModel(str(model_path), dtype=getattr(torch, args.dtype), device={"": args.device} if args.device else None)


def role_conversations(convs_all: pd.DataFrame, role: str) -> dict:
    cv = convs_all[convs_all.role == role].set_index(["question_index", "sample"])
    return {(q, s): [{"role": "system", "content": cv.loc[(q, s), "system"]},
                     {"role": "user", "content": rs.QUESTIONS[q]}] for q, s in nc.ITEMS}


def generate(args, data: Path, first: Path, out: Path, make_conditions, inputs) -> None:
    """Replies, sizes and measured changes for this shard's roles. make_conditions(role) gives the conditions and inputs(role)
    the role's inputs. Changes are measured on out/alone.parquet if it exists, else on the replies to the role prompt alone."""
    use_upstream(fetch_upstream(config.upstream_dir()))
    pm = load_model(args)
    model, tok = pm.model, pm.tokenizer
    w_down = lambda l: model.model.layers[l].mlp.down_proj.weight
    convs_all = pd.read_parquet(first / "conversations.parquet")
    for d in ("replies", "sizes") + (("changes",) if args.changes else ()):
        (out / d).mkdir(parents=True, exist_ok=True)
    roles = shard(args.roles or s6.steered_roles(data / "role_steering"), args.shard)
    for role in roles:
        path = out / "replies" / f"{role}.parquet"
        if path.exists():
            continue
        inp, conds, convs = inputs(role), make_conditions(role), role_conversations(convs_all, role)
        parts, sizes = [], []
        for ctx, group in nc.condition_groups(conds, inp, model, args.chunk, args.target):
            rows_c, seeds, payloads, meta = [], [], [], []
            for cond, payload in group:
                for q, s in nc.ITEMS:
                    rows_c.append(convs[(q, s)])
                    seeds.append(rs.reply_seed(inp["role_index"], q, s))
                    payloads.append(payload)
                    meta.append((cond["condition_id"], cond["patch"], cond["set"], q, s))
                if cond["patch"] != "prompt":
                    sizes.append(nc.write_size(cond, payload, inp["vector"], inp["sets"], w_down))
            with ctx:
                replies = rs.generate_rows(model, tok, rows_c, seeds, max_new_tokens=args.max_new_tokens,
                                           rows=None if all(p is None for p in payloads) else payloads)
            parts.append(pd.DataFrame(meta, columns=["condition_id", "patch", "set", "question_index", "sample"]).assign(
                seed=seeds, response=replies))
        pd.DataFrame(sizes).assign(role=role).to_parquet(out / "sizes" / f"{role}.parquet", index=False)
        write_parquet(pd.concat(parts, ignore_index=True).assign(role=role), path)
        print(f"generate {role}: {sum(map(len, parts))} replies", flush=True)
    if not args.changes:
        return
    for role in roles:  # after every role's replies, so that this measure can never cost a run its replies
        to = out / "changes" / f"{role}.parquet"
        if to.exists():
            continue
        if (out / "alone.parquet").exists():
            alone = pd.read_parquet(out / "alone.parquet", filters=[("role", "==", role)])
        else:
            j = pd.read_parquet(out / "replies" / f"{role}.parquet")
            alone = j[j.patch == "prompt"]
        ch = nc.residual_changes(model, tok, make_conditions(role), inputs(role), role_conversations(convs_all, role),
                                 alone.sort_values(["question_index", "sample"]), args.chunk, args.target)
        write_parquet(ch.assign(role=role), to)
        print(f"changes {role}: {len(ch)} rows", flush=True)


def run_generate(args, data: Path, first: Path, out: Path) -> None:
    generate(args, data, first, out, nc.scaled_conditions if args.scaled else nc.conditions, lambda r: role_inputs(data, r))


def read_replies(data: Path, out: Path, roles: list[str] | None) -> pd.DataFrame:
    roles = roles or s6.steered_roles(data / "role_steering")
    missing = [r for r in roles if not (out / "replies" / f"{r}.parquet").exists()]
    if missing:
        raise SystemExit(f"no replies for {missing}; run generate first, or name the roles with --roles")
    return pd.concat([pd.read_parquet(out / "replies" / f"{r}.parquet") for r in roles], ignore_index=True)


def label(patch: str, kind: str) -> str:
    return f"{patch}-{kind}"


def run_judge(args, data: Path, first: Path, out: Path) -> int:
    """Each reply judged as its own role by the authors' role-playing judge prompt."""
    from persona_neurons import judge

    run_dir = out / "judge"
    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    replies = read_replies(data, out, args.roles)
    if args.action == "submit":
        requests = []
        for role, g in replies.groupby("role", sort=True):
            prompt = json.loads((checkout / "data" / "roles" / "instructions" / f"{role}.json").read_text())
            items = [{"label": label(x.patch, x.set), "prompt_index": int(x.sample), "question_index": int(x.question_index),
                      "question": rs.QUESTIONS[int(x.question_index)],
                      "conversation": [{"role": "user", "content": rs.QUESTIONS[int(x.question_index)]},
                                       {"role": "assistant", "content": x.response}]} for x in g.itertuples()]
            requests += judge.build_role_requests(role, items, prompt["eval_prompt"])
        print(f"{len(requests)} requests")
        if requests:
            print("submitted batches:", " ".join(judge.submit(requests, run_dir)))
        return 0
    from assistant_axis.judge import parse_judge_score

    s6.refuse([out / "labels.parquet"])
    batch_ids = json.loads((run_dir / "batches.json").read_text())
    try:
        texts = judge.collect(batch_ids, run_dir)
    except judge.BatchNotEnded as e:
        print(f"not yet: {e}")
        return 3
    cids = [judge.custom_id(r, f"{label(p, k)}_p{s}_q{q}") for r, p, k, s, q in
            zip(replies.role, replies.patch, replies.set, replies["sample"], replies.question_index)]
    scores = [parse_judge_score(texts[c]) if texts.get(c) is not None else None for c in cids]
    labels = replies[["role", "condition_id", "question_index", "sample"]].assign(custom_id=cids, score=scores)
    labels.to_parquet(out / "labels.parquet", index=False)
    print(f"labels: {labels.score.notna().sum()} of {len(labels)} replies scored")
    return 0


def run_build(args, data: Path, first: Path, out: Path) -> None:
    s6.refuse([out / "judged_replies.parquet"])
    replies = read_replies(data, out, args.roles)
    labels = pd.read_parquet(out / "labels.parquet")
    j = replies.merge(labels[["role", "condition_id", "question_index", "sample", "score"]],
                      on=["role", "condition_id", "question_index", "sample"], validate="1:1")
    j = j[["role", "patch", "set", "question_index", "sample", "seed", "response", "score"]]
    j.to_parquet(out / "judged_replies.parquet", index=False)
    print(j.groupby(["patch", "set"]).score.apply(lambda s: s.isin([2, 3]).mean()).round(3).to_string())


def read_folder(folder: Path) -> pd.DataFrame | None:
    files = sorted(folder.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def run_analyze(args, data: Path, first: Path, out: Path) -> None:
    to = data / args.to if args.to else out
    s6.refuse([to / n for n in ("per_role.parquet", "paired.parquet", "share_of_vector_drop.parquet")])
    j = pd.read_parquet(out / "judged_replies.parquet")
    per = nc.per_role_table(j[j.role.isin(args.roles)] if args.roles else j)
    if args.scaled:
        f = pd.read_parquet(first / "per_role.parquet")
        per = nc.add_drops(per, f[f.patch == "prompt"].set_index("role"))
    else:
        per = nc.add_drops(per, per[per.patch == "prompt"].set_index("role"))
    per = per.merge(read_folder(out / "sizes"), on=["role", "patch", "set"], how="left")
    changes = read_folder(out / "changes")
    if changes is not None:
        per = nc.merge_changes(per, changes)
    to.mkdir(parents=True, exist_ok=True)
    per.to_parquet(to / "per_role.parquet", index=False)
    if args.scaled:
        paired, ratios = nc.scaled_tables(per, pd.read_parquet(first / "per_role.parquet"))
    else:
        paired, ratios = nc.paired_table(per), nc.share_of_vector_drop(per)
    paired.to_parquet(to / "paired.parquet", index=False)
    ratios.to_parquet(to / "share_of_vector_drop.parquet", index=False)
    print(per.groupby(["patch", "set"])[["in_role", "fully", "drop_in_role"]].mean().round(3).to_string())
    print(ratios.round(3).to_string(index=False))


def run_usual_range(args, data: Path, first: Path, out: Path) -> None:
    to = out / "usual_range"
    s6.refuse([to])
    stats = torch.load(data / "neuron_stats" / "default.pt", map_location="cpu", weights_only=False)
    bound, sub = nc.reply_sds(stats)
    t = nc.usual_range_table(pd.read_parquet(data / "role_steering" / "sets.parquet"), read_folder(out / "sizes"), bound, sub)
    to.mkdir(parents=True)
    t.to_parquet(to / "per_neuron.parquet", index=False)
    long = pd.concat([t.melt(id_vars=["patch", "set"], value_vars=v, var_name="measure") for v in
                      (["in_sd_bound", "in_sd_subsample"], ["own_size_in_sd_bound", "own_size_in_sd_subsample"])])
    q = long.dropna().groupby(["patch", "set", "measure"]).value.describe(percentiles=[0.1, 0.5, 0.9]).reset_index()
    q.to_parquet(to / "summary.parquet", index=False)
    print(q.round(2).to_string(index=False))


def add_generate_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("action", nargs="?", choices=["submit", "collect"], help="judge: submit or collect")
    ap.add_argument("--shard", help="GPU parts: i/n")
    ap.add_argument("--roles", nargs="+", help="only these roles")
    ap.add_argument("--changes", action="store_true", help="generate: also each patch's measured change to the residual")
    ap.add_argument("--target", type=int, default=config.TARGET_LAYER)
    ap.add_argument("--chunk", type=int, default=12, help="generate: conditions per batch (10 replies each)")
    ap.add_argument("--max-new-tokens", type=int, default=rs.MAX_NEW_TOKENS)
    ap.add_argument("--model-path", type=Path, help="a local model folder, in place of Qwen3-8B at the fixed revision")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["prepare", "generate", "judge", "build", "analyze", "prepare-scaled", "usual-range"])
    add_generate_arguments(ap)
    ap.add_argument("--scaled", action="store_true", help="generate, judge, build and analyze: the scaled run")
    ap.add_argument("--to", help="analyze: another folder in the data directory, to run it again beside released outputs")
    args = ap.parse_args(argv)
    if args.part == "judge" and not args.action:
        ap.error("judge needs submit or collect")
    data = config.data_dir()
    first = data / "necessity"
    out = data / "necessity_scaled" if args.scaled or args.part in ("prepare-scaled", "usual-range") else first
    run = {"prepare": run_prepare, "generate": run_generate, "judge": run_judge, "build": run_build, "analyze": run_analyze,
           "prepare-scaled": run_prepare_scaled, "usual-range": run_usual_range}[args.part]
    return run(args, data, first, out) or 0


if __name__ == "__main__":
    sys.exit(main())
