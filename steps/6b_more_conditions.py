#!/usr/bin/env python3
"""Step 6b: the conditions beyond step 6's (`persona_neurons.role_steering`, more_conditions): the top k neurons, a
random set, the top k randomly rotated directions and the top k principal components at every other set size, draws 1
to 4 of the rotated directions and the random neurons, the two 50-neuron halves of the top 100 by role-specificity, and no steering. Run it
after step 6, whose roles, vectors and top 100 neurons it reads. Parts, working in role_steering_more/ in the data
directory:

  prepare   (CPU) the neuron sets: sets.parquet; roles.parquet; and each role's top 1,000 principal components, chosen
            from step 5's contributions: component_rank.parquet
  components (CPU) for some layers, each layer's principal-component basis as step 5 computes it, and the components each
            role chose there: components-parts/layer_LL.npz
  rotate    (CPU) for one draw and some layers (--draw 1 --layers 0-4), the layer's random rotation and each role's top
            directions in it: for draws 1 to 4, as many as the role's top 100 neurons have in that layer; for draw 0, as
            many as its top 1,000 have, of which a smaller set takes each layer's first: rotation-parts/dD/layer_LL.npz
  finish    (CPU) for each role, the top 1,000 components (components/{role}.npz) and the rotated directions of draws 0
            to 4 (directions/{role}.npz)
  generate  (GPU) 10 replies per role and condition: replies/{role}.parquet; --shard i/n takes every n-th role
  judge     (API) `judge submit`, later `judge collect`: the role-playing judge on every reply: judge/, labels.parquet
  build     (CPU) what the figures read: role_steering_sizes/judged_replies.parquet,
            role_steering_control_sizes/judged_replies.parquet, role_steering_draws/judged_replies.parquet and
            shares.parquet, and role_steering/unsteered_sommelier.parquet

Each part refuses to overwrite its outputs; generate skips roles whose file exists. Memory: about 24 GB and several cores
for components, about 12 GB and 8 cores for rotate, and 12 GB for the other CPU parts. The components and directions
take about 40 GB of disk.

    uv run steps/6b_more_conditions.py prepare
    uv run steps/6b_more_conditions.py components --layers 0-4          (then 5-9, 10-14, 15-18)
    uv run steps/6b_more_conditions.py rotate --draw 0 --layers 0-4     (then 5-9, 10-14, 15-18, for draws 0 to 4)
    uv run steps/6b_more_conditions.py finish
    uv run steps/6b_more_conditions.py generate [--shard i/n]
    uv run steps/6b_more_conditions.py judge submit                     (later: judge collect)
    uv run steps/6b_more_conditions.py build

`--roles ROLE ...` limits generate, judge and build to those roles, and `judge submit --limit N` sends at most N
requests (for a trial run).
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

from persona_neurons import config, role_steering as rs
from persona_neurons.shards import shard
from persona_neurons.upstream import fetch_upstream, model_snapshot, use_upstream

_spec = importlib.util.spec_from_file_location("step6", Path(__file__).with_name("6_role_steering.py"))
s6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s6)

TOP_RANKED = 1000   # the most components and rotated directions a set takes
ROTATION_DRAWS = (0,) + rs.MORE_DRAWS
CONTROLS = ("rotation_matched", "components")
REPLY_COLUMNS = ["role", "condition_id", "kind", "k", "draw", "strength", "question_index", "sample", "seed", "response",
                 "score"]


def run_prepare(args, data: Path, out: Path) -> None:
    s6.refuse([out / "sets.parquet", out / "roles.parquet", out / "component_rank.parquet"])
    out.mkdir(parents=True, exist_ok=True)
    base = data / "role_steering"
    roles = pd.read_parquet(base / "roles.parquet")
    arr = s6.load_split(data)
    index = {r: i for i, r in enumerate(arr["roles"].tolist())}
    parts = [rs.more_neuron_sets(r, int(ri), arr["delta_a"][index[r]], arr["p"][index[r]])
             for r, ri in zip(roles.role, roles.role_index)]
    top = [rs.top_neurons(arr["delta_a"][index[r]], arr["p"][index[r]]).reset_index(names="rank").assign(role=r)
           for r in roles.role]
    halves = rs.specificity_halves(pd.concat(top, ignore_index=True), arr["roles"].tolist(), arr["delta_a"], arr["p"], arr["norm"])
    parts.append(halves[["role", "kind", "draw", "layer", "neuron", "delta", "contribution", "k"]])
    pd.concat(parts, ignore_index=True).to_parquet(out / "sets.parquet", index=False)
    roles.to_parquet(out / "roles.parquet", index=False)
    pca = []
    for l in range(args.target + 1):
        with np.load(data / "split" / "pca" / f"layer_{l:02d}.npz") as z:
            assert z["roles"].tolist() == arr["roles"].tolist(), f"layer {l}'s components have another role list"
            pca.append(z["contributions"])
    ranks = [rs.top_components(np.stack([c[index[r]] for c in pca]), TOP_RANKED).reset_index(names="rank").assign(role=r)
             for r in roles.role]
    pd.concat(ranks, ignore_index=True).to_parquet(out / "component_rank.parquet", index=False)
    print(f"prepare: {len(roles)} roles", flush=True)


def run_rotate(args, data: Path, out: Path) -> None:
    layers = s6.parse_layers(args.layers)
    files = [out / "rotation-parts" / f"d{args.draw}" / f"layer_{l:02d}.npz" for l in layers]
    s6.refuse(files)
    files[0].parent.mkdir(parents=True, exist_ok=True)
    arr = s6.load_split(data)
    index = {r: i for i, r in enumerate(arr["roles"].tolist())}
    if args.draw == 0:
        top = pd.read_parquet(out / "sets.parquet", filters=[("kind", "==", "neurons"), ("k", "==", TOP_RANKED)])
        counts = {r: g.layer.value_counts().sort_index() for r, g in top.groupby("role")}
    else:
        counts = s6.neuron_layer_counts(data / "role_steering")
    for l, f in zip(layers, files):
        q = rs.haar_rotation(arr["delta_a"].shape[2], [rs.ROTATION_SEED, args.draw, l])
        q64 = q.astype(np.float64)
        saved = {}
        for r, n in counts.items():
            if l in n.index:
                top = rs.top_rotated(q64, arr["delta_a"][index[r], l], arr["p"][index[r], l], int(n[l]))
                saved |= {f"{r}|{c}": top[c].to_numpy() for c in ("direction", "coef", "contribution")}
                saved[f"{r}|cols"] = q[:, top.direction.to_numpy()]
        np.savez(f, layer=l, **saved)
        print(f"rotate: draw {args.draw}, layer {l}, {len(saved) // 4} roles", flush=True)


def run_finish(args, data: Path, out: Path) -> None:
    s6.refuse([out / "directions", out / "components"])
    names = [f"layer_{l:02d}.npz" for l in range(args.target + 1)]
    missing = [f"d{d}/{n}" for d in ROTATION_DRAWS for n in names if not (out / "rotation-parts" / f"d{d}" / n).exists()]
    missing += [f"components {n}" for n in names if not (out / "components-parts" / n).exists()]
    if missing:
        raise SystemExit(f"not computed yet: {missing}; run components for every layer and rotate for every draw and layer")
    # arrays are read per role: the parts hold every role's columns, tens of GB in all
    parts = {d: [np.load(out / "rotation-parts" / f"d{d}" / n) for n in names] for d in ROTATION_DRAWS}
    comp_parts = [np.load(out / "components-parts" / n) for n in names]
    (out / "directions").mkdir()
    (out / "components").mkdir()
    n_neurons = s6.load_split(data)["delta_a"].shape[2]
    ranks = pd.read_parquet(out / "component_rank.parquet")
    for r in s6.steered_roles(data / "role_steering"):
        rank = ranks[ranks.role == r].sort_values("rank")
        cols, coef = np.zeros((n_neurons, len(rank)), np.float32), np.zeros(len(rank))
        for z in comp_parts:
            if f"{r}|rank" in z:
                cols[:, z[f"{r}|rank"]], coef[z[f"{r}|rank"]] = z[f"{r}|cols"], z[f"{r}|coef"]
        np.savez(out / "components" / f"{r}.npz", cols=cols, coef=coef,
                 **{c: rank[c].to_numpy() for c in ("layer", "component", "contribution")})
        dirs = {}
        for d in ROTATION_DRAWS:
            here = [z for z in parts[d] if f"{r}|cols" in z]   # in layer order
            key = f"rotation_matched|d{d}"
            dirs[f"{key}|layer"] = np.concatenate([np.full(len(z[f"{r}|direction"]), int(z["layer"])) for z in here])
            dirs |= {f"{key}|{c}": np.concatenate([z[f"{r}|{c}"] for z in here])
                     for c in ("direction", "coef", "contribution")}
            dirs[f"{key}|cols"] = np.concatenate([z[f"{r}|cols"] for z in here], axis=1)
            assert len(dirs[f"{key}|layer"]) == (len(rank) if d == 0 else rs.K), r
        np.savez(out / "directions" / f"{r}.npz", **dirs)
    print("finish: done", flush=True)


def role_inputs(data: Path, out: Path, role: str) -> dict:
    vec = np.load(data / "role_steering" / "vectors.npz")
    i = vec["roles"].tolist().index(role)
    with np.load(out / "components" / f"{role}.npz") as z:
        comps = {("components", 0): {"rank": pd.DataFrame({c: z[c] for c in ("layer", "component", "coef", "contribution")}),
                                     "v": z["cols"]}}
    with np.load(out / "directions" / f"{role}.npz") as z:
        for key in sorted({k.rsplit("|", 1)[0] for k in z.files}):
            kind, d = key.split("|")
            comps[(kind, int(d[1:]))] = {"rank": pd.DataFrame({c: z[f"{key}|{c}"] for c in ("layer", "direction", "coef",
                                                                                             "contribution")}),
                                         "v": z[f"{key}|cols"]}
    return {"target_norm": rs.STRENGTH * float(np.linalg.norm(vec["v_without"][i])), "role_index": int(vec["role_index"][i]),
            "comps": comps, "sets": pd.read_parquet(out / "sets.parquet", filters=[("role", "==", role)])}


def run_generate(args, data: Path, out: Path) -> None:
    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    from assistant_axis.internals import ProbingModel

    model_path = args.model_path or model_snapshot(config.MODEL_ID, config.MODEL_REVISION)
    pm = ProbingModel(str(model_path), dtype=getattr(torch, args.dtype), device={"": args.device} if args.device else None)
    model, tok = pm.model, pm.tokenizer
    (out / "replies").mkdir(parents=True, exist_ok=True)
    for role in shard(args.roles or s6.steered_roles(data / "role_steering"), args.shard):
        path = out / "replies" / f"{role}.parquet"
        if path.exists():
            continue
        inp = role_inputs(data, out, role)
        parts = []
        for ctx, group in rs.steering_groups(rs.more_conditions(role), inp["sets"], inp["comps"], model, None,
                                             inp["target_norm"], args.chunk, layer=args.target):
            convs, seeds, rows, meta = [], [], [], []
            for cond, payload in group:
                for q in range(len(rs.QUESTIONS)):
                    for s in range(rs.SAMPLES):
                        convs.append([{"role": "user", "content": rs.QUESTIONS[q]}])
                        seeds.append(rs.reply_seed(inp["role_index"], q, s))
                        rows.append(payload)
                        meta.append((cond["condition_id"], cond["kind"], cond["k"], cond["draw"], q, s))
            with ctx:
                replies = rs.generate_rows(model, tok, convs, seeds, max_new_tokens=args.max_new_tokens,
                                           rows=None if all(r is None for r in rows) else rows)
            parts.append(pd.DataFrame(meta, columns=["condition_id", "kind", "k", "draw", "question_index", "sample"])
                         .assign(seed=seeds, response=replies))
        df = pd.concat(parts, ignore_index=True).assign(role=role)
        df.to_parquet(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").rename(path)
        print(f"generate {role}: {len(df)} replies", flush=True)


def label(kind: str, k: int, draw: int) -> str:
    return f"{kind}{k}d{draw}"


def judge_inputs(g: pd.DataFrame) -> list[dict]:
    return [{"label": label(x.kind, x.k, x.draw), "prompt_index": int(x.sample), "question_index": int(x.question_index),
             "question": rs.QUESTIONS[int(x.question_index)],
             "conversation": [{"role": "user", "content": rs.QUESTIONS[int(x.question_index)]},
                              {"role": "assistant", "content": x.response}]} for x in g.itertuples()]


def run_judge(args, data: Path, out: Path) -> int:
    """Each reply judged as its own role by the authors' role-playing judge prompt."""
    from persona_neurons import judge

    run_dir = out / "judge"
    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    replies = s6.read_replies(out, args.roles)
    if args.action == "submit":
        requests = []
        for role, g in replies.groupby("role", sort=True):
            prompt = json.loads((checkout / "data" / "roles" / "instructions" / f"{role}.json").read_text())
            requests += judge.build_role_requests(role, judge_inputs(g), prompt["eval_prompt"])
        requests = requests[:args.limit] if args.limit else requests
        print(f"{len(requests)} requests")
        if requests and not args.dry_run:
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
    cids = [judge.custom_id(r, f"{label(k, n, d)}_p{s}_q{q}") for r, k, n, d, s, q in
            zip(replies.role, replies.kind, replies.k, replies.draw, replies["sample"], replies.question_index)]
    scores = [parse_judge_score(texts[c]) if texts.get(c) is not None else None for c in cids]
    labels = replies[["role", "condition_id", "question_index", "sample"]].assign(custom_id=cids, score=scores)
    labels.to_parquet(out / "labels.parquet", index=False)
    print(f"labels: {labels.score.notna().sum()} of {len(labels)} replies scored")
    return 0


def run_build(args, data: Path, out: Path) -> None:
    sizes, controls = data / "role_steering_sizes", data / "role_steering_control_sizes"
    draws = data / "role_steering_draws"
    unsteered = data / "role_steering" / "unsteered_sommelier.parquet"
    s6.refuse([sizes / "judged_replies.parquet", controls / "judged_replies.parquet", draws / "judged_replies.parquet",
               draws / "shares.parquet", unsteered])
    replies = s6.read_replies(out, args.roles)
    labels = pd.read_parquet(out / "labels.parquet")
    j = replies.merge(labels[["role", "condition_id", "question_index", "sample", "score"]],
                      on=["role", "condition_id", "question_index", "sample"], validate="1:1").assign(strength=rs.STRENGTH)
    j = j[REPLY_COLUMNS]
    is_draw = j.draw > 0
    is_control = j.kind.isin(CONTROLS) & ~is_draw
    for folder in (sizes, controls, draws):
        folder.mkdir(exist_ok=True)
    j[~is_draw & ~is_control & (j.kind != "none")].to_parquet(sizes / "judged_replies.parquet", index=False)
    j[is_control].to_parquet(controls / "judged_replies.parquet", index=False)
    j[is_draw].to_parquet(draws / "judged_replies.parquet", index=False)
    none = j[(j.kind == "none") & (j.role == "sommelier")]
    none.drop(columns=["k", "strength"]).to_parquet(unsteered, index=False)
    # each draw's contribution to the role's vector; draw 0 is step 6's
    norm = dict(zip(*(lambda a: (a["roles"].tolist(), a["norm"]))(s6.load_split(data))))
    sets = pd.read_parquet(out / "sets.parquet")
    picked = sets[(sets.kind == "random") & (sets.k == rs.K)]
    shares = [{"role": s.role, "kind": "random", "draw": s.draw, "share": 100 * s.contribution / norm[s.role]}
              for s in picked.groupby(["role", "draw"], as_index=False).contribution.sum().itertuples()]
    for r in sorted(set(j.role)):
        with np.load(out / "directions" / f"{r}.npz") as z:
            shares += [{"role": r, "kind": "rotation_matched", "draw": d,
                        "share": 100 * float(z[f"rotation_matched|d{d}|contribution"].sum()) / norm[r]}
                       for d in rs.MORE_DRAWS]
    first = pd.read_parquet(data / "role_steering" / "shares.parquet")
    first = first[first.kind.isin(["rotation_matched", "random"]) & first.role.isin(j.role)]
    pd.concat([first, pd.DataFrame(shares)], ignore_index=True).to_parquet(draws / "shares.parquet", index=False)
    print(s6.in_role_summary(j).round(3).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["prepare", "components", "rotate", "finish", "generate", "judge", "build"])
    ap.add_argument("action", nargs="?", choices=["submit", "collect"], help="judge: submit or collect")
    ap.add_argument("--draw", type=int, choices=ROTATION_DRAWS, help="rotate: the draw")
    ap.add_argument("--layers", help="components and rotate: the layers, e.g. 0-4 or 5,6,7")
    ap.add_argument("--shard", help="generate: i/n")
    ap.add_argument("--roles", nargs="+", help="generate, judge and build: only these roles (for a trial run)")
    ap.add_argument("--limit", type=int, help="judge submit: send at most this many requests (for a trial run)")
    ap.add_argument("--target", type=int, default=config.TARGET_LAYER)
    ap.add_argument("--chunk", type=int, default=7, help="generate: conditions per batch (10 replies each)")
    ap.add_argument("--max-new-tokens", type=int, default=rs.MAX_NEW_TOKENS)
    ap.add_argument("--model-path", type=Path, help="a local model folder, in place of Qwen3-8B at the fixed revision")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="judge submit: count the requests without sending them")
    args = ap.parse_args(argv)
    if args.part == "rotate" and (args.draw is None or not args.layers):
        ap.error("rotate needs --draw and --layers")
    if args.part == "components" and not args.layers:
        ap.error("components needs --layers")
    if args.part == "judge" and not args.action:
        ap.error("judge needs submit or collect")
    data = config.data_dir()
    out = data / "role_steering_more"
    run = {"prepare": run_prepare, "components": s6.run_components, "rotate": run_rotate, "finish": run_finish, "generate": run_generate,
           "judge": run_judge, "build": run_build}[args.part]
    return run(args, data, out) or 0


if __name__ == "__main__":
    sys.exit(main())
