#!/usr/bin/env python3
"""Step 6: steering Qwen3-8B towards every role with the persona vector, the top 100 neurons, the top 100 principal
components, the top 100 randomly rotated directions and 100 random neurons, each at 4× the vector's length, and
judging the replies (`persona_neurons.role_steering`). Parts, writing into role_steering/ in the data directory:

  prepare     (CPU) the roles: every role of the split with both vectors and an eval prompt in the authors'
              repository, with its index among all roles sorted (roles.parquet; other roles in skipped.parquet,
              with the reason); the vectors (vectors.npz); the top 100 neurons and the random neurons
              (sets.parquet); and the top 100 principal components, chosen from step 5's contributions
              (component_rank.parquet)
  components  (CPU) for some layers (--layers 0-4, or 5,6,7), each layer's principal-component basis as step 5
              computes it, and the components each role chose there: components-parts/layer_LL.npz
  rotate      (CPU) for some layers, each layer's random rotation and each role's top directions in it, as many as
              the top 100 neurons have in that layer: rotation-parts/layer_LL.npz
  finish      (CPU) for each role, the top 100 components (components/{role}.npz) and rotated directions
              (directions/{role}.npz), and each set's contribution to its role's vector (shares.parquet)
  generate    (GPU) 10 replies per role and condition, 2 to each of the paper's 5 questions, no system prompt,
              up to 512 tokens: replies/{role}.parquet; --shard i/n takes every n-th role
  judge       (API) `judge submit`, later `judge collect`: the role-playing judge (the authors' prompt, Claude
              Haiku 5.5) on every reply: judge/, labels.parquet
  build       (CPU) judged_replies.parquet: one row per reply with its condition and score, which the figures read

Inputs: vectors/ (step 4), split/arrays.npz, split/pca/ and neuron_stats/ (steps 3 and 5). Every part refuses to
overwrite its outputs; generate skips roles whose file exists. Memory: about 8 GB for prepare and finish; for
components about 24 GB and several cores, 20 to 50 minutes per 5 layers; for rotate about 8 GB and 8 cores.

    uv run steps/6_role_steering.py prepare
    uv run steps/6_role_steering.py components --layers 0-4      (then 5-9, 10-14, 15-18)
    uv run steps/6_role_steering.py rotate --layers 0-4          (then 5-9, 10-14, 15-18)
    uv run steps/6_role_steering.py finish
    uv run steps/6_role_steering.py generate [--shard i/n]
    uv run steps/6_role_steering.py judge submit       (later: judge collect)
    uv run steps/6_role_steering.py build

`--roles ROLE ...` limits generate, judge and build to those roles, and `judge submit --limit N` sends at most N
requests (for a trial run).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config, role_steering as rs, split as sp
from persona_neurons.shards import shard
from persona_neurons.upstream import fetch_upstream, model_snapshot, use_upstream


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


def load_vector(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["vector"].double()


def load_split(data: Path) -> dict:
    """The split's arrays (roles, delta_a, p, norm), read once (an npz reads an array again at every access)."""
    with np.load(data / "split" / "arrays.npz") as z:
        return {k: z[k] for k in ("roles", "delta_a", "p", "norm")}


def steered_roles(out: Path) -> list[str]:
    return pd.read_parquet(out / "roles.parquet").role.tolist()


def run_prepare(args, data: Path, out: Path) -> None:
    refuse([out / n for n in ("roles.parquet", "skipped.parquet", "vectors.npz", "sets.parquet",
                              "component_rank.parquet")])
    out.mkdir(parents=True, exist_ok=True)
    arr = load_split(data)
    split_roles = arr["roles"].tolist()
    indices = rs.role_indices(split_roles)
    checkout = fetch_upstream(config.upstream_dir())
    skipped = []
    for r in sorted(split_roles):
        f = checkout / "data" / "roles" / "instructions" / f"{r}.json"
        if not all((data / "vectors" / v / f"{r}.pt").exists() for v in ("all", "without")):
            skipped.append({"role": r, "reason": "no vector"})
        elif not f.exists() or not json.loads(f.read_text()).get("eval_prompt"):
            skipped.append({"role": r, "reason": "no eval prompt"})
    gone = {s["role"] for s in skipped}
    roles = [r for r in sorted(split_roles) if r not in gone]
    rows = [split_roles.index(r) for r in roles]
    pd.DataFrame({"role": roles, "role_index": [indices[r] for r in roles]}).to_parquet(out / "roles.parquet",
                                                                                       index=False)
    pd.DataFrame(skipped, columns=["role", "reason"]).to_parquet(out / "skipped.parquet", index=False)
    sets = [rs.neuron_sets(r, indices[r], arr["delta_a"][i], arr["p"][i]) for r, i in zip(roles, rows)]
    pd.concat(sets, ignore_index=True).to_parquet(out / "sets.parquet", index=False)
    vec = {}
    for version in ("all", "without"):
        default = load_vector(data / "vectors" / version / "default.pt")[args.target]
        vec[f"v_{version}"] = np.stack([(load_vector(data / "vectors" / version / f"{r}.pt")[args.target] - default)
                                        .numpy() for r in roles])
    np.savez(out / "vectors.npz", roles=np.array(roles), role_index=np.array([indices[r] for r in roles]), **vec)
    # the top 100 components over layers 0 to the target, by step 5's contributions
    pca = []
    for l in range(args.target + 1):
        with np.load(data / "split" / "pca" / f"layer_{l:02d}.npz") as z:
            assert z["roles"].tolist() == split_roles, f"layer {l}'s components have another role list"
            pca.append(z["contributions"])
    ranks = [rs.top_components(np.stack([c[i] for c in pca])).reset_index(names="rank").assign(role=r)
             for r, i in zip(roles, rows)]
    pd.concat(ranks, ignore_index=True).to_parquet(out / "component_rank.parquet", index=False)
    print(f"prepare: {len(roles)} roles, {len(skipped)} skipped {skipped}")


def run_components(args, data: Path, out: Path) -> None:
    layers = parse_layers(args.layers)
    files = [out / "components-parts" / f"layer_{l:02d}.npz" for l in layers]
    refuse(files)
    files[0].parent.mkdir(parents=True, exist_ok=True)
    arr = load_split(data)
    index = {r: i for i, r in enumerate(arr["roles"].tolist())}
    ranks = pd.read_parquet(out / "component_rank.parquet")
    roles_all = sp.roles_with_vectors(data / "vectors" / "all")
    for l, f in zip(layers, files):
        _, v = sp.layer_basis(data / "neuron_stats", roles_all, l, args.target)
        saved, gap = {}, 0.0
        for r, g in ranks[ranks.layer == l].groupby("role"):
            cols = v[:, g.component.to_numpy()]
            coef = cols.T @ arr["delta_a"][index[r], l].astype(np.float64)
            c = coef * (cols.T @ arr["p"][index[r], l].astype(np.float64))
            # the recomputed basis reproduces step 5's contributions (saved there in float32)
            gap = max(gap, float(np.max(np.abs(c - g.contribution.to_numpy())) / np.max(np.abs(c))))
            saved |= {f"{r}|rank": g["rank"].to_numpy(), f"{r}|coef": coef, f"{r}|cols": cols.astype(np.float32)}
        np.savez(f, layer=l, **saved)
        print(f"components: layer {l}, {len(saved) // 3} roles; largest gap to step 5's contributions {gap:.1e} of "
              f"the largest", flush=True)


def neuron_layer_counts(out: Path) -> dict[str, pd.Series]:
    """Each role's number of top 100 neurons in each layer."""
    top = pd.read_parquet(out / "sets.parquet", filters=[("kind", "==", "neurons")])
    return {r: g.layer.value_counts().sort_index() for r, g in top.groupby("role")}


def run_rotate(args, data: Path, out: Path) -> None:
    layers = parse_layers(args.layers)
    files = [out / "rotation-parts" / f"layer_{l:02d}.npz" for l in layers]
    refuse(files)
    files[0].parent.mkdir(parents=True, exist_ok=True)
    arr = load_split(data)
    index = {r: i for i, r in enumerate(arr["roles"].tolist())}
    counts = neuron_layer_counts(out)
    for l, f in zip(layers, files):
        q = rs.haar_rotation(arr["delta_a"].shape[2], [rs.ROTATION_SEED, rs.DRAW, l])
        q64 = q.astype(np.float64)   # the float64 cast top_rotated would make, done once for all roles
        saved = {}
        for r, n in counts.items():
            if l in n.index:
                top = rs.top_rotated(q64, arr["delta_a"][index[r], l], arr["p"][index[r], l], int(n[l]))
                saved |= {f"{r}|{c}": top[c].to_numpy() for c in ("direction", "coef", "contribution")}
                saved[f"{r}|cols"] = q[:, top.direction.to_numpy()]
        np.savez(f, layer=l, **saved)
        print(f"rotate: layer {l}, {len(saved) // 4} roles", flush=True)


def run_finish(args, data: Path, out: Path) -> None:
    refuse([out / "components", out / "directions", out / "shares.parquet"])
    names = [f"layer_{l:02d}.npz" for l in range(args.target + 1)]
    missing = [f"{d}/{n}" for d in ("components-parts", "rotation-parts") for n in names if not (out / d / n).exists()]
    if missing:
        raise SystemExit(f"not computed yet: {missing}; run components and rotate for every layer")
    comp_parts, rot_parts = [], []
    for n in names:
        with np.load(out / "components-parts" / n) as z:
            comp_parts.append(dict(z))
        with np.load(out / "rotation-parts" / n) as z:
            rot_parts.append(dict(z))
    (out / "components").mkdir()
    (out / "directions").mkdir()
    arr = load_split(data)
    norm = dict(zip(arr["roles"].tolist(), arr["norm"]))
    n_neurons = arr["delta_a"].shape[2]
    ranks = pd.read_parquet(out / "component_rank.parquet")
    share = lambda r, contribution: 100 * float(np.sum(contribution)) / norm[r]
    shares = []
    for r in steered_roles(out):
        rank = ranks[ranks.role == r].sort_values("rank")
        cols, coef = np.zeros((n_neurons, rs.K), np.float32), np.zeros(rs.K)
        for z in comp_parts:
            if f"{r}|rank" in z:
                cols[:, z[f"{r}|rank"]], coef[z[f"{r}|rank"]] = z[f"{r}|cols"], z[f"{r}|coef"]
        np.savez(out / "components" / f"{r}.npz", cols=cols, coef=coef,
                 **{c: rank[c].to_numpy() for c in ("layer", "component", "contribution")})
        shares.append({"role": r, "kind": "components", "draw": 0, "share": share(r, rank.contribution)})
        here = [z for z in rot_parts if f"{r}|cols" in z]   # in layer order
        key = f"rotation_matched|d{rs.DRAW}"
        dirs = {f"{key}|layer": np.concatenate([np.full(len(z[f"{r}|direction"]), int(z["layer"])) for z in here])}
        dirs |= {f"{key}|{c}": np.concatenate([z[f"{r}|{c}"] for z in here])
                 for c in ("direction", "coef", "contribution")}
        dirs[f"{key}|cols"] = np.concatenate([z[f"{r}|cols"] for z in here], axis=1)
        assert len(dirs[f"{key}|layer"]) == rs.K, r
        np.savez(out / "directions" / f"{r}.npz", **dirs)
        shares.append({"role": r, "kind": "rotation_matched", "draw": rs.DRAW,
                       "share": share(r, dirs[f"{key}|contribution"])})
    sets = pd.read_parquet(out / "sets.parquet")
    by_set = sets.groupby(["role", "kind", "draw"], as_index=False).contribution.sum()
    shares += [{"role": s.role, "kind": s.kind, "draw": s.draw, "share": share(s.role, s.contribution)}
               for s in by_set.itertuples()]
    pd.DataFrame(shares).to_parquet(out / "shares.parquet", index=False)
    print(pd.DataFrame(shares).groupby("kind").share.median().round(2).to_string())


def role_inputs(out: Path, role: str) -> dict:
    vec = np.load(out / "vectors.npz")
    i = vec["roles"].tolist().index(role)
    with np.load(out / "components" / f"{role}.npz") as z:
        comps = {("components", 0): {"rank": pd.DataFrame({c: z[c] for c in ("layer", "component", "coef",
                                                                              "contribution")}), "v": z["cols"]}}
    with np.load(out / "directions" / f"{role}.npz") as z:   # keys are "kind|dD|field"
        for key in sorted({k.rsplit("|", 1)[0] for k in z.files}):
            kind, d = key.split("|")
            comps[(kind, int(d[1:]))] = {"rank": pd.DataFrame({c: z[f"{key}|{c}"] for c in ("layer", "direction",
                                                                                            "coef", "contribution")}),
                                         "v": z[f"{key}|cols"]}
    return {"vector": torch.tensor(vec["v_all"][i]), "target_norm": rs.STRENGTH * float(np.linalg.norm(vec["v_without"][i])),
            "role_index": int(vec["role_index"][i]), "comps": comps,
            "sets": pd.read_parquet(out / "sets.parquet", filters=[("role", "==", role)])}


def run_generate(args, data: Path, out: Path) -> None:
    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    from assistant_axis.internals import ProbingModel

    model_path = args.model_path or model_snapshot(config.MODEL_ID, config.MODEL_REVISION)
    pm = ProbingModel(str(model_path), dtype=getattr(torch, args.dtype), device={"": args.device} if args.device else None)
    model, tok = pm.model, pm.tokenizer
    (out / "replies").mkdir(parents=True, exist_ok=True)
    for role in shard(args.roles or steered_roles(out), args.shard):
        path = out / "replies" / f"{role}.parquet"
        if path.exists():
            continue
        inp = role_inputs(out, role)
        parts = []
        for ctx, group in rs.steering_groups(rs.conditions(role), inp["sets"], inp["comps"], model, inp["vector"],
                                             inp["target_norm"], args.chunk, layer=args.target):
            convs, seeds, rows, meta = [], [], [], []
            for cond, payload in group:
                for q in range(len(rs.QUESTIONS)):
                    for s in range(rs.SAMPLES):
                        convs.append([{"role": "user", "content": rs.QUESTIONS[q]}])
                        seeds.append(rs.reply_seed(inp["role_index"], q, s))
                        rows.append(payload)
                        meta.append((cond["condition_id"], cond["kind"], cond["draw"], q, s))
            with ctx:
                replies = rs.generate_rows(model, tok, convs, seeds, max_new_tokens=args.max_new_tokens,
                                           rows=None if all(r is None for r in rows) else rows)
            parts.append(pd.DataFrame(meta, columns=["condition_id", "kind", "draw", "question_index", "sample"])
                         .assign(seed=seeds, response=replies))
        df = pd.concat(parts, ignore_index=True).assign(role=role)
        df.to_parquet(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").rename(path)
        print(f"generate {role}: {len(df)} replies", flush=True)


def read_replies(out: Path, roles: list[str] | None) -> pd.DataFrame:
    """The replies of `roles` (every steered role without), refusing any that are missing."""
    roles = roles or steered_roles(out)
    missing = [r for r in roles if not (out / "replies" / f"{r}.parquet").exists()]
    if missing:
        raise SystemExit(f"no replies for {missing}; run generate first, or name the roles with --roles")
    return pd.concat([pd.read_parquet(out / "replies" / f"{r}.parquet") for r in roles], ignore_index=True)


def reply_key(kind: str, draw: int, sample: int, question: int) -> str:
    return f"{kind}{draw}_p{sample}_q{question}"


def judge_inputs(g: pd.DataFrame) -> list[dict]:
    """The replies in g in the form the judge's request builder takes, labelled by kind and draw."""
    return [{"label": f"{x.kind}{x.draw}", "prompt_index": int(x.sample), "question_index": int(x.question_index),
             "question": rs.QUESTIONS[int(x.question_index)],
             "conversation": [{"role": "user", "content": rs.QUESTIONS[int(x.question_index)]},
                              {"role": "assistant", "content": x.response}]} for x in g.itertuples()]


def run_judge(args, data: Path, out: Path) -> int:
    """Each reply judged as its own role by the authors' role-playing judge prompt."""
    from persona_neurons import judge

    run_dir = out / "judge"
    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    replies = read_replies(out, args.roles)
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

    refuse([out / "labels.parquet"])
    batch_ids = json.loads((run_dir / "batches.json").read_text())
    try:
        texts = judge.collect(batch_ids, run_dir)
    except judge.BatchNotEnded as e:
        print(f"not yet: {e}")
        return 3
    cids = [judge.custom_id(r, reply_key(k, d, s, q)) for r, k, d, s, q in
            zip(replies.role, replies.kind, replies.draw, replies["sample"], replies.question_index)]
    scores = [parse_judge_score(texts[c]) if texts.get(c) is not None else None for c in cids]
    labels = replies[["role", "condition_id", "question_index", "sample"]].assign(custom_id=cids, score=scores)
    labels.to_parquet(out / "labels.parquet", index=False)
    print(f"labels: {labels.score.notna().sum()} of {len(labels)} replies scored")
    return 0


def run_build(args, data: Path, out: Path) -> None:
    refuse([out / "judged_replies.parquet"])
    replies = read_replies(out, args.roles)
    labels = pd.read_parquet(out / "labels.parquet")
    j = replies.merge(labels[["role", "condition_id", "question_index", "sample", "score"]],
                      on=["role", "condition_id", "question_index", "sample"], validate="1:1")
    j.to_parquet(out / "judged_replies.parquet", index=False)
    print(in_role_summary(j).round(3).to_string())


def in_role_summary(j: pd.DataFrame) -> pd.DataFrame:
    """Per kind: the number of scored replies and the share of them scored 2 or 3 (unscored replies left out)."""
    s = j[j.score.notna()]
    return s.assign(in_role=s.score >= 2).groupby("kind").agg(scored=("score", "size"), in_role=("in_role", "mean"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["prepare", "components", "rotate", "finish", "generate", "judge", "build"])
    ap.add_argument("action", nargs="?", choices=["submit", "collect"], help="judge: submit or collect")
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
    if args.part in ("components", "rotate") and not args.layers:
        ap.error(f"{args.part} needs --layers")
    if args.part == "judge" and not args.action:
        ap.error("judge needs submit or collect")
    data = config.data_dir()
    out = data / "role_steering"
    run = {"prepare": run_prepare, "components": run_components, "rotate": run_rotate, "finish": run_finish,
           "generate": run_generate, "judge": run_judge, "build": run_build}[args.part]
    return run(args, data, out) or 0


if __name__ == "__main__":
    sys.exit(main())
