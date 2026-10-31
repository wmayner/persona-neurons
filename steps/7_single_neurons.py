#!/usr/bin/env python3
"""Step 7: steering along single neurons' output directions, random mixtures of neurons, random directions, principal
components and role vectors in the protocol of the Qwen3-8B neuron persona library, each reply labelled by one judge
(`persona_neurons.single_neurons`). Everything is written under single_neurons/ in the data directory:

  prepare        (CPU) inputs/: plan.parquet (every steered reply), roles.json, role_contrasts.pt (v_r(L), role minus
                 default, from vectors/all/), pcs.npz (the top 50 principal components of the per-reply mean
                 residuals of 48 replies per role and 48 default replies, from activations/all/),
                 natural_corpus.parquet (every default reply and 1,000 role replies drawn with seed 20261027, with each
                 token's weight in the per-reply mean), prepare.json
  natural        (GPU) per layer: r_L, each neuron's range over the corpus's reply tokens (natural/natural_range_L{L}
                 .parquet), the down_proj column norms (natural/w_norms_L{L}.npy)
  push           (GPU) the steered replies, greedy, in chunks in replies/ (finished chunks are skipped); needs natural
  collect        (CPU) pushes.parquet: every reply, cut where a loop starts
  label submit   (API) the labelling requests, as Message Batches; label collect, once they have ended: labelled.parquet
                 (every steered reply with its label) and kept.parquet (the strongest coherent reply of each direction
                 and sign)
  tables         (CPU) tables/: share.parquet and share_natural.parquet (the share labelled persona per kind, sign and
                 layer, with 95% Wilson intervals), push_size.parquet (each kept neuron reply's steering size over
                 the neuron's largest value)

    uv run steps/7_single_neurons.py prepare
    uv run steps/7_single_neurons.py natural [--shard i/n]
    uv run steps/7_single_neurons.py push [--shard i/n]
    uv run steps/7_single_neurons.py collect
    uv run steps/7_single_neurons.py label submit      (later: label collect)
    uv run steps/7_single_neurons.py tables
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config
from persona_neurons import single_neurons as sn
from persona_neurons.shards import shard

SUBSAMPLE_SEED, SUBSAMPLE_N = 20261027, 48
CORPUS_SEED, CORPUS_N_ROLE, CORPUS_MAX_LENGTH = 20261027, 1000, 2048


def root() -> Path:
    return config.data_dir() / "single_neurons"


def refuse(*paths: Path) -> None:
    there = [str(p) for p in paths if p.exists()]
    if there:
        sys.exit(f"{there} exist; move them away to redo this command")


def tokenizer():
    from transformers import AutoTokenizer
    from persona_neurons.upstream import model_snapshot

    files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]
    path = model_snapshot(config.MODEL_ID, config.MODEL_REVISION, allow_patterns=files)
    return AutoTokenizer.from_pretrained(path), path


def natural_corpus(data: Path, tok, model_name: str) -> pd.DataFrame:
    """Every default reply and CORPUS_N_ROLE replies drawn with CORPUS_SEED from the sorted (role, key) pairs of all
    roles' replies, each as token ids cut at CORPUS_MAX_LENGTH with each token's weight in the authors' per-reply mean."""
    from assistant_axis.internals import ConversationEncoder
    from persona_neurons.activations import response_weights

    def read(path):
        return {f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}": r["conversation"]
                for r in map(json.loads, path.open())}
    default = read(data / "responses" / "default.jsonl")
    roles = {p.stem: read(p) for p in sorted((data / "responses").glob("*.jsonl")) if p.stem != "default"}
    pairs = sorted((r, k) for r, ks in roles.items() for k in ks)
    idx = np.random.default_rng(CORPUS_SEED).choice(len(pairs), size=CORPUS_N_ROLE, replace=False)
    docs = [("default", "default", k, c) for k, c in default.items()] + \
           [("role", r, k, roles[r][k]) for r, k in (pairs[i] for i in sorted(idx))]
    encoder = ConversationEncoder(tok, model_name)
    rows = []
    for corpus, role, key, conv in docs:
        full_ids, spans, _ = encoder.build_batch_turn_spans([conv], enable_thinking=False)
        n = min(len(full_ids[0]), CORPUS_MAX_LENGTH)
        rows.append(dict(corpus=corpus, role=role, key=key, token_ids=full_ids[0][:n],
                         weights=response_weights(spans, [n], n)[0].numpy()))
    return pd.DataFrame(rows)


def prepare(args) -> None:
    from persona_neurons.activations import select_subsample
    from persona_neurons.upstream import fetch_upstream, use_upstream

    data, out = config.data_dir(), root() / "inputs"
    refuse(out / "plan.parquet")
    out.mkdir(parents=True, exist_ok=True)
    layers = list(args.layers)
    use_upstream(fetch_upstream(config.upstream_dir()))

    roles, v = sn.role_contrasts(data / "vectors" / "all", layers)
    torch.save({"layers": layers, "contrasts": v}, out / "role_contrasts.pt")
    (out / "roles.json").write_text(json.dumps(roles))
    print(f"{len(roles)} roles", flush=True)

    # principal components of the per-reply mean residuals of a seeded subsample of each role's and the default's replies
    xs, is_default = {l: [] for l in layers}, []
    li = torch.tensor(layers)
    files = sorted((data / "activations" / "all").glob("*.pt"))
    for f in files:
        acts = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        keys = select_subsample(list(acts), SUBSAMPLE_SEED, SUBSAMPLE_N)
        x = torch.stack([acts[k][li].float() for k in keys])
        for j, l in enumerate(layers):
            xs[l].append(x[:, j].numpy())
        is_default += [f.stem == "default"] * len(keys)
        del acts
    is_default = np.array(is_default)
    pcs, proj = [], []
    for l in layers:
        vt, m = sn.principal_components(np.concatenate(xs[l]), is_default, args.n_pc)
        pcs.append(vt.astype(np.float32))
        proj.append(m)
    np.savez(out / "pcs.npz", layers=np.array(layers), pcs=np.stack(pcs), proj_max=np.stack(proj),
             n_rows=len(is_default), n_default=int(is_default.sum()))
    print(f"principal components over {len(is_default)} replies", flush=True)

    tok, path = tokenizer()
    corpus = natural_corpus(data, tok, str(path))
    corpus.to_parquet(out / "natural_corpus.parquet", index=False)
    print(corpus.groupby("corpus").size().to_string(), flush=True)

    neurons = {l: [int(n) for n in sn.sample_neurons(l, args.n_neurons)] for l in layers}
    plan = pd.concat([sn.push_plan(l, neurons[l], args.n_random, args.n_random, args.n_pc, len(roles)) for l in layers],
                     ignore_index=True)
    plan.to_parquet(out / "plan.parquet", index=False)
    (out / "prepare.json").write_text(json.dumps(dict(
        seed=sn.SEED, streams={"neurons": 0, "random directions": 2, "random mixtures": 3}, layers=layers,
        neurons=neurons, n_random=args.n_random, n_pc=args.n_pc, subsample_seed=SUBSAMPLE_SEED,
        subsample_n=SUBSAMPLE_N, corpus_seed=CORPUS_SEED, corpus_n_role=CORPUS_N_ROLE), indent=1))
    print(plan.groupby(["layer", "kind"]).size().unstack().to_string())
    print(f"{len(plan):,} steered replies to make")


def load_model(dtype: str):
    """Qwen3-8B at the fixed revision, loaded as the authors' code loads it: (model, tokenizer)."""
    from persona_neurons.upstream import fetch_upstream, model_snapshot, use_upstream

    use_upstream(fetch_upstream(config.upstream_dir()))
    from assistant_axis.internals import ProbingModel

    pm = ProbingModel(str(model_snapshot(config.MODEL_ID, config.MODEL_REVISION)), dtype=getattr(torch, dtype))
    return pm.model.eval(), pm.tokenizer


def natural(args) -> None:
    prep, out = root() / "inputs", root() / "natural"
    out.mkdir(parents=True, exist_ok=True)
    layers = shard(json.loads((prep / "prepare.json").read_text())["layers"], args.shard)
    layers = [l for l in layers if not (out / f"natural_range_L{l}.parquet").exists()]
    if not layers:
        return
    model, tok = load_model(args.dtype)
    corpus = pd.read_parquet(prep / "natural_corpus.parquet")
    t0 = time.time()
    nat = sn.natural_range(model, tok, [list(map(int, t)) for t in corpus.token_ids],
                           [np.asarray(w) for w in corpus.weights], layers, batch_size=args.batch_size)
    r_l = sn.median_norms(model, tok, layers)
    for l in layers:
        (out / f"r_L{l}.json").write_text(json.dumps(dict(layer=l, r_L=r_l[l])))
        w = model.model.layers[l].mlp.down_proj.weight.detach()
        np.save(out / f"w_norms_L{l}.npy", w.float().norm(dim=0).cpu().numpy())
        nat[nat.layer == l].to_parquet(out / f"natural_range_L{l}.parquet", index=False)
    print(f"layers {layers}: natural range in {time.time() - t0:.0f} s", flush=True)


def directions(model, prep: Path, info: dict, layer: int) -> dict:
    """Every steered direction at one layer: kind -> (index-to-row map or None, unit directions as rows)."""
    w = model.model.layers[layer].mlp.down_proj.weight.detach()
    neurons = info["neurons"][str(layer)]
    contrasts = torch.load(prep / "role_contrasts.pt", weights_only=False)
    v = contrasts["contrasts"][:, contrasts["layers"].index(layer)].float()
    pcs = np.load(prep / "pcs.npz")
    n = int(info["n_random"])
    return {"neuron": ({x: i for i, x in enumerate(neurons)}, sn.neuron_directions(w, neurons).cpu()),
            "random": (None, sn.random_directions(w.shape[0], n, sn.stream_rng(layer, 2))),
            "combination": (None, sn.random_combinations(w, n, sn.stream_rng(layer, 3)).cpu()),
            "pc": (None, torch.tensor(pcs["pcs"][list(pcs["layers"]).index(layer)])),
            "role": (None, v / v.norm(dim=-1, keepdim=True))}


def push(args) -> None:
    prep, nat_dir, out = root() / "inputs", root() / "natural", root() / "replies"
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((prep / "prepare.json").read_text())
    plan = pd.read_parquet(prep / "plan.parquet")
    plan = plan[plan.push_id.isin(shard(plan.push_id, args.shard))]
    tag = (args.shard or "0/1").replace("/", "of")
    model, tok = load_model(args.dtype)
    t0 = time.time()
    for layer, rows in plan.groupby("layer"):
        rows = rows.sort_values("push_id").reset_index(drop=True)
        r_l = json.loads((nat_dir / f"r_L{layer}.json").read_text())["r_L"]
        natural_l = pd.read_parquet(nat_dir / f"natural_range_L{layer}.parquet")
        dirs = directions(model, prep, info, layer)
        pcs = np.load(prep / "pcs.npz")
        contrasts = torch.load(prep / "role_contrasts.pt", weights_only=False)
        role_norms = contrasts["contrasts"][:, contrasts["layers"].index(layer)].float().norm(dim=-1).numpy()
        sizes = sn.natural_sizes(natural_l, model.model.layers[layer].mlp.down_proj.weight.detach(),
                                 info["neurons"][str(layer)], int(info["n_random"]), int(info["n_random"]),
                                 pcs["proj_max"][list(pcs["layers"]).index(layer)], role_norms)
        c, vectors = sn.push_vectors(rows, dirs, r_l, sizes)
        rows["c"] = c
        for k, start in enumerate(range(0, len(rows), args.batch_size)):
            path = out / f"L{layer:02d}_shard{tag}_chunk{k:04d}.parquet"
            if path.exists():
                continue
            idx = list(range(start, min(start + args.batch_size, len(rows))))
            got = pd.DataFrame(sn.greedy_replies(model, tok, layer, vectors[idx]))
            pd.concat([rows.loc[idx].reset_index(drop=True), got], axis=1).to_parquet(path, index=False)
            print(f"layer {layer}: {idx[-1] + 1} of {len(rows)} steered replies, {time.time() - t0:.0f} s", flush=True)


def collect(args) -> None:
    run = root()
    refuse(run / "pushes.parquet")
    plan = pd.read_parquet(run / "inputs" / "plan.parquet")
    replies = pd.concat([pd.read_parquet(p) for p in sorted((run / "replies").glob("*.parquet"))], ignore_index=True)
    if replies.push_id.duplicated().any():
        sys.exit(f"{replies.push_id.duplicated().sum()} steered replies appear twice in replies/")
    missing = set(plan.push_id) - set(replies.push_id)
    if missing and not args.allow_missing:
        sys.exit(f"{len(missing)} of the plan's steered replies are missing; run push for every shard")
    sn.cut_replies(replies, tokenizer()[0]).to_parquet(run / "pushes.parquet", index=False)
    print(f"{len(replies):,} replies", flush=True)


def label(args) -> None:
    from persona_neurons import judge

    run = root()
    if args.action == "submit":
        if (run / "labels" / "batches.json").exists() and not args.resubmit_failed:
            sys.exit("labels were submitted already; pass --resubmit-failed to send only the requests without a text")
        existing = (judge.collect(json.loads((run / "labels" / "batches.json").read_text()), run / "labels")
                    if args.resubmit_failed else {})
        pushes = pd.read_parquet(run / "pushes.parquet", columns=["push_id", "reply_cut"])
        reqs = sn.label_requests(pushes.rename(columns={"reply_cut": "reply"}), existing=existing)[:args.limit]
        print(f"{len(reqs):,} requests", flush=True)
        if reqs:
            print(judge.submit(reqs, run / "labels"))
        return
    refuse(run / "labelled.parquet", run / "kept.parquet")
    texts = judge.collect(json.loads((run / "labels" / "batches.json").read_text()), run / "labels")
    pushes = pd.read_parquet(run / "pushes.parquet")
    labels = pd.DataFrame([dict(push_id=p, label_text=texts.get(p), **(sn.parse_label(texts.get(p)) or {}))
                           for p in pushes.push_id])
    labelled = sn.label_empty_replies(pushes.merge(labels, on="push_id", how="left"))
    print(f"{(labelled.label_source == 'empty reply').sum()} empty replies labelled incoherent; "
          f"{labelled.category.isna().sum()} steered replies without a label", flush=True)
    labelled.to_parquet(run / "labelled.parquet", index=False)
    scaled = labelled[labelled.alpha.notna() & labelled.category.notna()]
    sn.keep_strongest(scaled, by=["layer", "kind", "index", "sign"]).to_parquet(run / "kept.parquet", index=False)


def tables(args) -> None:
    run = root()
    out = run / "tables"
    refuse(out)
    out.mkdir()
    layers = json.loads((run / "inputs" / "prepare.json").read_text())["layers"]
    r_l = {l: json.loads((run / "natural" / f"r_L{l}.json").read_text())["r_L"] for l in layers}
    w_norms = {l: np.load(run / "natural" / f"w_norms_L{l}.npy") for l in layers}
    natural_all = pd.concat([pd.read_parquet(run / "natural" / f"natural_range_L{l}.parquet") for l in layers])
    kept = pd.read_parquet(run / "kept.parquet")
    sn.persona_share(kept).to_parquet(out / "share.parquet", index=False)
    labelled = pd.read_parquet(run / "labelled.parquet", columns=["layer", "kind", "sign", "size", "c", "category"])
    sn.natural_share(labelled, r_l).to_parquet(out / "share_natural.parquet", index=False)
    k = kept[kept.kind == "neuron"][["layer", "index", "sign", "alpha", "category"]].rename(columns={"index": "neuron"})
    sn.push_sizes(k, r_l, w_norms, natural_all).to_parquet(out / "push_size.parquet", index=False)
    print(f"tables in {out}: {sorted(p.name for p in out.iterdir())}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--layers", type=int, nargs="+", default=list(sn.LAYERS))
    p.add_argument("--n-neurons", type=int, default=sn.N_NEURONS)
    p.add_argument("--n-random", type=int, default=sn.N_RANDOM, help="random directions, and random mixtures")
    p.add_argument("--n-pc", type=int, default=sn.N_PC)
    for name, batch in (("natural", 16), ("push", 128)):
        g = sub.add_parser(name)
        g.add_argument("--shard", help="i/n: the i-th of n parts (i from 0)")
        g.add_argument("--batch-size", type=int, default=batch)
        g.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    col = sub.add_parser("collect")
    col.add_argument("--allow-missing", action="store_true", help="collect the replies there are (for a trial run)")
    lab = sub.add_parser("label")
    lab.add_argument("action", choices=["submit", "collect"])
    lab.add_argument("--resubmit-failed", action="store_true")
    lab.add_argument("--limit", type=int, help="submit only the first N requests (for a trial run)")
    sub.add_parser("tables")
    args = ap.parse_args(argv)
    {"prepare": prepare, "natural": natural, "push": push, "collect": collect, "label": label,
     "tables": tables}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
