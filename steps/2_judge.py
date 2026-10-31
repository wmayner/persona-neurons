#!/usr/bin/env python3
"""Step 2 (API): each role-playing reply labelled 0 to 3 by the authors' role-playing judge prompt, run with Claude
Haiku 5.5 through the Message Batches API (`persona_neurons.judge`).

    uv run steps/2_judge.py submit [--roles ROLE ...] [--limit N] [--dry-run]
    uv run steps/2_judge.py collect

`submit` sends every reply in responses/ with no score yet in scores/, for every role whose file in the authors'
repository has an `eval_prompt` (the default role has none and is not judged), and exits; batch IDs go to
judge/batches.json. `collect` runs once every batch has ended (at most 24 hours): it writes scores/{role}.json (reply
key to score) and lists failed requests in judge/failed.jsonl. Another `submit` sends whatever is still unscored.
Needs `ANTHROPIC_API_KEY` in the environment.
"""

import argparse
import json
import sys
from pathlib import Path

import jsonlines

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_neurons import config, judge
from persona_neurons.upstream import fetch_upstream, use_upstream


def build_requests(data: Path, roles_dir: Path, roles: list[str] | None, limit: int | None) -> list[dict]:
    requests = []
    for f in sorted((data / "responses").glob("*.jsonl")):
        role = f.stem
        if roles and role not in roles:
            continue
        role_file = roles_dir / f"{role}.json"
        eval_prompt = json.loads(role_file.read_text()).get("eval_prompt", "") if role_file.exists() else ""
        if not eval_prompt:
            print(f"skip {role}: no eval_prompt")
            continue
        scores_file = data / "scores" / f"{role}.json"
        existing = json.loads(scores_file.read_text()) if scores_file.exists() else {}
        with jsonlines.open(f) as reader:
            responses = list(reader)
        requests += judge.build_role_requests(role, responses, eval_prompt, existing_scores=existing)
    return requests[:limit] if limit else requests


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--roles", nargs="+", help="only these roles")
    s.add_argument("--limit", type=int, help="send at most this many requests (for a trial run)")
    s.add_argument("--dry-run", action="store_true", help="count the requests without sending them")
    sub.add_parser("collect")
    args = ap.parse_args(argv)

    data = config.data_dir()
    run_dir = data / "judge"
    checkout = fetch_upstream(config.upstream_dir())
    use_upstream(checkout)
    if args.cmd == "submit":
        requests = build_requests(data, checkout / "data" / "roles" / "instructions", args.roles, args.limit)
        chars = sum(len(r["params"]["messages"][0]["content"]) for r in requests)
        print(f"{len(requests)} requests, {chars:,} characters of prompt (about {chars // 4:,} tokens)")
        if requests and not args.dry_run:
            print("submitted batches:", " ".join(judge.submit(requests, run_dir)))
        return 0

    batch_ids = json.loads((run_dir / "batches.json").read_text())
    pending = [b for b in batch_ids if not (run_dir / "results" / f"{b}.jsonl").exists()]
    if not pending:
        print("every batch is already collected")
        return 0
    try:
        results = judge.collect(pending, run_dir)
    except judge.BatchNotEnded as e:
        print(f"not yet: {e}")
        return 3
    judge.write_scores(results, data / "scores")
    n_none = sum(v is None for v in results.values())
    print(f"collected {len(results)} results from {len(pending)} batches; {n_none} refused, errored or expired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
