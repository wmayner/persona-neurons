"""The role-playing judge, run with Claude Haiku 5.5 (thinking off) through the Message Batches API.

Like the authors' `pipeline/3_judge.py`, it sends each role's `eval_prompt`, filled with the question and the first
assistant message, as one user message with `max_tokens=10`, and parses the reply with
`assistant_axis.judge.parse_judge_score`. Scores are written in the authors' format (`{role}.json`, reply key to score
0 to 3). A reply with no score (refusal, error, empty or unparsable) goes to `{role}.missing.json`, never as a score.
The Anthropic client reads the key from `ANTHROPIC_API_KEY`.
"""

import json
import re
from pathlib import Path

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 10  # the authors' 3_judge.py's --max_tokens default
MAX_BATCH_REQUESTS = 100_000  # the Batches API's limit per batch
MAX_BATCH_BYTES = 200_000_000  # below the API's 256 MB per batch
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
THINKING_OFF = {"type": "disabled"}


class BatchNotEnded(RuntimeError):
    pass


def judge_params(model: str = MODEL, max_tokens: int = MAX_TOKENS) -> dict:
    """The model settings of a judge request, with thinking off."""
    return {"model": model, "max_tokens": max_tokens, "thinking": THINKING_OFF}


def response_key(resp: dict) -> str:
    return f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"


def assistant_text(resp: dict) -> str:
    """The first assistant message, as the authors' judge step takes it."""
    for msg in resp["conversation"]:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def custom_id(role: str, key: str) -> str:
    cid = f"{role}__{key}"
    if not CUSTOM_ID_RE.fullmatch(cid) or "__" in key:
        raise ValueError(f"cannot make a Batches custom_id from role {role!r} and key {key!r}")
    return cid


def parse_custom_id(cid: str) -> tuple[str, str]:
    role, _, key = cid.rpartition("__")
    return role, key


def build_role_requests(role: str, responses: list[dict], eval_prompt: str, model: str = MODEL,
                        existing_scores: dict | None = None) -> list[dict]:
    """One Batches request per reply not already scored, with the user message the authors' judge step sends."""
    existing_scores = existing_scores or {}
    requests = []
    for resp in responses:
        key = response_key(resp)
        if key in existing_scores:
            continue
        prompt = eval_prompt.format(question=resp["question"], answer=assistant_text(resp))
        requests.append({
            "custom_id": custom_id(role, key),
            "params": {**judge_params(model),
                       "messages": [{"role": "user", "content": prompt}]},
        })
    return requests


def make_client():
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def _chunks(requests: list[dict]):
    chunk, size = [], 2
    for r in requests:
        n = len(json.dumps(r)) + 2
        if chunk and (len(chunk) >= MAX_BATCH_REQUESTS or size + n > MAX_BATCH_BYTES):
            yield chunk
            chunk, size = [], 2
        chunk.append(r)
        size += n
    if chunk:
        yield chunk


def submit(requests: list[dict], run_dir: Path, client=None) -> list[str]:
    """Submit in batches within the API's limits; add the batch IDs to run_dir/batches.json."""
    client = client or make_client()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = run_dir / "batches.json"
    all_ids = json.loads(record.read_text()) if record.exists() else []
    ids = []
    for chunk in _chunks(requests):
        batch = client.messages.batches.create(requests=chunk)
        ids.append(batch.id)
        all_ids.append(batch.id)
        record.write_text(json.dumps(all_ids, indent=2))  # after each batch, so a crash loses no ID
    return ids


def _text(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def collect(batch_ids: list[str], run_dir: Path, client=None) -> dict[str, str | None]:
    """The judge's reply text by custom_id, for batches that have ended.

    Refused, errored, canceled and expired requests map to None and are appended to run_dir/failed.jsonl. Every
    result is also kept raw in run_dir/results/{batch_id}.jsonl.
    """
    client = client or make_client()
    run_dir = Path(run_dir)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    for batch_id in batch_ids:
        status = client.messages.batches.retrieve(batch_id).processing_status
        if status != "ended":
            raise BatchNotEnded(f"batch {batch_id} is {status}")

    texts: dict[str, str | None] = {}
    failed = []
    for batch_id in batch_ids:
        raw = []
        for item in client.messages.batches.results(batch_id):
            result = item.result
            row = {"custom_id": item.custom_id, "type": result.type}
            text, reason = None, result.type
            if result.type == "succeeded":
                message = result.message
                row["stop_reason"] = message.stop_reason
                row["input_tokens"] = message.usage.input_tokens
                row["output_tokens"] = message.usage.output_tokens
                if message.stop_reason == "refusal":
                    reason = "refusal"
                else:
                    text, reason = _text(message), None
                row["text"] = text
            elif result.type == "errored":
                row["error"] = getattr(getattr(result.error, "error", None), "type", None)
            texts[item.custom_id] = text
            raw.append(row)
            if reason:
                failed.append({"custom_id": item.custom_id, "batch_id": batch_id, "reason": reason})
        out = run_dir / "results" / f"{batch_id}.jsonl"
        if not out.exists():
            out.write_text("".join(json.dumps(r) + "\n" for r in raw))
    if failed:
        with open(run_dir / "failed.jsonl", "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in failed)
    return texts


def write_scores(results: dict[str, str | None], scores_dir: Path) -> None:
    """Parse with the authors' parse_judge_score and add to {role}.json; replies with no score go to
    {role}.missing.json with their text. A key already scored keeps its score."""
    from assistant_axis.judge import parse_judge_score

    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    by_role: dict[str, dict[str, str | None]] = {}
    for cid, text in results.items():
        role, key = parse_custom_id(cid)
        by_role.setdefault(role, {})[key] = text

    for role, replies in sorted(by_role.items()):
        scores_file, missing_file = scores_dir / f"{role}.json", scores_dir / f"{role}.missing.json"
        scores = json.loads(scores_file.read_text()) if scores_file.exists() else {}
        missing = json.loads(missing_file.read_text()) if missing_file.exists() else {}
        for key, text in sorted(replies.items()):
            if key in scores:
                continue
            score = parse_judge_score(text) if text else None
            if score is None:
                missing[key] = text
            else:
                scores[key] = score
                missing.pop(key, None)
        scores_file.write_text(json.dumps(scores, indent=2))
        if missing:
            missing_file.write_text(json.dumps(missing, indent=2))
        elif missing_file.exists():
            missing_file.unlink()
