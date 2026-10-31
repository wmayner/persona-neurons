"""The role-playing judge (`persona_neurons.judge`). No network: the Batches client is faked."""

import asyncio
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from persona_neurons import judge
from persona_neurons.upstream import use_upstream
MODEL = "claude-haiku-5-5"


@pytest.fixture(scope="session")
def upstream(upstream_checkout):
    """The checkout of the authors' repository, first on sys.path."""
    use_upstream(upstream_checkout)
    return upstream_checkout


def _response(prompt_index, question_index, answer, question="Who are you?"):
    return {
        "system_prompt": "You are an aberration.",
        "prompt_index": prompt_index,
        "question_index": question_index,
        "question": question,
        "conversation": [
            {"role": "system", "content": "You are an aberration."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "label": "pos",
    }


def _eval_prompt(checkout, role="aberration"):
    return json.loads((checkout / "data/roles/instructions" / f"{role}.json").read_text())["eval_prompt"]


def _load_upstream_judge_script(checkout):
    module_spec = importlib.util.spec_from_file_location("upstream_3_judge", checkout / "pipeline/3_judge.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_request_text_matches_upstream(upstream, monkeypatch):
    responses = [
        _response(0, 0, "I am a rupture in the order of things. {braces} stay as they are."),
        _response(3, 17, "As an AI, I can't be that, but here is some help.", question="What do you value?"),
    ]
    eval_prompt = _eval_prompt(upstream)

    script = _load_upstream_judge_script(upstream)
    captured = {}

    async def fake_call_judge_batch(client, prompts, model, max_tokens, rate_limiter, batch_size=50):
        captured["prompts"] = list(prompts)
        captured["max_tokens"] = max_tokens
        return ["3"] * len(prompts)

    monkeypatch.setattr(script, "call_judge_batch", fake_call_judge_batch)
    scores = asyncio.run(script.process_role(
        role="aberration", responses=responses, eval_prompt_template=eval_prompt, client=None, rate_limiter=None,
        judge_model="unused", max_tokens=10, batch_size=50, existing_scores={},
    ))

    requests = judge.build_role_requests("aberration", responses, eval_prompt, MODEL)
    assert [r["params"]["messages"] for r in requests] == [
        [{"role": "user", "content": p}] for p in captured["prompts"]
    ]
    assert [judge.parse_custom_id(r["custom_id"]) for r in requests] == [("aberration", k) for k in scores]
    for r in requests:
        params = r["params"]
        assert params["model"] == MODEL
        assert params["max_tokens"] == 10  # the authors' 3_judge.py's --max_tokens default
        assert params["thinking"] == {"type": "disabled"}
        assert not {"temperature", "top_p", "top_k", "system"} & params.keys()


def test_custom_id_roundtrip(upstream):
    roles = sorted(p.stem for p in (upstream / "data/roles/instructions").glob("*.json"))
    assert len(roles) == 276
    for role in roles:
        for key in ("pos_p0_q0", "pos_p4_q239"):
            cid = judge.custom_id(role, key)
            assert judge.CUSTOM_ID_RE.fullmatch(cid)
            assert judge.parse_custom_id(cid) == (role, key)


def test_custom_id_rejects_unsafe():
    with pytest.raises(ValueError):
        judge.custom_id("x" * 60, "pos_p4_q239")
    with pytest.raises(ValueError):
        judge.custom_id("devil's advocate", "pos_p0_q0")


def test_missing_never_zero(upstream, tmp_path):
    cid = lambda q: judge.custom_id("aberration", f"pos_p0_q{q}")
    results = {cid(0): None, cid(1): "", cid(2): "I can't rate this", cid(3): "3", cid(4): "0"}
    judge.write_scores(results, tmp_path)

    scores = json.loads((tmp_path / "aberration.json").read_text())
    missing = json.loads((tmp_path / "aberration.missing.json").read_text())
    assert scores == {"pos_p0_q3": 3, "pos_p0_q4": 0}
    assert set(missing) == {"pos_p0_q0", "pos_p0_q1", "pos_p0_q2"}


class FakeBatches:
    """Stands in for client.messages.batches. Results come back shuffled with the given seed."""

    def __init__(self, outcomes, seed=0, status="ended"):
        self.outcomes = outcomes  # custom_id -> (type, stop_reason, text)
        self.created = []
        self.seed = seed
        self.status = status

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status=self.status)

    def results(self, batch_id):
        items = []
        for cid, (kind, stop_reason, text) in self.outcomes.items():
            if kind == "succeeded":
                content = [SimpleNamespace(type="text", text=text)] if text is not None else []
                message = SimpleNamespace(content=content, stop_reason=stop_reason,
                                          usage=SimpleNamespace(input_tokens=100, output_tokens=1))
                result = SimpleNamespace(type="succeeded", message=message)
            elif kind == "errored":
                result = SimpleNamespace(type="errored", error=SimpleNamespace(
                    type="error", error=SimpleNamespace(type="overloaded_error", message="busy")))
            else:
                result = SimpleNamespace(type=kind)
            items.append(SimpleNamespace(custom_id=cid, result=result))
        random.Random(self.seed).shuffle(items)
        return iter(items)


def _client(batches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def test_results_matched_by_id(upstream, tmp_path):
    outcomes = {judge.custom_id(role, f"pos_p{p}_q{q}"): ("succeeded", "end_turn", str((p + q) % 4))
                for role in ("aberration", "accountant") for p in range(5) for q in range(6)}
    written = []
    for seed in (0, 1):
        run_dir, scores_dir = tmp_path / f"run{seed}", tmp_path / f"scores{seed}"
        results = judge.collect(["msgbatch_1"], run_dir, client=_client(FakeBatches(outcomes, seed=seed)))
        judge.write_scores(results, scores_dir)
        written.append({p.name: json.loads(p.read_text()) for p in sorted(scores_dir.glob("*.json"))})
    assert written[0] == written[1]
    assert written[0]["accountant.json"]["pos_p2_q3"] == 1


def test_errored_written_for_resubmission(tmp_path):
    cid = lambda q: judge.custom_id("aberration", f"pos_p0_q{q}")
    outcomes = {
        cid(0): ("succeeded", "end_turn", "2"),
        cid(1): ("succeeded", "refusal", None),
        cid(2): ("errored", None, None),
        cid(3): ("expired", None, None),
        cid(4): ("canceled", None, None),
    }
    results = judge.collect(["msgbatch_1"], tmp_path, client=_client(FakeBatches(outcomes)))
    assert results == {cid(0): "2", cid(1): None, cid(2): None, cid(3): None, cid(4): None}

    failed = [json.loads(line) for line in (tmp_path / "failed.jsonl").read_text().splitlines()]
    assert {f["custom_id"]: f["reason"] for f in failed} == {
        cid(1): "refusal", cid(2): "errored", cid(3): "expired", cid(4): "canceled"}
    # The raw judge replies are kept, one line per request.
    raw = (tmp_path / "results" / "msgbatch_1.jsonl").read_text().splitlines()
    assert len(raw) == 5


def test_collect_refuses_unfinished_batch(tmp_path):
    with pytest.raises(judge.BatchNotEnded):
        judge.collect(["msgbatch_1"], tmp_path, client=_client(FakeBatches({}, status="in_progress")))


def test_rerun_skips_scored(upstream, tmp_path):
    responses = [_response(p, q, "text") for p in range(2) for q in range(3)]
    eval_prompt = _eval_prompt(upstream)
    existing = {"pos_p0_q0": 3, "pos_p1_q2": 1}
    requests = judge.build_role_requests("aberration", responses, eval_prompt, MODEL, existing_scores=existing)
    keys = {judge.parse_custom_id(r["custom_id"])[1] for r in requests}
    assert len(keys) == 4 and not keys & existing.keys()


def test_write_scores_keeps_existing(upstream, tmp_path):
    (tmp_path / "aberration.json").write_text(json.dumps({"pos_p0_q0": 2}))
    judge.write_scores({judge.custom_id("aberration", "pos_p0_q0"): "3",
                        judge.custom_id("aberration", "pos_p0_q1"): "1"}, tmp_path)
    assert json.loads((tmp_path / "aberration.json").read_text()) == {"pos_p0_q0": 2, "pos_p0_q1": 1}


def test_submit_splits_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "MAX_BATCH_REQUESTS", 3)
    requests = [{"custom_id": judge.custom_id("aberration", f"pos_p0_q{q}"),
                 "params": {"model": MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}}
                for q in range(7)]
    batches = FakeBatches({})
    ids = judge.submit(requests, tmp_path, client=_client(batches))
    assert [len(b) for b in batches.created] == [3, 3, 1]
    assert json.loads((tmp_path / "batches.json").read_text()) == ids == ["msgbatch_1", "msgbatch_2", "msgbatch_3"]

    # A second submission adds to the record rather than replacing it.
    judge.submit(requests[:1], tmp_path, client=_client(batches))
    assert len(json.loads((tmp_path / "batches.json").read_text())) == 4


def test_submit_splits_by_size(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "MAX_BATCH_BYTES", 1000)
    requests = [{"custom_id": judge.custom_id("aberration", f"pos_p0_q{q}"),
                 "params": {"model": MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": "y" * 400}]}}
                for q in range(5)]
    batches = FakeBatches({})
    judge.submit(requests, tmp_path, client=_client(batches))
    assert all(len(json.dumps(b)) <= 1000 for b in batches.created)
    assert sum(len(b) for b in batches.created) == 5



def test_text_ignores_thinking_blocks(tmp_path):
    cid = judge.custom_id("aberration", "pos_p0_q0")
    batches = FakeBatches({cid: ("succeeded", "end_turn", "3")})
    original = batches.results

    def with_thinking(batch_id):
        for item in original(batch_id):
            item.result.message.content.insert(0, SimpleNamespace(type="thinking", thinking=""))
            yield item

    batches.results = with_thinking
    assert judge.collect(["msgbatch_1"], tmp_path, client=_client(batches)) == {cid: "3"}


