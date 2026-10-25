"""Regression checks for leakage, bounded agents, dataset integrity and resume."""

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from alarb import agentic, data, dataset_bundle, providers, runner, tasks
from alarb.judge import HeuristicJudge


def case(ident="train", fact="يجب إثبات عقد البيع", law="نظام الإثبات:1: يجب إثبات عقد البيع"):
    return data.Case(ident, (fact,), ("SECRET_COURT_REASONING",), (law,), "SECRET_GOLD_VERDICT")


class ScriptedModel(providers.Model):
    name = "test:scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        text = self.outputs.pop(0)
        return providers.Completion(text, self.name, 1, 1)


def answer(citations=None, abstain=False):
    return {"verdict": "حكم تجريبي", "reasoning": "استدلال تجريبي",
            "citations": citations or [], "abstain": abstain}


def test_agent_uses_facts_without_gold_fields():
    retriever = agentic.ArticleRetriever([case()])
    key = retriever.articles[0].key
    model = ScriptedModel([json.dumps(answer([key]))])
    result = agentic.analyze(agentic.FactInput("test", ("إثبات عقد البيع",)), retriever, model)
    assert result["verification"]["status"] == "PASSED"
    assert result["needs_human_review"]
    assert "SECRET_GOLD_VERDICT" not in model.prompts[0]
    assert "SECRET_COURT_REASONING" not in model.prompts[0]


def test_hallucinated_citation_triggers_one_repair():
    retriever = agentic.ArticleRetriever([case()])
    key = retriever.articles[0].key
    model = ScriptedModel([json.dumps(answer(["invented:99"])), json.dumps(answer([key]))])
    result = agentic.analyze(agentic.FactInput("test", ("عقد البيع",)), retriever, model)
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["verification"]["status"] == "FAILED"
    assert result["verification"]["status"] == "PASSED"
    assert "REPAIR_JSON" in model.prompts[1]


def test_failed_repair_stops_at_budget():
    retriever = agentic.ArticleRetriever([case()])
    model = ScriptedModel(["not json", "not json"])
    result = agentic.analyze(agentic.FactInput("test", ("عقد البيع",)), retriever, model)
    assert len(model.prompts) == 2
    assert result["verification"]["status"] == "FAILED"


@pytest.mark.parametrize("output", [None, {}, answer(["fabricated:3"]), answer(),
                                     {**answer(), "abstain": "false"},
                                     {**answer(), "citations": [123]}])
def test_verifier_rejects_invalid_or_unsupported_output(output):
    assert agentic.verify(output, [agentic.Hit("law:1", "text", 1.0)])["status"] == "FAILED"


def test_abstention_has_separate_status_without_claiming_correctness():
    check = agentic.verify(answer(abstain=True), [])
    assert check["status"] == "ABSTAINED"
    assert not check["legal_correctness_verified"]
    assert not check["citation_check_passed"]


def test_retrieval_returns_no_evidence_for_empty_overlap():
    retriever = agentic.ArticleRetriever([case()])
    assert retriever.retrieve(("zzzzzzzzzzz",)) == []


def test_experiment_indexes_train_only_and_never_requests_validation(tmp_path, monkeypatch):
    requested = []
    train = [case()]
    test = [case("test", law="نظام سري:99: TEST_ONLY_LAW")]
    def load(name):
        requested.append(name)
        return {"train": train, "test": test}[name]
    monkeypatch.setattr(data, "load_split", load)
    model = ScriptedModel([json.dumps(answer(abstain=True))])
    report = agentic.run_experiment(model, tmp_path / "agent", limit=1)
    assert requested == ["train", "test"]
    assert report["retrieval_macro_recall_at_k"] == 0.0
    assert "TEST_ONLY_LAW" not in model.prompts[0]
    assert "SECRET_GOLD_VERDICT" not in model.prompts[0]
    assert not report["validation_accessed"]


def test_dataset_bundle_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="integrity"):
        dataset_bundle.verify_file(path, {"bytes": 4, "sha256": "bad"})


def test_collection_preserves_bytes_and_can_be_verified_without_parquet_read(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    local = tmp_path / "raw"
    local.mkdir()
    file_specs = {}
    for name in ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet"):
        file = local / name
        file.write_bytes(b"opaque parquet bytes")
        file_specs[name] = {"bytes": file.stat().st_size, "sha256": dataset_bundle.sha256(file)}
    source = {"id": "THIQAH-RD/ALARB", "url": "https://huggingface.co/datasets/THIQAH-RD/ALARB",
              "git_commit": "revision", "license": "Apache-2.0", "files": file_specs}
    lock = root / "SOURCE_LOCK.json"
    lock.write_text(json.dumps({"official_dataset": source, "parent_paper": {"url": "paper"}}))
    card = root / ".cache/huggingface/hub/datasets--THIQAH-RD--ALARB/blobs/71551080570e3e00c7a6875b788522db632038a4"
    card.parent.mkdir(parents=True)
    card.write_text("license: apache-2.0")
    paths = {}
    for name in ("train", "test", "validation"):
        relative = f"data/{name}.parquet"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"opaque {name}".encode())
        paths[name] = relative
    (root / "artifacts").mkdir()
    (root / "artifacts/split_manifest.json").write_text(json.dumps({
        "source": {"revision": "revision"}, "paths": paths,
        "rows": {"train": 9339, "test": 2673, "validation": 1329}}))
    (root / "DATA_SPLIT_POLICY.md").write_text("sealed")
    (root / "licenses").mkdir()
    (root / "licenses/DATASET_APACHE_2_0.txt").write_text("Apache license fixture")
    monkeypatch.setattr(dataset_bundle, "ROOT", root)
    monkeypatch.setattr(dataset_bundle, "LOCK_PATH", lock)
    output = tmp_path / "dataset.zip"
    report = dataset_bundle.collect(output, local_dir=local, offline=True)
    assert report["status"] == "PASS" and not report["holdout_decoded"]
    with zipfile.ZipFile(output) as archive:
        assert archive.read("course/validation.parquet") == b"opaque validation"
        assert archive.testzip() is None
    with pytest.raises(FileExistsError):
        dataset_bundle.collect(output, local_dir=local, offline=True)


@pytest.fixture
def isolated_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    items = [tasks.TaskItem("verdict_facts", f"id{i}", f"facts {i}", "gold") for i in range(3)]
    monkeypatch.setattr(runner, "build_items", lambda config: items)
    monkeypatch.setattr(runner.provenance, "current", lambda task: {"prompt_digest": "fixed"})
    config = runner.RunConfig(task="verdict_facts", model="constant", limit=3)
    runner.run_inference(config, providers.ConstantModel(), progress_every=0)
    return config


def test_resume_recovers_truncated_tail_and_preserves_it(isolated_run):
    path = isolated_run.directory / "predictions.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b'{"case_id":"unfinished')
    runner.run_inference(isolated_run, providers.ConstantModel(), progress_every=0)
    records = runner.read_jsonl(path)
    assert len(records) == 3
    assert len({r["case_id"] for r in records}) == 3
    assert path.with_name(path.name + ".interrupted-tail").is_file()


def test_changed_parameters_cannot_silently_mix_runs(isolated_run):
    before = (isolated_run.directory / "config.json").read_bytes()
    with pytest.raises(ValueError, match="temperature changed"):
        runner.run_inference(replace(isolated_run, temperature=0.8), providers.ConstantModel())
    assert (isolated_run.directory / "config.json").read_bytes() == before


def test_changed_input_cannot_resume(isolated_run, monkeypatch):
    monkeypatch.setattr(runner, "build_items", lambda c: [tasks.TaskItem("verdict_facts", "new", "p", "g")])
    with pytest.raises(ValueError, match="task inputs changed"):
        runner.run_inference(isolated_run, providers.ConstantModel())


def test_judging_resume_rejects_changed_judge(isolated_run):
    runner.run_judging(isolated_run, HeuristicJudge(), progress_every=0)
    other = HeuristicJudge()
    other.name = "other-judge"
    with pytest.raises(ValueError, match="changed predictions or judge"):
        runner.run_judging(isolated_run, other)


def test_changed_heuristic_thresholds_cannot_mix_judgments(isolated_run):
    runner.run_judging(isolated_run, HeuristicJudge(), progress_every=0)
    with pytest.raises(ValueError, match="changed predictions or judge"):
        runner.run_judging(isolated_run, HeuristicJudge(correct_at=0.9))


@pytest.mark.parametrize("overrides", [{"limit": 0}, {"limit": -1}, {"split": "validation"},
                                       {"max_tokens": 0}, {"task": "unknown"}])
def test_invalid_development_configs_are_rejected(overrides):
    with pytest.raises(ValueError):
        runner.RunConfig(**({"task": "verdict_facts", "model": "constant"} | overrides))


def test_score_rejects_missing_and_duplicate_judgments(isolated_run):
    runner.run_judging(isolated_run, HeuristicJudge(), progress_every=0)
    path = isolated_run.directory / "judgments.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(lines[0])
    with pytest.raises(ValueError, match="cover exactly"):
        runner.score(isolated_run)
    path.write_text("".join(lines) + lines[0])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.score(isolated_run)


def test_score_rejects_empty_predictions(isolated_run):
    (isolated_run.directory / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="empty"):
        runner.score(isolated_run)


def test_malformed_middle_line_is_not_silently_skipped(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"case_id":"a"}\nBROKEN\n{"case_id":"b"}\n')
    with pytest.raises(ValueError, match="Invalid JSONL"):
        runner.completed_case_ids(path)
