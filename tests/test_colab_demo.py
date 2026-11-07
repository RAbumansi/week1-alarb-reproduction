"""CPU-safe checks for the real-model Colab demo; no GPU scores are fabricated."""
import ast
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from alarb import colab_demo as demo, tasks


@pytest.mark.parametrize(("text", "expected"), [
    ('[ANSWER]\n"B"\n[\\ANSWER]', "B"),
    ('A', "A"), ('Answer: c', "C"),
    ('A might work but B is another option.', None),
    ('reasoning mentions A and B </think> [ANSWER]D[/ANSWER]', "D"),
    ('<think>A?</think>[ANSWER]C[/ANSWER]', "C"),
    ('[ANSWER]A or B[/ANSWER]', None), ('', None),
])
def test_strict_choice_parser(text, expected):
    assert demo.strict_answer(text) == expected


def test_wilson_interval_and_empty_sample():
    lo, hi = demo.wilson_interval(80, 100)
    assert 0.71 < lo < 0.72 and 0.86 < hi < 0.87
    assert demo.wilson_interval(0, 0) is None


class FixtureBackend:
    metadata = {"backend": "fixture", "version": "test-only"}
    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after
    def predict(self, prompt, case_id):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("Simulated interruption")
        self.calls.append((prompt, case_id))
        return {"answer": "A", "truncated": False, "input_tokens": 2, "output_tokens": 1}


def items():
    return [tasks.TaskItem("mcq_same_statute", "id1", "facts and choices one", "A"),
            tasks.TaskItem("mcq_same_statute", "id2", "facts and choices two", "B")]


def test_exact_scoring_and_incomplete_status():
    records = [{"case_id": "x", "reference": "B", "answer": "B", "truncated": False},
               {"case_id": "y", "reference": "A", "answer": None, "truncated": True}]
    report = demo.score_records(records, 3, demo.DemoConfig(sample_size=3))
    assert report["accuracy"] == 0.5 and report["correct"] == 1
    assert report["status"] == "INCOMPLETE" and report["unparseable"] == 1
    assert report["truncated"] == 1 and not report["validation_accessed"]
    assert "NOT established" in report["interpretation"]


def test_resume_does_not_repeat_completed_model_calls(tmp_path):
    config = demo.DemoConfig(sample_size=2)
    backend = FixtureBackend()
    summary = demo.run_evaluation(items(), backend, tmp_path, config)
    assert summary["status"] == "COMPLETE" and summary["accuracy"] == 0.5
    again = FixtureBackend()
    demo.run_evaluation(items(), again, tmp_path, config)
    assert again.calls == []
    assert backend.calls == [(item.prompt, item.case_id) for item in items()]


def test_partial_progress_survives_model_failure(tmp_path):
    config = demo.DemoConfig(sample_size=2)
    with pytest.raises(RuntimeError, match="interruption"):
        demo.run_evaluation(items(), FixtureBackend(fail_after=1), tmp_path, config)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="Finish"):
        demo.export_evidence(tmp_path)
    backend = FixtureBackend()
    report = demo.run_evaluation(items(), backend, tmp_path, config)
    assert report["completed"] == 2 and len(backend.calls) == 1


def test_fixture_outputs_cannot_be_exported_as_experimental_evidence(tmp_path):
    demo.run_evaluation(items(), FixtureBackend(), tmp_path, demo.DemoConfig(sample_size=2))
    with pytest.raises(ValueError, match="Fixture"):
        demo.export_evidence(tmp_path)


def test_changed_settings_cannot_silently_resume(tmp_path):
    config = demo.DemoConfig(sample_size=2)
    demo.run_evaluation(items(), FixtureBackend(), tmp_path, config)
    with pytest.raises(ValueError, match="different inputs/settings"):
        demo.run_evaluation(items(), FixtureBackend(), tmp_path, replace(config, temperature=0.7))


def test_prediction_reference_is_never_supplied_to_backend(tmp_path):
    backend = FixtureBackend()
    demo.run_evaluation(items(), backend, tmp_path, demo.DemoConfig(sample_size=2))
    assert all(len(call) == 2 for call in backend.calls)
    assert backend.calls[0] == (items()[0].prompt, items()[0].case_id)


def test_only_official_training_file_is_requested(tmp_path, monkeypatch):
    source = tmp_path / "original.parquet"
    source.write_bytes(b"opaque bytes")
    requested = []
    hub = types.ModuleType("huggingface_hub")
    def download(repo_id, filename, **kwargs):
        requested.append((repo_id, filename, kwargs["revision"]))
        return str(source)
    hub.hf_hub_download = download
    datasets = types.ModuleType("datasets")
    datasets.Dataset = types.SimpleNamespace(from_parquet=lambda path, **kwargs: "training-only")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(demo, "TRAIN_SHA256", demo.sha256_file(source))
    assert demo.load_development(tmp_path) == "training-only"
    assert requested == [(demo.DATASET_ID, "data/train-00000-of-00001.parquet", demo.DATASET_REVISION)]


def test_duplicate_predictions_are_rejected():
    row = {"case_id": "x", "reference": "A", "answer": "A", "truncated": False}
    with pytest.raises(ValueError, match="Duplicate"):
        demo.score_records([row, row], 2, demo.DemoConfig(sample_size=2))


def test_notebook_cells_have_valid_python_and_no_fabricated_outputs():
    path = Path(__file__).resolve().parents[1] / "colab/ALARB_Free_GPU_Demo.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
