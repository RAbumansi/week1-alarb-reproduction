"""Free-GPU, modular Qwen3-8B article-identification demonstration.

This is a reduced development replication of Table 5, not an exact reproduction.
The original publisher test split is never downloaded or decoded by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import re
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from alarb import corpus, data, mcq, parsing, tasks

DATASET_ID = "THIQAH-RD/ALARB"
DATASET_REVISION = "e64bfdc867146294a65434c5ca16c2c4c5288ca2"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TRAIN_SHA256 = "16a642aeff40e9c8d050ca99a81f7a1802cdbee64234778d6c5790d20750ae32"
PAPER_ACCURACY = 0.846
PAPER_URL = "https://arxiv.org/html/2510.00694v1#S5.T5"


@dataclass(frozen=True)
class DemoConfig:
    sample_size: int = 100
    sample_seed: int = 42
    split_seed: int = 42
    max_new_tokens: int = 4096
    max_input_tokens: int = 6144
    thinking: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    quantization: str = "bitsandbytes NF4 double quantization float16 compute"

    def __post_init__(self):
        if self.sample_size < 1 or self.max_new_tokens < 1 or self.max_input_tokens < 1:
            raise ValueError("Sample and token limits must be positive.")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("Invalid sampling parameters.")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def cases_from_rows(rows) -> list[data.Case]:
    return [data.Case(data.case_id(r["case_facts"]), tuple(r["case_facts"]),
                      tuple(r["court_reasoning"]), tuple(r["applicable_laws"]), r["verdict"])
            for r in rows]


def prepare_development(source, split_seed: int = 42):
    """Reuse the existing course split algorithm on publisher TRAINING only."""
    if len(source) != 12012:
        raise ValueError("Expected the pinned 12,012-row publisher training file.")
    shuffled = source.shuffle(seed=split_seed)
    training = cases_from_rows(shuffled.select(range(9339)))
    testing = cases_from_rows(shuffled.select(range(9339, 12012)))
    built, stats = corpus.build_corpus(training + testing)
    train_ids = {case.case_id for case in training}
    test_ids = {case.case_id for case in testing}
    if len(train_ids) != 9339 or len(test_ids) != 2673 or train_ids & test_ids:
        raise ValueError("Course split has duplicate or overlapping case IDs.")
    return training, testing, built, stats


def load_development(cache_dir: Path):
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    # Request exactly one Parquet file: do not invoke a multi-split HF builder.
    path = Path(hf_hub_download(DATASET_ID, "data/train-00000-of-00001.parquet",
                               repo_type="dataset", revision=DATASET_REVISION,
                               cache_dir=str(cache_dir)))
    if sha256_file(path) != TRAIN_SHA256:
        raise ValueError("Official training-file SHA256 verification failed.")
    return Dataset.from_parquet(str(path), cache_dir=str(cache_dir / "prepared"))


def prepare_items(testing: list[data.Case], articles: corpus.Corpus, config: DemoConfig):
    items, skipped = mcq.build_mcqs("mcq_same_statute", testing, articles, seed=config.sample_seed)
    if config.sample_size > len(items):
        raise ValueError(f"Requested {config.sample_size} questions but only {len(items)} are eligible.")
    selected = random.Random(config.sample_seed).sample(
        sorted(items, key=lambda item: item.case_id), config.sample_size)
    by_id = {case.case_id: case for case in testing}
    rendered = [tasks.build_article_mcq(item.task, item.case_id, by_id[item.case_id].case_facts,
                                        item.choices, item.answer, item.metadata) for item in selected]
    return rendered, {"eligible_questions": len(items), "skipped": skipped,
                      "sample_case_ids": [item.case_id for item in rendered]}


def strict_answer(text: str) -> str | None:
    """Never score the first A-D mentioned in a model's discarded reasoning."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    cleaned = parsing.strip_thinking(text).strip()
    section = parsing.extract_section(cleaned, "ANSWER")
    if section is not None:
        return section.strip(" \n\t\"'") if re.fullmatch(r"[ABCD]", section.strip(" \n\t\"'")) else None
    match = re.fullmatch(r"(?:Answer\s*[:=]\s*)?[\"']?([ABCD])[\"']?[.。]?", cleaned, re.I)
    return match.group(1).upper() if match else None


def wilson_interval(correct: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def score_records(records: list[dict], requested: int, config: DemoConfig) -> dict:
    ids = [record["case_id"] for record in records]
    if len(set(ids)) != len(ids) or len(records) > requested:
        raise ValueError("Duplicate or excess predictions cannot be scored.")
    if any(r["reference"] not in tasks.CHOICE_LETTERS for r in records):
        raise ValueError("An MCQ reference is not one of A, B, C, D.")
    if any(r["answer"] not in (*tasks.CHOICE_LETTERS, None) for r in records):
        raise ValueError("Invalid predicted choice.")
    total = len(records)
    correct = sum(r["answer"] == r["reference"] and not r["truncated"] for r in records)
    rate = correct / total if total else None
    return {
        "status": "COMPLETE" if total == requested else "INCOMPLETE",
        "task": "mcq_same_statute", "model": config.model_id,
        "model_revision": config.model_revision, "requested": requested, "completed": total,
        "correct": correct, "accuracy": rate, "wilson_95_percent_interval": wilson_interval(correct, total),
        "unparseable": sum(r["answer"] is None for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "paper_model": "Qwen3-8B", "paper_table": "Table 5, Same Regulation",
        "paper_accuracy": PAPER_ACCURACY, "paper_questions": 1159, "paper_url": PAPER_URL,
        "difference_percentage_points": 100 * (rate - PAPER_ACCURACY) if rate is not None else None,
        "interpretation": "Reduced development task replication. Similarity to the paper is NOT established automatically by numeric proximity.",
        "deviations": ["Different development test cases; final validation remains sealed",
                       "693-article development-derived corpus rather than full statutory corpus",
                       "Independently reconstructed MCQ prompt and distractors",
                       "NF4 quantization and recorded generation settings; paper does not specify these"],
        "validation_accessed": False,
    }


class TransformersBackend:
    """Real, local-to-Colab inference; no API keys or paid inference endpoints."""
    def __init__(self, config: DemoConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU. In Colab select Runtime > Change runtime type > T4 GPU. Do not buy compute units.")
        memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if memory < 12:
            raise RuntimeError("This model configuration requires a GPU with at least 12 GiB VRAM.")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True,
                                          bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.model_revision, quantization_config=quantization,
            device_map={"": 0}, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
        self.model.eval()
        self.metadata = {"backend": "transformers", "gpu": torch.cuda.get_device_name(0),
                         "gpu_vram_gib": memory, "quantization": config.quantization,
                         "versions": {name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "bitsandbytes", "accelerate", "datasets")}}

    def predict(self, prompt: str, case_id: str) -> dict:
        import torch
        from transformers import set_seed
        config = self.config
        # Seed each case separately so resuming does not change its RNG sequence.
        set_seed((config.sample_seed + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)) % (2 ** 32))
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=config.thinking)
        inputs = self.tokenizer(chat, return_tensors="pt").to("cuda:0")
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens > config.max_input_tokens:
            raise RuntimeError("Prompt exceeds the frozen token limit; no input was silently truncated.")
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=True, temperature=config.temperature,
                                         top_p=config.top_p, top_k=config.top_k,
                                         max_new_tokens=config.max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, input_tokens:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        truncated = len(generated) >= config.max_new_tokens and int(generated[-1]) not in eos_ids
        final = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
        for special in self.tokenizer.all_special_tokens:
            final = final.replace(special, "")
        thinking_complete = not config.thinking or "</think>" in raw
        answer = strict_answer(final) if not truncated and thinking_complete else None
        return {"answer": answer, "truncated": truncated,
                "thinking_complete": thinking_complete, "input_tokens": int(input_tokens),
                "output_tokens": len(generated), "seconds": round(time.monotonic() - start, 3)}


def run_evaluation(items: list[tasks.TaskItem], backend, output: Path, config: DemoConfig,
                   preparation: dict | None = None) -> dict:
    if len(items) != config.sample_size or len({item.case_id for item in items}) != len(items):
        raise ValueError("Sample size or case uniqueness does not match the frozen config.")
    backend_config = getattr(backend, "config", None)
    if backend_config is not None:
        for key, value in asdict(config).items():
            if key != "sample_size" and getattr(backend_config, key) != value:
                raise ValueError(f"Backend configuration differs for {key}; reload with the frozen config.")
    signature = digest({"config": asdict(config), "inputs": [asdict(item) for item in items],
                        "backend": backend.metadata})
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Existing run has different inputs/settings. Choose a new output directory.")
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
        config_path.write_text(json.dumps({"signature": signature, "config": asdict(config),
            "backend": backend.metadata, "code_commit": commit, "preparation": preparation or {},
            "dataset_revision": DATASET_REVISION, "validation_accessed": False}, indent=2) + "\n")
    predictions = output / "predictions.jsonl"
    records = []
    if predictions.is_file():
        # Reuse the already tested crash-tail handling and duplicate detection.
        from alarb.runner import completed_case_ids, recover_tail, read_jsonl
        completed_case_ids(predictions)
        recover_tail(predictions)
        records = read_jsonl(predictions)
    expected = {item.case_id: item.reference for item in items}
    done = set()
    for record in records:
        if record["case_id"] not in expected or record["reference"] != expected[record["case_id"]]:
            raise ValueError("Existing prediction does not match the frozen sample.")
        done.add(record["case_id"])
    def checkpoint():
        summary = score_records(records, config.sample_size, config)
        summary["execution_backend"] = backend.metadata.get("backend")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    checkpoint()
    with predictions.open("a", encoding="utf-8") as stream:
        for item in items:
            if item.case_id in done:
                continue
            # The reference answer is deliberately NOT passed to the backend.
            completion = backend.predict(item.prompt, item.case_id)
            if completion["answer"] not in (*tasks.CHOICE_LETTERS, None):
                raise ValueError("Backend returned an invalid choice.")
            record = {"case_id": item.case_id, "reference": item.reference, **completion}
            if completion["truncated"]:
                record["answer"] = None
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            summary = checkpoint()
            print(f"{len(records)}/{len(items)} | answer={record['answer']} | "
                  f"correct={summary['correct']} | accuracy={summary['accuracy']:.1%}", flush=True)
    summary = checkpoint()
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "questions", "accuracy", "scope"])
        writer.writerow(["Parent paper Qwen3-8B Table 5", 1159, PAPER_ACCURACY, "publisher test; full article pool"])
        writer.writerow(["This run", len(records), summary["accuracy"], "development subset; reconstructed article pool; NF4"])
    return summary


def export_evidence(output: Path) -> Path:
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "COMPLETE":
        raise ValueError("Finish the experiment before exporting submission evidence.")
    if summary.get("execution_backend") != "transformers":
        raise ValueError("Fixture outputs cannot be exported as real-model submission evidence.")
    destination = output / "ALARB_result_evidence.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("summary.json", "comparison.csv", "config.json", "predictions.jsonl"):
            archive.write(output / name, name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-model", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs/colab-paper-demo"))
    args = parser.parse_args()
    config = DemoConfig(sample_size=args.limit)
    source = load_development(Path(".cache/huggingface"))
    training, testing, articles, stats = prepare_development(source, config.split_seed)
    items, preparation = prepare_items(testing, articles, config)
    print(f"Prepared {len(items)} MCQs; train={len(training)}, test={len(testing)}, articles={len(articles.articles)}")
    print("Final validation was not downloaded or accessed.")
    if args.run_model:
        summary = run_evaluation(items, TransformersBackend(config), args.output, config, preparation)
        print(json.dumps(summary, indent=2))
        print(f"Evidence: {export_evidence(args.output)}")


if __name__ == "__main__":
    main()
