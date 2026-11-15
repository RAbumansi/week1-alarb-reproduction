"""Week 11 experiment on a free Colab GPU: do retrieved articles help verdicts?

Four conditions on the same development-test cases, with one model and one judge:

  verdict_facts                facts only (paper task 1)
  verdict_rag__popularity      facts + the most-cited training articles (a control:
                               legal text that ignores the case)
  verdict_rag__hybrid-tfidf    facts + articles from the training-only retriever
  verdict_laws                 facts + the court's own cited articles (paper task 2,
                               the ceiling no retriever can beat)

The three article conditions share the paper's Figure 10 prompt word for word, so
differences between them come from which articles the model was given.

Generation is batched so the run fits a free session. Nothing is written to Google
Drive: records stay on the temporary Colab disk and the notebook prints a summary
holding only case hashes, labels and counts, never case text or party names.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from alarb import comparison, outcomes, parsing, prompts, runner, tasks
from alarb.data import Case
from alarb.judge import LABELS, UNPARSEABLE

CONDITIONS = (
    ("verdict_facts", None),
    ("verdict_rag", "popularity"),
    ("verdict_rag", "hybrid-tfidf"),
    ("verdict_laws", None),
)
RESULT_BEGIN = "WEEK11_RESULT_BEGIN"
RESULT_END = "WEEK11_RESULT_END"
CONDITION_BEGIN = "WEEK11_CONDITION_BEGIN"
CONDITION_END = "WEEK11_CONDITION_END"


def condition_id(task: str, retriever: str | None) -> str:
    return f"{task}__{retriever}" if retriever else task


CONDITION_IDS = tuple(condition_id(task, retriever) for task, retriever in CONDITIONS)


@dataclass(frozen=True)
class Week11Config:
    cases: int = 20
    seed: int = 42
    split_seed: int = 42
    model_id: str = "Qwen/Qwen3-8B"
    model_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    quantization: str = "bitsandbytes NF4 double quantization float16 compute"
    thinking: bool = True
    max_new_tokens: int = 2048
    judge_thinking: bool = False
    judge_max_new_tokens: int = 1024
    batch_size: int = 10
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    def __post_init__(self) -> None:
        if min(self.cases, self.max_new_tokens, self.judge_max_new_tokens, self.batch_size) < 1:
            raise ValueError("Case count, token limits and batch size must be positive.")


def select_cases(testing: list[Case], n: int, seed: int) -> list[Case]:
    """Cases citing at least one article, so all four conditions share them."""
    eligible = [case for case in testing if case.applicable_laws]
    return runner.stratified_sample(eligible, n, seed)


def build_conditions(training: list[Case], cases: list[Case]) -> tuple[dict, dict]:
    from alarb.retrieval import Retrieval

    retrieval = Retrieval(training)
    built, recall = {}, {}
    for task, retriever in CONDITIONS:
        if task == "verdict_rag":
            hits = retrieval.retrieve_batch(retriever, [case.case_facts for case in cases], tasks.RAG_TOP_K)
            items = [
                tasks.build_verdict_rag(case, [(h.key, h.text) for h in found], retriever)
                for case, found in zip(cases, hits)
            ]
            per_case = []
            for case, found in zip(cases, hits):
                positions, total = retrieval.gold(case)
                got = {retrieval.position[h.key] for h in found}
                per_case.append(len(positions & got) / total if total else 0.0)
            recall[retriever] = round(sum(per_case) / len(per_case), 4)
        else:
            items = tasks.build_verdict_items(task, cases)
        if len(items) != len(cases) or any(item is None for item in items):
            raise ValueError(f"{task} could not build an item for every selected case.")
        built[condition_id(task, retriever)] = items
    return built, {"recall@5_on_selected_cases": recall, "index_articles": len(retrieval.articles)}


class BatchedBackend:
    """Qwen3-8B in 4-bit on a free CUDA GPU, generating several prompts at once."""

    def __init__(self, config: Week11Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.model_revision, quantization_config=quant,
            device_map={"": 0}, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self.metadata = {
            "backend": "transformers, batched",
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "bitsandbytes", "accelerate")},
        }

    def _generate(self, prompts_: list[str], thinking: bool, max_new_tokens: int, seed: int) -> list[dict]:
        import torch
        from transformers import set_seed

        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for p in prompts_
        ]
        enc = self.tokenizer(chats, return_tensors="pt", padding=True).to("cuda:0")
        width = enc["input_ids"].shape[1]
        set_seed(seed)
        start = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, do_sample=True, temperature=self.config.temperature, top_p=self.config.top_p,
                top_k=self.config.top_k, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        seconds = round(time.monotonic() - start, 1)
        results = []
        for row in range(len(prompts_)):
            generated = out[row, width:].tolist()
            ended = next((i for i, tok in enumerate(generated) if tok in self.eos_ids), None)
            kept = generated if ended is None else generated[:ended]
            raw = self.tokenizer.decode(kept, skip_special_tokens=False)
            final = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
            for special in self.tokenizer.all_special_tokens:
                final = final.replace(special, "")
            results.append({
                "final": final.strip(),
                "truncated": ended is None,
                "thinking_complete": (not thinking) or "</think>" in raw,
                "input_tokens": int(enc["attention_mask"][row].sum()),
                "output_tokens": len(kept),
                "batch_seconds": seconds,
            })
        return results

    def generate(self, prompts_: list[str], thinking: bool, max_new_tokens: int, seed: int) -> list[dict]:
        """Generate, halving the batch whenever the GPU runs out of memory."""
        import torch

        try:
            return self._generate(prompts_, thinking, max_new_tokens, seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts_) == 1:
                raise
            half = len(prompts_) // 2
            return (self.generate(prompts_[:half], thinking, max_new_tokens, seed)
                    + self.generate(prompts_[half:], thinking, max_new_tokens, seed + 1))


def _seed(config: Week11Config, label: str, batch: int) -> int:
    return (config.seed + int(hashlib.sha256(f"{label}:{batch}".encode()).hexdigest()[:8], 16)) % 2**32


def _done(path: Path) -> dict:
    return {r["case_id"]: r for r in runner.read_jsonl(path)} if path.is_file() else {}


def run_condition(items: list[tasks.TaskItem], backend, directory: Path, config: Week11Config,
                  label: str, retriever: str | None) -> list[dict]:
    """Generate one condition's verdicts. Rerunning resumes; finished cases are kept."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(
        {"task": items[0].task, "retriever": retriever, "week11": asdict(config)}, indent=2) + "\n")
    path = directory / "predictions.jsonl"
    done = _done(path)
    pending = sorted((i for i in items if i.case_id not in done), key=lambda i: len(i.prompt))
    with path.open("a", encoding="utf-8") as stream:
        for batch_no, start in enumerate(range(0, len(pending), config.batch_size)):
            batch = pending[start:start + config.batch_size]
            outs = backend.generate([i.prompt for i in batch], config.thinking, config.max_new_tokens,
                                    _seed(config, label, batch_no))
            for item, out in zip(batch, outs):
                verdict = parsing.parse_verdict(out["final"]) if not out["truncated"] else parsing.ParsedVerdict(None, None)
                record = {
                    "case_id": item.case_id, "task": item.task, "reference": item.reference,
                    "parsed": {"verdict": verdict.verdict, "reasoning": verdict.reasoning},
                    "parsed_ok": verdict.ok, "raw": out["final"], "model": config.model_id,
                    "cached": False, "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
                    "truncated": out["truncated"], "batch_seconds": out["batch_seconds"],
                    "metadata": item.metadata,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                done[item.case_id] = record
            print(f"  {label}: {len(done)}/{len(items)} verdicts", flush=True)
    return [done[i.case_id] for i in items]


def judge_label(config: Week11Config) -> str:
    return f"llm:{config.model_id} (thinking {'on' if config.judge_thinking else 'off'})"


def judge_condition(directory: Path, backend, config: Week11Config, label: str) -> list[dict]:
    """Grade with the paper's Appendix B.1 prompt. Unparseable verdicts are not sent."""
    records = runner.read_jsonl(directory / "predictions.jsonl")
    path = directory / "judgments.jsonl"
    done = _done(path)
    judge_name = judge_label(config)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            if record["case_id"] not in done and not record["parsed_ok"]:
                row = {"case_id": record["case_id"], "label": UNPARSEABLE, "judge": judge_name, "raw": ""}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[record["case_id"]] = row
        pending = [r for r in records if r["case_id"] not in done]
        for batch_no, start in enumerate(range(0, len(pending), config.batch_size)):
            batch = pending[start:start + config.batch_size]
            judge_prompts = [prompts.JUDGE.format(judge_verdict=r["reference"], predicted_verdict=r["parsed"]["verdict"])
                             for r in batch]
            outs = backend.generate(judge_prompts, config.judge_thinking, config.judge_max_new_tokens,
                                    _seed(config, "judge:" + label, batch_no))
            for record, out in zip(batch, outs):
                parsed = parsing.parse_judgement(out["final"]) if not out["truncated"] else None
                row = {"case_id": record["case_id"], "label": parsed or UNPARSEABLE, "judge": judge_name,
                       "raw": out["final"]}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                done[record["case_id"]] = row
        print(f"  {label}: {len(done)}/{len(records)} judged", flush=True)
    return [done[r["case_id"]] for r in records]


def compact(directory: Path) -> list[dict]:
    """One condition's results as case hashes, labels, outcome categories and
    counts. No case text, so it can be printed and kept."""
    preds = runner.read_jsonl(directory / "predictions.jsonl")
    judged = {r["case_id"]: r["label"] for r in runner.read_jsonl(directory / "judgments.jsonl")}

    def category(text: str) -> str:
        return outcomes.PAPER_CATEGORY[outcomes.classify_verdict(text)]

    return [{
        "case_id": r["case_id"],
        "label": judged[r["case_id"]],
        "gold_outcome": category(r["reference"]),
        "predicted_outcome": category(r["parsed"]["verdict"] or "") if r["parsed_ok"] else None,
        "parsed_ok": r["parsed_ok"],
        "truncated": r["truncated"],
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
    } for r in preds]


def table(results: dict[str, list[dict]], baseline: str = "verdict_facts") -> dict:
    """The comparison table from compact results: the same cases, paired
    against the baseline, with a sign test."""
    ids = sorted(set.intersection(*({r["case_id"] for r in rows} for rows in results.values())))
    if not ids:
        raise ValueError("The conditions share no cases.")
    by = {cid: {r["case_id"]: r for r in rows} for cid, rows in results.items()}
    out = []
    for cid in [baseline] + [c for c in results if c != baseline]:
        rows = [by[cid][i] for i in ids]
        labels = [r["label"] for r in rows]
        good = sum(label in ("CORRECT", "PARTIALLY CORRECT") for label in labels)
        scored = [r for r in rows if r["gold_outcome"] != outcomes.UNCLASSIFIED]
        agree = sum(r["predicted_outcome"] == r["gold_outcome"] for r in scored)
        task, _, retriever = cid.partition("__")
        row = {
            "run_id": cid,
            "condition": comparison.describe(task, retriever or None),
            "cases": len(ids),
            "labels": {label: round(labels.count(label) / len(ids), 4) for label in (*LABELS, UNPARSEABLE)},
            "correct_or_partial": round(good / len(ids), 4),
            "correct_or_partial_95ci": comparison.wilson(good, len(ids)),
            "mean_grade": round(sum(comparison.GRADE[label] for label in labels) / len(ids), 4),
            "outcome_agreement": {"count": agree, "of": len(scored),
                                  "share": round(agree / len(scored), 4) if scored else None},
            "truncated": sum(r["truncated"] for r in rows),
            "unparseable": sum(not r["parsed_ok"] for r in rows),
            "mean_input_tokens": round(sum(r["input_tokens"] for r in rows) / len(rows)),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in rows) / len(rows)),
        }
        if cid != baseline:
            deltas = [comparison.GRADE[by[cid][i]["label"]] - comparison.GRADE[by[baseline][i]["label"]] for i in ids]
            better, worse = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
            row["vs_baseline"] = {"better": better, "worse": worse, "same": len(ids) - better - worse,
                                  "sign_test_p": comparison.sign_test(better, worse)}
        out.append(row)
    return {"baseline": baseline, "shared_cases": len(ids),
            "grade_scale": "CORRECT 2, PARTIALLY CORRECT 1, INCORRECT or unparseable 0", "rows": out}


def print_condition(cid: str, rows: list[dict]) -> None:
    print(CONDITION_BEGIN)
    print(json.dumps({"condition": cid, "results": rows}, separators=(",", ":")))
    print(CONDITION_END)


def summarise(results: dict[str, list[dict]], config: Week11Config, preparation: dict, metadata: dict,
              judge: str) -> dict:
    """Everything the paper and recording need, without any case text."""
    return {
        "experiment": "ALARB week 11: retrieval-grounded verdict prediction",
        "config": asdict(config), "backend": metadata, "preparation": preparation,
        "judge": judge, "comparison": table(results), "per_condition": results,
        "validation_accessed": False,
    }


def parse_printed(text: str) -> dict[str, list[dict]]:
    """Recover per-condition results from notebook output, so conditions that
    finished before a disconnect are not lost."""
    found = {}
    for chunk in text.split(CONDITION_BEGIN)[1:]:
        payload = json.loads(chunk.split(CONDITION_END)[0].strip())
        found[payload["condition"]] = payload["results"]
    return found


def print_result(summary: dict) -> None:
    print(RESULT_BEGIN)
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    print(RESULT_END)
