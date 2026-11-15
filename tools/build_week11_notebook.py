"""Build the unexecuted week 11 Colab notebook from reviewable source cells."""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_REF = "week11-retrieval-novelty"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})


def code(text):
    source = textwrap.dedent(text).strip() + "\n"
    ast.parse(source)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})


md("""
# ALARB week 11: do retrieved articles help verdict prediction?

ALARB's authors tried retrieving statute articles from case facts, found it
"extremely poor", and gave the model the court's own articles instead (their task 2).
A real assistant never has those. This notebook tests the realistic setting: Qwen3-8B
predicts verdicts for the same 20 development-test cases under four conditions.

| Condition | Articles in the prompt |
|---|---|
| Facts only (paper task 1) | none |
| Most-cited articles (control) | the 5 most-cited training articles, the same for every case |
| Retrieved articles (our method) | top 5 from a retriever indexed on training cases only |
| Court-cited articles (paper task 2) | the articles the court actually cited (ceiling) |

The three article conditions use the paper's Figure 10 prompt word for word.
Every verdict is graded with the paper's Appendix B.1 judge prompt.

**Before running:** Runtime > Change runtime type > **T4 GPU**. No paid plan or API key
is needed. Nothing is saved to Google Drive; the final cell prints the results.
The final validation holdout is never downloaded.
""")
md("## 1 Check the GPU")
code("""
import sys, torch
print("Python:", sys.version.split()[0], "| PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > T4 GPU, then run again.")
print("GPU:", torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GiB")
""")
md("## 2 Install pinned libraries")
code("""
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1",
                "accelerate==1.10.1", "bitsandbytes==0.48.1", "datasets==4.0.0"], check=True)
""")
md("## 3 Load our code from GitHub")
code(f"""
from pathlib import Path
import os
PROJECT = Path("/content/alarb-project")
if not PROJECT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "{CODE_REF}",
                    "https://github.com/RAbumansi/week1-alarb-reproduction.git", str(PROJECT)], check=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Code commit:", COMMIT)
""")
md("## 4 Freeze the settings\nSampling follows Qwen's guidance for thinking mode; the paper does not state its decoding settings. Do not change the seed after seeing results.")
code("""
import json
from dataclasses import asdict
from alarb import colab_week11 as w
from alarb.colab_demo import load_development, prepare_development
CONFIG = w.Week11Config(cases=20, seed=42, batch_size=6)
RESULTS = Path("/content/week11-results")
print(json.dumps(asdict(CONFIG), indent=2))
""")
md("## 5 Prepare the cases and the four prompts\nOnly the publisher's training file is downloaded (its SHA-256 is checked), and the course split is rebuilt from it. The retriever's index uses the 9,339 training cases; the 20 test cases' own citations are read only to build the ceiling condition and to measure recall.")
code("""
source = load_development(Path("/content/alarb-source-cache"))
training, testing, _, _ = prepare_development(source, CONFIG.split_seed)
cases = w.select_cases(testing, CONFIG.cases, CONFIG.seed)
conditions, preparation = w.build_conditions(training, cases)
preparation.update({"training_cases": len(training), "development_test_cases": len(testing),
                    "selected_case_ids": [c.case_id for c in cases], "code_commit": COMMIT})
print("Training:", len(training), "| Development test:", len(testing), "| Selected:", len(cases))
print("Retrieval recall@5 on these cases:", preparation["recall@5_on_selected_cases"])
print("Final validation: not downloaded")
""")
md("## 6 Load Qwen3-8B on the GPU\nThe weights download to this Colab machine (several minutes), not to your computer.")
code("""
if "backend" not in globals():
    backend = w.BatchedBackend(CONFIG)
print(json.dumps(backend.metadata, indent=2))
""")
md("## 7 Generate verdicts for all four conditions\nThis is the long step. Progress prints after each batch; if the session drops, run this cell again and it resumes.")
code("""
for (task, retriever), cid in zip(w.CONDITIONS, w.CONDITION_IDS):
    w.run_condition(conditions[cid], backend, RESULTS / "runs" / cid, CONFIG, cid, retriever)
""")
md("## 8 Grade every verdict with the paper's judge prompt")
code("""
for cid in w.CONDITION_IDS:
    w.judge_condition(RESULTS / "runs" / cid, backend, CONFIG, cid)
""")
md("## 9 Results\nCase hashes, labels and counts only; no case text or names.")
code("""
import pandas as pd
summary = w.summarise(RESULTS, CONFIG, preparation, backend.metadata)
rows = summary["comparison"]["rows"]
display(pd.DataFrame([{
    "Condition": r["condition"], "Cases": r["cases"],
    "Correct %": round(100 * r["labels"]["CORRECT"]), "Partially %": round(100 * r["labels"]["PARTIALLY CORRECT"]),
    "Incorrect %": round(100 * (r["labels"]["INCORRECT"] + r["labels"]["UNPARSEABLE"])),
    "Correct or partial %": round(100 * r["correct_or_partial"]),
    "Outcome agreement %": None if r["outcome_agreement"]["share"] is None else round(100 * r["outcome_agreement"]["share"]),
    "Better / worse vs facts": "" if "vs_baseline" not in r else f'{r["vs_baseline"]["better"]} / {r["vs_baseline"]["worse"]} (p={r["vs_baseline"]["sign_test_p"]})',
} for r in rows]))
print(json.dumps(summary["generation"], indent=1))
w.print_result(summary)
""")

notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU",
                         "colab": {"name": "ALARB_Week11_Retrieval_Novelty.ipynb", "provenance": [], "gpuType": "T4"}}}
for index, cell in enumerate(cells):
    cell["id"] = f"week11-cell-{index:02d}"
destination = ROOT / "colab/ALARB_Week11_Retrieval_Novelty.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
