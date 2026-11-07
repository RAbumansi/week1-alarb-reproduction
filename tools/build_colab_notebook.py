"""Build an unexecuted Colab notebook from reviewable Python source cells."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})


def code(text):
    source = textwrap.dedent(text).strip() + "\n"
    ast.parse(source)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})


md("""
# ALARB parent paper demonstration on a free GPU

This notebook runs a **real Qwen3-8B model** on reconstructed Arabic legal-article
multiple-choice questions. No paid API or API key is required.

Before running: choose **Runtime > Change runtime type > T4 GPU** (or another
available free CUDA GPU). Free GPUs are subject to availability. Do not purchase
a plan or compute units for this notebook. Save a copy to your Drive if desired.

The parent paper reports **84.60% on 1,159 same-regulation questions** for Qwen3-8B
(Table 5). This notebook uses a fixed **100-question development sample**, a
reconstructed article corpus and NF4 quantization. Its measured score is displayed
only after inference; numeric similarity must not be assumed beforehand.
The final validation dataset is never downloaded or opened.

Sources: [ALARB paper](https://arxiv.org/html/2510.00694v1#S5.T5),
[official dataset](https://huggingface.co/datasets/THIQAH-RD/ALARB),
[official Qwen3 model](https://huggingface.co/Qwen/Qwen3-8B).
""")
md("## 1 Check the free coding platform\nThis should show a CUDA GPU before downloading model weights.")
code("""
import sys
import torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > T4 GPU, then rerun. Free GPU availability is not guaranteed.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
""")
md("## 2 Install the recorded dependency versions\nRun once in a fresh runtime. These are reproducibility pins, not a claim that the paper used these versions.")
code("""
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.57.1", "accelerate==1.10.1",
                "bitsandbytes==0.48.1", "datasets==4.0.0"], check=True)
""")
md("## 3 Load our public modular code\nThe release tag fixes the source version. This does not download raw case data from GitHub.")
code("""
from pathlib import Path
import os

REPOSITORY = "https://github.com/RAbumansi/week1-alarb-reproduction.git"
CODE_REF = "alarb-colab-demo-v1"
PROJECT = Path("/content/alarb-project")
if not PROJECT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", CODE_REF, REPOSITORY, str(PROJECT)], check=True)
else:
    expected = subprocess.check_output(["git", "rev-parse", CODE_REF], cwd=PROJECT, text=True).strip()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if actual != expected:
        raise RuntimeError("Existing project has a different revision. Use a fresh runtime; do not overwrite your edits.")
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
print("Source commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
""")
md("## 4 Freeze the experiment before seeing scores\nThinking is enabled as in the paper. Sampling settings follow the Qwen model guidance; the paper does not specify its decoding settings. Do not change the seed to seek a better score.")
code("""
import json
from dataclasses import asdict, replace
from alarb.colab_demo import (DemoConfig, load_development, prepare_development,
                              prepare_items, TransformersBackend, run_evaluation,
                              export_evidence, PAPER_ACCURACY)
CONFIG = DemoConfig(sample_size=100, sample_seed=42, split_seed=42)
RESULTS = Path("/content/alarb-results")
print(json.dumps(asdict(CONFIG), indent=2))
""")
md("## 5 Verify access and prepare the questions\nOnly the publisher's 12,012-row training file is requested. We recover the course training/development-test split from it. MCQ references are used to construct and score the questions, but are never passed as answer labels to the model.")
code("""
source = load_development(Path("/content/alarb-source-cache"))
training, testing, articles, corpus_stats = prepare_development(source, CONFIG.split_seed)
items, preparation = prepare_items(testing, articles, CONFIG)
print("Verified publisher training rows:", len(source))
print("Course training:", len(training), "| Development test:", len(testing))
print("Reconstructed article pool:", len(articles.articles))
print("Eligible MCQs:", preparation["eligible_questions"], "| Frozen sample:", len(items))
print("Final validation: NOT downloaded or opened")
""")
md("## 6 Load the real model in the cloud\nThis step downloads model weights to the Colab VM, not your Mac. It can take several minutes. NF4 quantization reduces GPU memory use and is an explicit departure from the paper's unspecified precision.")
code("""
if "backend" not in globals():
    backend = TransformersBackend(CONFIG)
elif backend.config != CONFIG:
    raise RuntimeError("Config changed after model loading. Restart the session and freeze settings again.")
print(json.dumps(backend.metadata, indent=2))
""")
md("## 7 Verify a small real-model run\nThese three questions are part of the frozen sample; they are not an independent validation set. The output shows letters and aggregate scores, not personal case text.")
code("""
smoke_config = replace(CONFIG, sample_size=3)
smoke = run_evaluation(items[:3], backend, RESULTS / "smoke-three", smoke_config,
                       {"scope": "first three questions of the frozen sample"})
print(json.dumps(smoke, indent=2))
if smoke["unparseable"] == 3:
    raise RuntimeError("All three outputs failed parsing. Inspect truncation before starting the larger run; do not report these as a successful reproduction.")
""")
md("## 8 Run the frozen evaluation\nThis is real inference and can take a substantial part of a free GPU session. Progress is saved after each answer. Rerunning this cell resumes completed questions only if settings and inputs match. Colab files are temporary; download evidence after completion. No paid upgrade is needed or selected by this notebook.")
code("""
summary = run_evaluation(items, backend, RESULTS / "same-statute-100", CONFIG, preparation)
print(json.dumps(summary, indent=2))
""")
md("## 9 Compare the measured result with Table 5\nThe difference and confidence interval describe this development sample. They do not remove the differences in cases, article corpus, reconstructed questions or quantization. Discuss these with the instructor before claiming that the assignment's similarity requirement has been met.")
code("""
import pandas as pd
comparison = pd.DataFrame([
    {"Experiment": "Parent paper Qwen3-8B", "Questions": 1159, "Accuracy percent": 100*PAPER_ACCURACY,
     "Protocol": "Original test cases and full article corpus"},
    {"Experiment": "Our recorded development run", "Questions": summary["completed"],
     "Accuracy percent": 100*summary["accuracy"], "Protocol": "Development sample; reconstructed corpus; NF4"},
])
display(comparison)
print("Our 95% Wilson interval:", [round(100*x, 2) for x in summary["wilson_95_percent_interval"]])
print("Difference in percentage points:", round(summary["difference_percentage_points"], 2))
print("Status:", summary["status"])
print(summary["interpretation"])
""")
md("## 10 Download evidence for the recording and submission\nThe evidence contains configurations, case hashes, answer letters, counts and scores. It excludes raw facts, party names and generated reasoning. Save the executed notebook too; do not present an unexecuted notebook as experimental evidence.")
code("""
from google.colab import files
evidence = export_evidence(RESULTS / "same-statute-100")
files.download(str(evidence))
""")
md("""
## Zoom recording checklist

1. Show the public repository and this notebook's source commit.
2. Show the Python version, assigned GPU and successful dataset-access checks.
3. Explain the reusable modules for data preparation, prompts, inference and scoring.
4. Show an actual completed model run and the two-row comparison with Table 5.
5. State the measured score and the protocol differences plainly. Tests or the
   25% random baseline alone do not demonstrate reproduction of the paper.
6. Show the saved evidence file. Briefly explain the planned agentic variation.

Record your own narration in Zoom after reviewing the actual results. This
notebook does not create a recording or certify that the paper's result was matched.
""")

notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU", "colab": {"name": "ALARB_Free_GPU_Demo.ipynb", "provenance": []}}}
for index, cell in enumerate(cells):
    cell["id"] = f"alarb-cell-{index:02d}"
destination = ROOT / "colab/ALARB_Free_GPU_Demo.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
