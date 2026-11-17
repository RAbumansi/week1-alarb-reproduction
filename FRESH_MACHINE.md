# ALARB fresh-computer run and recording

Start with the public GitHub repository. The one-command CPU workflow uses a
fresh Python environment and regenerates the reported retrieval metrics from
the official release. It also recalculates the prior GPU pilot's accuracy
from its saved prediction records.

## Recording steps

1. Use a computer that none of you used to run the project. Sign in to PSU Zoom.
2. Start recording, share the screen and turn on the webcams. All teammates
   should appear on webcam and each should explain part of the implementation.
3. With the recording running, open the public repository, click **Code →
   Download ZIP**, and extract the archive. Open a terminal in the extracted
   folder and show the README.
4. On Windows run `py -3.13 run_project.py`. On macOS or Linux run
   `python3.13 run_project.py`. Follow the setup in the README if Python 3.13 is
   missing. Before recording day, confirm a lab machine has access to Python,
   the package index, and Hugging Face without running the project on it.
5. After the run, show the terminal output and open
   `runs/fresh-machine/results.json`. Compare the values to your manuscript.
6. If the paper reports 75% for Qwen3-8B, explain the difference between
   rescoring those archived predictions and generating new predictions.
   For a demonstration of fresh Qwen3 inference, run the separate pilot notebook
   on an available CUDA runtime and show its new measured output.

### Expected values

| Output | Expected | How the command gets it |
|---|---:|---|
| Course training cases | 9,339 | Splits the pinned publisher training file with seed 42 |
| Course test cases | 2,673 | Remaining cases in that shuffled file |
| Retrieval experiment | 50 cases and 654 training articles | Fits TF-IDF and ranks articles on the CPU |
| Retrieval macro recall@5 | 56.1978% | Scores the retrieved articles on 49 cases with gold articles |
| Qwen3-8B pilot accuracy | 15/20 (75.00%) | Rebuilds the pilot inputs and scores the archived model responses |
| Pilot unparseable and truncated responses | 3 and 0 | Counts all saved predictions |
| Parent paper same-regulation accuracy | 84.60% | Published reference, recorded as context |

The retrieval experiment uses the existing abstention fixture for the drafting
stage, so the 50 fixture abstentions measure pipeline behavior. The archived
pilot was measured with Qwen3-8B on a Colab Tesla T4. Its evaluation used a
smaller development sample, reconstructed questions and NF4 quantization.
See [the measured pilot report](COLAB_PILOT_RESULTS.md) for more detail.

### What each teammate can explain

- `run_project.py` creates the Python environment and installs the pinned CPU packages.
- `alarb/reproduce.py` checks the dataset checksum, rebuilds the two development
  splits and checks that the pilot question IDs and answer keys match the archive.
- `alarb/agentic.py` fits the training-only TF-IDF index, retrieves articles and
  calculates recall on the development test cases.
- `alarb/colab_demo.py` prepares the pilot questions and scores the model choices.

## Setup and troubleshooting

Use 64-bit CPython 3.13 on Windows 10/11 x64, macOS 14+ on Intel or Apple Silicon,
or a Linux distribution with glibc 2.28+ on x86-64 or ARM64. Use a writable
folder and allow at least 1 GB of disk space for dependencies, cache, and
temporary data. Use the x64 Python installer on ARM Windows when x64 emulation
is available. Use your campus software center or a lab administrator if the
machine restricts installing Python. On Linux, install the corresponding
Python `venv` package if `ensurepip` is missing.

- **Python command missing:** Install 64-bit Python 3.13 or use its full path.
  On Windows with Python 3.13 installed but no `py` launcher, try
  `python --version` and then `python run_project.py` if it shows Python 3.13.
- **Dataset access blocked:** Use a verified copy of the publisher's original
  training Parquet file. Supply `--train-parquet "/path/to/train-00000-of-00001.parquet"`
  after the normal command. The SHA256 check still runs. The published href is
  [the pinned training download](https://huggingface.co/datasets/THIQAH-RD/ALARB/resolve/e64bfdc867146294a65434c5ca16c2c4c5288ca2/data/train-00000-of-00001.parquet).
- **Interrupted run:** Rerun the same command. The launcher reuses installed
  packages and checksum-verified downloads and rebuilds the experiment from
  scratch. If the cache file was damaged, remove the file at the
  reported path under `.cache/reproduction/` before retrying.
- **Lab computer cannot reach PyPI:** Ask the lab staff which mirror or setup
  the machine allows, or use another fresh machine with network access.
- **GPU quota unavailable:** The CPU experiment and archived score check still
  run. The GPU experiment requires a CUDA runtime with at least 12 GiB of VRAM.

## Changes in this update

| Change | Reason |
|---|---|
| Pulled GitHub `main` at `c49ea77` | The fixes start from the latest shared code and preserve the README edits there |
| Added `run_project.py` and `requirements-reproduction.txt` | The same Python launcher works with Windows and Unix virtual-environment layouts and pins CPU-only dependencies |
| Added `alarb/reproduce.py` | Rebuilds the development splits from a verified training file, runs retrieval, matches archived pilot IDs and answer keys, and checks both sets of metrics |
| Use temporary experiment files and a single JSON report | Runs can repeat from the same download and the generated output contains aggregate metrics and provenance |
| Corrected the supported Python range in `run_week1.sh` | The legacy audit now checks Python versions against its NumPy and datasets pins before installation |
| Updated the generated Colab notebooks and their builder | Per-release clone directories avoid collisions, and the resume check resolves an annotated tag to its commit |
| Expanded GitHub Actions | Windows, macOS and Ubuntu jobs run the README command on clean runners and save the results |

The reproduction path downloads only publisher training data. It leaves the
publisher test file, which serves as the project's final validation holdout,
outside the workflow.
