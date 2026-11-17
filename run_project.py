"""Set up the CPU reproduction and run it from a fresh GitHub download."""

import argparse
import os
import struct
import subprocess
import sys
import venv
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def environment_python() -> Path:
    folder = ROOT / ".venv-reproduction"
    executable = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        print("Creating the project Python environment", flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, help="Use a copy of the official training Parquet file.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "fresh-machine" / "results.json",
                        help="File to write the aggregate results to.")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8 or sys.implementation.name != "cpython":
        parser.exit(2, "Use 64-bit CPython 3.13. On Windows run: py -3.13 run_project.py\n"
                       "On macOS or Linux run: python3.13 run_project.py\n")
    if find_spec("ensurepip") is None:
        parser.exit(2, "Python's venv package is required. On Debian or Ubuntu install python3.13-venv.\n")

    python = environment_python()
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                   "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT)}
    command = [str(python), "-X", "utf8"]
    install = command + ["-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
                         "-r", str(ROOT / "requirements-reproduction.txt")]
    print("Checking pinned CPU dependencies", flush=True)
    try:
        subprocess.run(install, cwd=ROOT, env=environment, check=True)
    except subprocess.CalledProcessError:
        parser.exit(1, "Dependency installation failed. Check the requirements and network access in the README.\n")

    run = command + ["-m", "alarb.reproduce", "--output", str(args.output.resolve())]
    if args.train_parquet is not None:
        run += ["--train-parquet", str(args.train_parquet.resolve())]
    return subprocess.run(run, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
