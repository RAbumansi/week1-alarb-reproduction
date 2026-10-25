"""Collect the pinned publisher files without decoding the validation holdout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "SOURCE_LOCK.json"
DEFAULT_OUTPUT = Path("data/collected/ALARB_dataset.zip")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_file(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"Publisher file integrity check failed: {path.name}")


def _copy_or_download(name: str, spec: dict, destination: Path, source: dict,
                      local_dir: Path | None, offline: bool) -> None:
    candidates = []
    if local_dir is not None:
        candidates.append(local_dir / name)
    candidates.append(ROOT / ".cache/huggingface/hub/datasets--THIQAH-RD--ALARB/blobs" / spec["sha256"])
    for candidate in candidates:
        if candidate.is_file():
            verify_file(candidate, spec)
            shutil.copyfile(candidate, destination)
            return
    if offline:
        raise FileNotFoundError(f"No verified local copy of {name}; rerun without --offline.")
    url = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['git_commit']}/data/{name}"
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    verify_file(destination, spec)


def collect(output: Path = DEFAULT_OUTPUT, local_dir: Path | None = None,
            offline: bool = False, force: bool = False) -> dict:
    """Archive unmodified original files and prepared splits, with SHA256 checks.

    The original test file and prepared validation file are copied as bytes only.
    Neither is decoded, sampled, used to fit a model, or used for statistics here.
    """
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; choose a new output or use --force.")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    source = lock["official_dataset"]
    split_path = ROOT / "artifacts/split_manifest.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if splits["source"]["revision"] != source["git_commit"]:
        raise ValueError("Source lock and prepared split revisions disagree.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alarb-bundle-", dir=output.parent) as temporary:
        staging = Path(temporary)
        original = staging / "original"
        original.mkdir()
        files = {}
        for name, spec in source["files"].items():
            destination = original / name
            _copy_or_download(name, spec, destination, source, local_dir, offline)
            files[f"original/{name}"] = {**spec, "publisher_hash_verified": True}
        # Preserve the dataset card supplied at the pinned revision as attribution.
        card_cache = ROOT / ".cache/huggingface/hub/datasets--THIQAH-RD--ALARB/blobs/71551080570e3e00c7a6875b788522db632038a4"
        card = staging / "PUBLISHER_DATASET_CARD.md"
        if card_cache.is_file():
            shutil.copyfile(card_cache, card)
        elif not offline:
            url = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['git_commit']}/README.md"
            with urllib.request.urlopen(url, timeout=60) as response:
                card.write_bytes(response.read())
        else:
            raise FileNotFoundError("Publisher dataset card is not cached.")
        for split, relative in splits["paths"].items():
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing {relative}; run prepare_splits.py first.")
            destination = staging / "course" / Path(relative).relative_to("data")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            files[str(destination.relative_to(staging))] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "rows": splits["rows"][split],
                "row_count_source": "existing split manifest; no holdout content inspection",
            }
        shutil.copyfile(LOCK_PATH, staging / "SOURCE_LOCK.json")
        shutil.copyfile(ROOT / "licenses/DATASET_APACHE_2_0.txt", staging / "DATASET_LICENSE.txt")
        shutil.copyfile(split_path, staging / "split_manifest.json")
        shutil.copyfile(ROOT / "DATA_SPLIT_POLICY.md", staging / "DATA_SPLIT_POLICY.md")
        readme = (
            "# ALARB project dataset collection\n\n"
            f"Parent paper: {lock['parent_paper']['url']}\n\n"
            f"Official dataset: {source['url']}\n\n"
            f"Pinned revision: {source['git_commit']}\n\n"
            f"Publisher dataset license declaration: {source['license']}. "
            "The publisher card and its citation are included unchanged; the full license "
            "is included as DATASET_LICENSE.txt. "
            "License text: https://www.apache.org/licenses/LICENSE-2.0\n\n"
            "The archive contains the two unmodified original Parquet files (12,012 train and "
            "1,329 test cases) and the existing course splits (9,339 training, 2,673 test, "
            "1,329 validation). These are two views of the same 13,341 cases; do not add them together. "
            "The paper reports 13,344 cases, three more than this released revision.\n\n"
            "Schema: case_facts, court_reasoning and applicable_laws are lists of strings; "
            "verdict is a string. Data are Arabic Saudi commercial-court cases.\n\n"
            "The publisher test split is the course validation holdout. It is packaged as bytes "
            "only and must remain unopened during development. No holdout cases were decoded "
            "or evaluated by this collection command.\n\n"
            "Residual personal names are known to exist in the release. This is an unmodified "
            "research-data archive, not a newly anonymized release. Keep case text and model "
            "outputs out of the public project repository.\n\n"
            "To load development data: pandas.read_parquet('course/development/train.parquet'). "
            "SHA256 digests and sizes are recorded in collection_manifest.json.\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8")
        manifest = {
            "status": "PASS", "source": source,
            "parent_paper": lock["parent_paper"],
            "official_rows": {"train": 12012, "test": 1329, "total": 13341},
            "course_rows": splits["rows"], "files": files,
            "holdout_decoded": False,
            "privacy_note": "Published release contains residual identifiers; no new anonymization claimed.",
        }
        (staging / "collection_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_zip = staging / "bundle.zip"
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file() and path != temp_zip:
                    archive.write(path, str(path.relative_to(staging)))
        with zipfile.ZipFile(temp_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("Dataset archive failed ZIP integrity verification.")
        os.replace(temp_zip, output)
    return {**manifest, "archive": output.name, "archive_bytes": output.stat().st_size,
            "archive_sha256": sha256(output)}
