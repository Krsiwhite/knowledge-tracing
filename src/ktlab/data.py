"""Pinned pyKT dataset preparation. No model training or model metrics live here."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import http.client
import importlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(name: str) -> str:
    return {"nips34": "nips_task34", "assist09": "assist2009", "assist15": "assist2015"}.get(name.lower(), name.lower())


def _catalog(root: Path) -> dict:
    return json.loads((root / "configs/datasets.json").read_text(encoding="utf-8"))


def _validate_preprocessing_pin(root: Path, catalog: dict) -> None:
    lock = json.loads((root / "third_party/pykt.lock.json").read_text(encoding="utf-8"))
    if lock["commit"] != catalog["pykt_commit"]:
        raise ValueError("Dataset catalog and vendored pyKT commit differ")
    expected_protocol = {"min_seq_len": 3, "maxlen": 200, "kfold": 5, "split_seed": 1024, "test_ratio": 0.2}
    if catalog["protocol"] != expected_protocol:
        raise ValueError("This preparation entrypoint implements only the frozen v1 pyKT protocol")
    for relative, expected in lock["files_sha256"].items():
        if relative.startswith("pykt/preprocess/") and relative.endswith(".py"):
            if sha256_file(root / "third_party/pykt" / relative) != expected:
                raise ValueError(f"Vendored preprocessing bytes differ from upstream lock: {relative}")


def download_file(url: str, target: Path, expected_sha256: str | None = None, retries: int = 4) -> dict:
    """Resume HTTP transfers, reject HTML/error pages, then verify content hash.

    A server that ignores Range restarts the partial file instead of appending.
    Complete existing downloads are checked on every reuse, never overwritten.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(target.suffix + ".source.json")
    partial = target.with_suffix(target.suffix + ".part")
    partial_receipt = partial.with_suffix(partial.suffix + ".json")
    if target.exists():
        digest = sha256_file(target)
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f"Source hash mismatch for {target.name}; refusing changed dataset")
        if previous and (previous.get("url") != url or previous.get("sha256") != digest):
            raise ValueError(f"Download receipt mismatch for {target.name}")
        result = {**previous, "url": url, "sha256": digest, "bytes": target.stat().st_size}
        _write_json(receipt_path, result)
        return result
    if partial.exists() and partial_receipt.exists() and expected_sha256:
        previous = json.loads(partial_receipt.read_text())
        if previous.get("url") != url:
            raise ValueError("Partial download has no matching source receipt")
        if sha256_file(partial) == expected_sha256:
            # Recovery when transfer completed but interruption preceded rename.
            previous.update(sha256=expected_sha256, bytes=partial.stat().st_size)
            partial.replace(target)
            _write_json(receipt_path, previous)
            partial_receipt.unlink()
            return previous
    last_error = None
    for attempt in range(retries):
        previous = json.loads(partial_receipt.read_text()) if partial_receipt.exists() else {}
        if partial.exists() and previous.get("url") != url:
            raise ValueError("Partial download has no matching source receipt")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "ktlab-research/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if previous.get("etag"):
                headers["If-Range"] = previous["etag"]
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Dataset endpoint returned HTML; access/confirmation may be required")
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected resume range")
                receipt = {"url": url, "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
                _write_json(partial_receipt, receipt)
                received = 0
                with partial.open("ab" if resumed else "wb") as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        received += len(chunk)
                expected_length = response.headers.get("Content-Length")
                if expected_length and received != int(expected_length):
                    raise OSError("Incomplete download; retry will resume")
            digest = sha256_file(partial)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError(f"Source hash mismatch for {target.name}; refusing changed dataset")
            receipt.update(sha256=digest, bytes=partial.stat().st_size)
            partial.replace(target)
            _write_json(receipt_path, receipt)
            partial_receipt.unlink(missing_ok=True)
            return receipt
        except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Download failed for {target.name}: {last_error}") from last_error


def _upstream(root: Path, module: str):
    """Import preprocessing only, avoiding pyKT's eager import of every model."""
    root_tag = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    package_name = f"_ktlab_pinned_preprocess_{root_tag}"
    if package_name not in sys.modules:
        path = root / "third_party/pykt/pykt/preprocess"
        spec = importlib.util.spec_from_file_location(package_name, path / "__init__.py", submodule_search_locations=[str(path)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.{module}")


def _acquire(name: str, root: Path, spec: dict) -> tuple[Path, dict]:
    raw = root / "data/raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    download = root / "data/downloads" / spec["download_filename"]
    source = download_file(spec["url"], download, spec.get("sha256"))
    if "archive_files" in spec:
        with zipfile.ZipFile(download) as archive:
            for basename, relative in spec["archive_files"].items():
                matches = [item for item in archive.infolist() if Path(item.filename).name == basename and not item.is_dir()]
                if len(matches) != 1:
                    raise ValueError(f"Expected one archive member for {basename}, found {len(matches)}")
                destination = raw / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.stat().st_size == matches[0].file_size:
                    checksum = 0
                    with destination.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            checksum = zlib.crc32(chunk, checksum)
                    if checksum == matches[0].CRC:
                        continue
                temporary = destination.with_suffix(destination.suffix + ".part")
                with archive.open(matches[0]) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                temporary.replace(destination)
    else:
        destination = raw / spec["raw_filename"]
        if not destination.exists():
            try:
                os.link(download, destination)
            except OSError:
                shutil.copyfile(download, destination)
        elif sha256_file(destination) != source["sha256"]:
            raise ValueError("Raw input differs from downloaded source")
    source["raw_files"] = {str(p.relative_to(raw)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(raw.rglob("*.csv"))}
    return raw, source


def audit_student_splits(train_valid: Path, test: Path, folds: int = 5) -> dict:
    """Audit only student membership; never estimate metrics from test labels."""
    def read_membership(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return [(row["uid"], int(row["fold"])) for row in csv.DictReader(stream)]
    train = read_membership(train_valid)
    heldout = read_membership(test)
    if len({uid for uid, _ in train}) != len(train) or len({uid for uid, _ in heldout}) != len(heldout):
        raise ValueError("Original split contains duplicate student rows")
    train_uids, test_uids = {x[0] for x in train}, {x[0] for x in heldout}
    if train_uids & test_uids:
        raise ValueError("Student overlap between development and sealed test")
    if {fold for _, fold in train} != set(range(folds)) or {fold for _, fold in heldout} != {-1}:
        raise ValueError("Unexpected fold identifiers")
    counts = {}
    for fold in range(folds):
        valid = {uid for uid, part in train if part == fold}
        training = train_uids - valid
        counts[str(fold)] = {"training_students": len(training), "validation_students": len(valid), "train_validation_disjoint": not bool(training & valid), "validation_test_disjoint": not bool(valid & test_uids)}
    assignments = sorted(train + heldout)
    digest = hashlib.sha256(json.dumps(assignments, separators=(",", ":")).encode()).hexdigest()
    return {"students": len(train_uids) + len(test_uids), "train_valid_students": len(train_uids), "test_students": len(test_uids), "train_test_disjoint": True, "folds": counts, "assignment_sha256": digest}


def normalize_encoding(source: Path, target: Path, encoding: str) -> dict:
    """Transcode descriptive fields; verify every consumed column is ASCII."""
    original = source.read_bytes()
    text = original.decode(encoding, errors="strict")
    required = ["user_id", "order_id", "problem_id", "skill_id", "correct"]
    rows = csv.DictReader(io.StringIO(text))
    if not set(required).issubset(rows.fieldnames or []):
        raise ValueError("Encoding normalization input has missing required columns")
    count = 0
    for row in rows:
        if any(not row[key].isascii() for key in required):
            raise ValueError("Encoding normalization would affect a consumed data column")
        count += 1
    target.write_bytes(text.encode("utf-8"))
    return {"from_encoding": encoding, "to_encoding": "utf-8", "rows": count, "required_columns_ascii_unchanged": required, "source_sha256": hashlib.sha256(original).hexdigest(), "normalized_sha256": sha256_file(target)}


@contextlib.contextmanager
def _utc_timezone():
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        if hasattr(time, "tzset"):
            time.tzset()


def _validation_files(root: Path, prepared: Path, config: dict) -> dict:
    if config["num_q"] == 0:
        return {}
    import pandas as pd
    split = _upstream(root, "split_datasets")
    original = pd.read_csv(prepared / config["train_valid_original_file"], dtype={"uid": str})
    keys = set(original.columns) & set(split.ALL_KEYS)
    files = {}
    for fold in config["folds"]:
        filename = f"validation_question_fold{fold}.csv"
        frame = original[original["fold"] == fold].copy()
        valid, expanded = split.generate_question_sequences(frame, keys, window=False, min_seq_len=config["min_seq_len"], maxlen=config["maxlen"])
        if not valid or expanded.empty:
            raise ValueError(f"No causal validation sequences for fold {fold}")
        expanded.to_csv(prepared / filename, index=False)
        files[str(fold)] = filename
    return files


def audit_causal_validation(path: Path, fold: int, allowed_students: set[str], maxlen: int = 200) -> dict:
    """Check actual generated target masks, beyond a synthetic regression test."""
    students, targets = set(), set()
    rows = scored = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            uid = row["uid"]
            if uid not in allowed_students or int(row["fold"]) != fold:
                raise ValueError("Causal validation contains a student outside its fold")
            students.add(uid)
            qidxs = [int(x) for x in row["qidxs"].split(",")]
            masks = [int(x) for x in row["selectmasks"].split(",")]
            if len(qidxs) != maxlen or len(masks) != maxlen or not set(masks) <= {-1, 0, 1}:
                raise ValueError("Invalid causal validation shape or target mask")
            for index in range(1, len(masks)):
                if masks[index] == 1:
                    if qidxs[index] < 0 or qidxs[index] in qidxs[:index]:
                        raise ValueError("Target question answer leaked into its history")
                    scored += 1
                    targets.add((uid, qidxs[index]))
    if not rows or not scored:
        raise ValueError("Causal validation contains no scored targets")
    return {"students": len(students), "rows": rows, "scored_concept_targets": scored, "question_targets": len(targets), "fold_membership_valid": True, "same_question_history_hidden": True}


def _validation_audits(prepared: Path, config: dict, validation: dict) -> dict:
    if not validation:
        return {}
    with (prepared / config["train_valid_original_file"]).open(newline="", encoding="utf-8") as stream:
        membership = [(row["uid"], int(row["fold"])) for row in csv.DictReader(stream)]
    return {key: audit_causal_validation(prepared / filename, int(key), {uid for uid, fold in membership if fold == int(key)}, config["maxlen"]) for key, filename in validation.items()}


def prepare_dataset(name: str, root: Path) -> dict:
    """Acquire, prepare and hash a dataset under the project root; idempotent."""
    root = Path(root).resolve()
    name = canonical_name(name)
    catalog = _catalog(root)
    if name not in catalog["datasets"]:
        raise ValueError(f"Unknown dataset: {name}")
    _validate_preprocessing_pin(root, catalog)
    prepared = root / "data/prepared" / name
    manifest_path = prepared / "manifest.json"
    if manifest_path.exists():
        return inspect_dataset(name, root)
    prepared.mkdir(parents=True, exist_ok=True)
    spec = catalog["datasets"][name]
    raw, source = _acquire(name, root, spec)
    config_path = prepared / "data_config.json"
    _write_json(config_path, {})
    preprocessing = _upstream(root, name + "_preprocess")
    intermediate = prepared / "data.txt"
    protocol = catalog["protocol"]
    input_path = raw / spec["raw_filename"]
    if spec.get("source_encoding"):
        input_path = prepared / "raw_utf8.csv"
        source["normalization"] = normalize_encoding(raw / spec["raw_filename"], input_path, spec["source_encoding"])
    # pyKT converts timestamp strings through time.mktime; fix the timezone.
    with _utc_timezone(), (prepared / "preprocess.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        if name == "nips_task34":
            preprocessing.read_data_from_csv(str(raw / spec["raw_filename"]), str(raw / "metadata"), "task_3_4", str(intermediate))
        else:
            preprocessing.read_data_from_csv(str(input_path), str(intermediate))
        split = _upstream(root, "split_datasets")
        split.main(str(prepared), str(intermediate), name, str(config_path), protocol["min_seq_len"], protocol["maxlen"], protocol["kfold"])
        config = json.loads(config_path.read_text())[name]
        if config["num_q"] != spec["expected_num_q"] or config["num_c"] != spec["expected_num_c"]:
            raise ValueError(f"Dataset vocabulary differs from pinned pyKT: {config['num_q']} questions / {config['num_c']} concepts")
        validation = _validation_files(root, prepared, config)
    audit = audit_student_splits(prepared / config["train_valid_original_file"], prepared / config["test_original_file"], protocol["kfold"])
    validation_audits = _validation_audits(prepared, config, validation)
    tracked = [p for p in prepared.iterdir() if p.suffix in {".csv", ".json", ".txt"} and p.name != "manifest.json"]
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(tracked)}
    preprocess_dir = root / "third_party/pykt/pykt/preprocess"
    manifest = {"schema_version": 1, "dataset": name, "data_dir": str(prepared), "pykt_commit": catalog["pykt_commit"], "protocol": protocol, "source": source, "data_config": config, "validation_question_files": validation, "split_audit": audit, "files": files, "preprocessing_sha256": {p.name: sha256_file(p) for p in sorted(preprocess_dir.glob("*.py"))}, "test_policy": "Prepared and hashed; no model predictions or metrics calculated. Final evaluation requires separate alignment.", "manifest_path": str(manifest_path)}
    manifest["runtime"] = {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy"), "pandas": importlib.metadata.version("pandas"), "preprocessing_timezone": "UTC"}
    manifest["validation_audit"] = validation_audits
    _write_json(manifest_path, manifest)
    return manifest


def inspect_dataset(name: str, root: Path) -> dict:
    """Verify prepared bytes and student disjointness, returning the manifest."""
    root = Path(root).resolve()
    name = canonical_name(name)
    prepared = root / "data/prepared" / name
    manifest_path = prepared / "manifest.json"
    if not manifest_path.exists():
        return {"dataset": name, "status": "not_prepared", "manifest_path": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = _catalog(root)
    _validate_preprocessing_pin(root, catalog)
    if manifest["pykt_commit"] != catalog["pykt_commit"] or manifest["protocol"] != catalog["protocol"]:
        raise ValueError("Prepared manifest does not match the pinned upstream/protocol")
    expected_source = catalog["datasets"][name].get("sha256")
    if expected_source and manifest["source"]["sha256"] != expected_source:
        raise ValueError("Prepared manifest source hash differs from catalog")
    for filename, expected in manifest["preprocessing_sha256"].items():
        if sha256_file(root / "third_party/pykt/pykt/preprocess" / filename) != expected:
            raise ValueError(f"Pinned preprocessing source changed: {filename}")
    for filename, expected in manifest["files"].items():
        path = prepared / filename
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Prepared dataset integrity check failed: {filename}")
    config = manifest["data_config"]
    audit = audit_student_splits(prepared / config["train_valid_original_file"], prepared / config["test_original_file"], manifest["protocol"]["kfold"])
    if audit != manifest["split_audit"]:
        raise ValueError("Student split audit changed")
    manifest["status"] = "verified"
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "inspect", "download"])
    parser.add_argument("dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.action == "download":
        name = canonical_name(args.dataset)
        _, result = _acquire(name, args.root.resolve(), _catalog(args.root)["datasets"][name])
    elif args.action == "prepare":
        result = prepare_dataset(args.dataset, args.root)
    else:
        result = inspect_dataset(args.dataset, args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
