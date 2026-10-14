import csv
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import zipfile

import pandas as pd
import pytest

from ktlab.data import _acquire, _upstream, _validate_preprocessing_pin, audit_student_splits, audit_causal_validation, download_file, inspect_dataset, normalize_encoding


def write_membership(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["uid", "fold", "responses"])
        writer.writerows((uid, fold, "0,1,0") for uid, fold in rows)


def test_student_folds_are_disjoint_and_hash_stable(tmp_path):
    training, heldout = tmp_path / "training.csv", tmp_path / "test.csv"
    rows = [(str(i), i % 5) for i in range(20)]
    write_membership(training, rows)
    write_membership(heldout, [("heldout", -1)])
    audit = audit_student_splits(training, heldout)
    assert audit["train_valid_students"] == 20
    assert audit["test_students"] == 1
    assert all(x["training_students"] == 16 and x["validation_students"] == 4 for x in audit["folds"].values())
    write_membership(training, rows[::-1])
    assert audit_student_splits(training, heldout) == audit


def test_student_overlap_rejected(tmp_path):
    training, heldout = tmp_path / "training.csv", tmp_path / "test.csv"
    write_membership(training, [(str(i), i) for i in range(5)])
    write_membership(heldout, [("0", -1)])
    with pytest.raises(ValueError, match="overlap"):
        audit_student_splits(training, heldout)


def test_duplicate_student_rejected(tmp_path):
    training, heldout = tmp_path / "training.csv", tmp_path / "test.csv"
    write_membership(training, [("same", i) for i in range(5)])
    write_membership(heldout, [("heldout", -1)])
    with pytest.raises(ValueError, match="duplicate"):
        audit_student_splits(training, heldout)


@pytest.mark.parametrize("server_resumes", [True, False])
def test_partial_download_resumes_or_safely_restarts(tmp_path, server_resumes):
    content = b"user_id,correct\n" + b"1,0\n" * 32
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range")
            requests.append(requested)
            start = int(requested.split("=")[1].split("-")[0]) if requested and server_resumes else 0
            self.send_response(206 if requested and server_resumes else 200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(content) - start))
            if requested and server_resumes:
                self.send_header("Content-Range", f"bytes {start}-{len(content)-1}/{len(content)}")
            self.end_headers()
            self.wfile.write(content[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/dataset.csv"
        target = tmp_path / "dataset.csv"
        target.with_suffix(".csv.part").write_bytes(content[:17])
        target.with_suffix(".csv.part.json").write_text(json.dumps({"url": url}))
        expected = hashlib.sha256(content).hexdigest()
        receipt = download_file(url, target, expected)
        assert target.read_bytes() == content
        assert receipt["sha256"] == expected
        assert requests == ["bytes=17-"]
        assert download_file(url, target, expected) == receipt
        target.write_bytes(content + b"changed")
        with pytest.raises(ValueError, match="hash mismatch"):
            download_file(url, target, expected)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_question_validation_hides_current_multikc_answer():
    root = Path(__file__).resolve().parents[1]
    split = _upstream(root, "split_datasets")
    # Questions 10 and 12 each have two concepts. Neither target may see the
    # other concept's answer to the same question in its available history.
    frame = pd.DataFrame([{"uid": "7", "fold": 2, "questions": "10,10,11,12,12,13", "concepts": "0,1,2,3,4,5", "responses": "1,1,0,1,1,0", "is_repeat": "0,1,0,0,1,0"}])
    ok, expanded = split.generate_question_sequences(frame, set(frame.columns), window=False, min_seq_len=3, maxlen=20)
    assert ok
    targets = []
    for row in expanded.itertuples(index=False):
        qidxs = [int(x) for x in row.qidxs.split(",")]
        masks = [int(x) for x in row.selectmasks.split(",")]
        for index, mask in enumerate(masks):
            if mask == 1:
                assert qidxs[index] not in qidxs[:index]
                targets.append(qidxs[index])
        assert row.uid == "7" and row.fold == 2
    assert sorted(targets) == [1, 2, 2, 3]


def test_unprepared_inspection_does_not_download(tmp_path):
    result = inspect_dataset("nips34", tmp_path)
    assert result["dataset"] == "nips_task34"
    assert result["status"] == "not_prepared"
    assert not (tmp_path / "data").exists()


def test_encoding_normalization_preserves_consumed_fields(tmp_path):
    source, target = tmp_path / "original.csv", tmp_path / "utf8.csv"
    source.write_bytes(b"user_id,order_id,problem_id,skill_id,correct,description\n1,2,3,4_5,1,cost \x802\n")
    report = normalize_encoding(source, target, "cp1252")
    assert report["rows"] == 1
    assert target.read_text().endswith("1,2,3,4_5,1,cost \u20ac2\n")
    source.write_bytes(b"user_id,order_id,problem_id,skill_id,correct\n1,2,3,\x80,1\n")
    with pytest.raises(ValueError, match="consumed"):
        normalize_encoding(source, target, "cp1252")


def test_causal_audit_rejects_same_question_leakage(tmp_path):
    path = tmp_path / "validation.csv"
    frame = pd.DataFrame([{"uid": "7", "fold": 0, "qidxs": "0,1,1,-1", "selectmasks": "-1,-1,1,-1"}])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="leaked"):
        audit_causal_validation(path, 0, {"7"}, 4)
    frame["qidxs"] = "0,1,2,-1"
    frame.to_csv(path, index=False)
    assert audit_causal_validation(path, 0, {"7"}, 4)["question_targets"] == 1
    with pytest.raises(ValueError, match="outside"):
        audit_causal_validation(path, 0, {"8"}, 4)


def test_same_size_corrupted_archive_extraction_is_replaced(tmp_path):
    archive = tmp_path / "data/downloads/source.zip"
    archive.parent.mkdir(parents=True)
    content = b"uid,label\n1,0\n"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/input.csv", content)
    destination = tmp_path / "data/raw/example/input.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content.replace(b"1,0", b"1,1"))
    spec = {"url": "https://example.invalid/source.zip", "download_filename": "source.zip", "archive_files": {"input.csv": "input.csv"}}
    raw, receipt = _acquire("example", tmp_path, spec)
    assert (raw / "input.csv").read_bytes() == content
    assert receipt["raw_files"]["input.csv"]["sha256"] == hashlib.sha256(content).hexdigest()


def test_upstream_modules_do_not_cross_project_roots(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for value, root in enumerate(roots):
        package = root / "third_party/pykt/pykt/preprocess"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "example.py").write_text(f"VALUE = {value}\n")
    assert _upstream(roots[0], "example").VALUE == 0
    assert _upstream(roots[1], "example").VALUE == 1


def test_modified_upstream_preprocessing_is_rejected(tmp_path):
    source = tmp_path / "third_party/pykt/pykt/preprocess/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("original = True\n")
    lock = {"commit": "pinned", "files_sha256": {"pykt/preprocess/example.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path / "third_party/pykt.lock.json").write_text(json.dumps(lock))
    catalog = {"pykt_commit": "pinned", "protocol": {"min_seq_len": 3, "maxlen": 200, "kfold": 5, "split_seed": 1024, "test_ratio": 0.2}}
    _validate_preprocessing_pin(tmp_path, catalog)
    source.write_text("original = False\n")
    with pytest.raises(ValueError, match="bytes differ"):
        _validate_preprocessing_pin(tmp_path, catalog)
    catalog["protocol"]["test_ratio"] = 0.3
    with pytest.raises(ValueError, match="frozen"):
        _validate_preprocessing_pin(tmp_path, catalog)


def test_completed_partial_is_finalized_without_network(tmp_path):
    target = tmp_path / "complete.csv"
    content = b"uid,label\n1,0\n"
    url = "https://example.invalid/complete.csv"
    target.with_suffix(".csv.part").write_bytes(content)
    target.with_suffix(".csv.part.json").write_text(json.dumps({"url": url}))
    receipt = download_file(url, target, hashlib.sha256(content).hexdigest())
    assert target.read_bytes() == content
    assert receipt["bytes"] == len(content)
