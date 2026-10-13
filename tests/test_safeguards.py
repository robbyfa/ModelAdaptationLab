"""Synthetic-only freeze, source integrity, CLI, and persistence tests."""

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from model_adaptation_lab.cli import main
from model_adaptation_lab.data.audit import write_report
from model_adaptation_lab.data.ingest import extract_original, load_config, original_questions
from model_adaptation_lab.data.splits import freeze, load_split, verify
from model_adaptation_lab.io import LabError, canonical, digest


def synthetic_audit():
    return {
        "splits": {"train": {"rows": 1}, "validation": {"rows": 1}},
        "cross_split_exact_question_groups": [],
        "evidence": {"status": "available"},
        "dataset_config": {"original": {"revision": "a" * 40}},
        "dataset_config_sha256": "b" * 64,
        "sources": {},
        "source_files_verified": True,
    }


def synthetic_rows(row, write_json):
    q = original_questions(write_json([row]), "train")[0]
    return [q, replace(q, question_id="SYNTH_DEV_1", split="validation", question="Other query")]


def test_stable_freeze_and_integrity(row, write_json, tmp_path):
    rows = synthetic_rows(row, write_json)
    target = tmp_path / "manifests"
    lock = freeze(rows, synthetic_audit(), target)
    assert freeze(list(reversed(rows)), synthetic_audit(), target) == lock
    assert verify(target) == lock
    assert load_split(target, "train", purpose="training")[0]["question_id"] == rows[0].question_id
    (target / "train.jsonl").write_text("tampered")
    with pytest.raises(LabError, match="hash mismatch"):
        verify(target)


def test_frozen_files_not_overwritten(row, write_json, tmp_path):
    rows = synthetic_rows(row, write_json)
    target = tmp_path / "manifests"
    freeze(rows, synthetic_audit(), target)
    before = (target / "train.jsonl").read_bytes()
    with pytest.raises(LabError, match="Frozen artifact differs"):
        freeze([replace(rows[0], question="changed"), rows[1]], synthetic_audit(), target)
    assert (target / "train.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "split,purpose",
    [
        ("validation", "training"),
        ("test", "training"),
        ("validation", "demonstrations"),
        ("train", "final_evaluation"),
        ("../../secret", "training"),
    ],
)
def test_role_boundaries(tmp_path, split, purpose):
    with pytest.raises(LabError, match="cannot be accessed"):
        load_split(tmp_path, split, purpose=purpose)


def test_pending_test_refused(row, write_json, tmp_path):
    freeze(synthetic_rows(row, write_json), synthetic_audit(), tmp_path)
    assert (tmp_path / "test.jsonl").read_bytes() == b""
    with pytest.raises(LabError, match="protocol is pending"):
        load_split(tmp_path, "test", purpose="final_evaluation")


def test_overlap_frozen_but_use_blocked(row, write_json, tmp_path):
    audit = synthetic_audit()
    audit["cross_split_exact_question_groups"] = [["synthetic-a", "synthetic-b"]]
    lock = freeze(synthetic_rows(row, write_json), audit, tmp_path)
    assert lock["experimental_use_blockers"] == ["cross_split_exact_questions"]
    with pytest.raises(LabError, match="Experimental use blocked"):
        load_split(tmp_path, "train", purpose="training")


def test_derivative_cannot_enter_manifests(row, write_json, tmp_path):
    rows = synthetic_rows(row, write_json)
    with pytest.raises(LabError, match="Only official"):
        freeze([replace(rows[0], source="nvidia"), rows[1]], synthetic_audit(), tmp_path)


def test_archive_path_safety(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("../../training_Q_A.json", "x/dev_Q_A.json", "x/training_dev_technotes.json"):
            entry = tarfile.TarInfo(name)
            entry.size = 2
            tar.addfile(entry, io.BytesIO(b"[]"))
        link = tarfile.TarInfo("x/link")
        link.type, link.linkname = tarfile.SYMTYPE, "/tmp/should-not-extract"
        tar.addfile(link)
    output = tmp_path / "out"
    assert len(extract_original(archive, output)) == 3
    assert sorted(p.name for p in output.iterdir()) == [
        "dev_Q_A.json",
        "training_Q_A.json",
        "training_dev_technotes.json",
    ]


def test_immutable_config_and_canonical_metadata(write_json):
    config = load_config(Path("configs/dataset.yaml"))
    assert len(config["original"]["revision"]) == 40
    assert digest(canonical({"b": 1, "a": 2})) == digest(canonical({"a": 2, "b": 1}))
    config["original"]["revision"] = "main"
    with pytest.raises(LabError, match="immutable"):
        load_config(write_json(config))


def test_missing_sources_report_and_nonzero_exit(tmp_path):
    out = tmp_path / "reports"
    assert (
        main(["data", "audit", "--raw-dir", str(tmp_path / "missing"), "--reports-dir", str(out)])
        == 2
    )
    report = json.loads((out / "dataset_audit.json").read_text())
    assert report["evidence"]["gold_document_coverage"] is None
    assert not report["final_test_ready"]
    assert "Missing official" in (out / "dataset_audit.md").read_text()
    write_report(report, out)
    assert json.loads((out / "dataset_audit.json").read_text()) == report


def test_processed_records_are_checked(row, write_json, tmp_path):
    from model_adaptation_lab.data.splits import load_annotations
    from model_adaptation_lab.io import jsonl

    rows = synthetic_rows(row, write_json)
    freeze(rows, synthetic_audit(), tmp_path / "manifests")
    path = tmp_path / "train_annotations.jsonl"
    path.write_bytes(jsonl([rows[0].record()]))
    assert (
        load_annotations(tmp_path / "manifests", tmp_path, "train", purpose="training")[0]["answer"]
        == "Reset it."
    )
    path.write_bytes(jsonl([replace(rows[0], answer="tampered").record()]))
    with pytest.raises(LabError, match="record hash mismatch"):
        load_annotations(tmp_path / "manifests", tmp_path, "train", purpose="training")


def test_cached_download_checksum_required(tmp_path):
    from model_adaptation_lab.data.ingest import download

    cached = tmp_path / "cached"
    cached.write_bytes(b"synthetic")
    with pytest.raises(LabError, match="Checksum mismatch"):
        download("https://example.invalid/never-requested", cached, "0" * 64)


def test_download_bad_bytes_removed(tmp_path, monkeypatch):
    from model_adaptation_lab.data.ingest import download

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"bad bytes"))
    dest = tmp_path / "download.json"
    with pytest.raises(LabError, match="checksum mismatch"):
        download("https://example.invalid/synthetic", dest, "0" * 64)
    assert not dest.exists()
    assert not dest.with_suffix(".json.partial").exists()


def test_changed_source_rejected_before_freeze(row, write_json, tmp_path):
    from model_adaptation_lab.data.audit import run_audit

    config = load_config(Path("configs/dataset.yaml"))
    write_json([row], "original/training_Q_A.json")
    with pytest.raises(LabError, match="differs from pinned bytes"):
        run_audit(tmp_path, config)


def test_complete_synthetic_pipeline(row, write_json, tmp_path):
    """No downloaded dataset is touched; these sources are synthetic and pinned locally."""
    from model_adaptation_lab.io import file_hash

    config = load_config(Path("configs/dataset.yaml"))
    raw = tmp_path / "raw"
    train = write_json([row], "raw/original/training_Q_A.json")
    dev_row = dict(row, QUESTION_ID="SYNTH_DEV", QUESTION_TEXT="A different request")
    dev = write_json([dev_row], "raw/original/dev_Q_A.json")
    corpus = write_json(
        {
            "synthetic-doc": {
                "id": "synthetic-doc",
                "title": "Synthetic",
                "text": "Reset it.",
                "metadata": {},
            }
        },
        "raw/original/training_dev_technotes.json",
    )
    nvidia = write_json(
        [
            {
                "id": "SYNTH_INSPECT",
                "question": "Inspection only?",
                "answer": "-",
                "is_impossible": True,
                "contexts": [],
            }
        ],
        "raw/nvidia/train.json",
    )
    config["original"]["extracted_sha256"] = {p.name: file_hash(p) for p in (train, dev, corpus)}
    config["nvidia"]["sha256"] = file_hash(nvidia)
    config_path = write_json(config, "synthetic-config.json")
    manifests, processed = tmp_path / "manifests", tmp_path / "processed"
    args = [
        "data",
        "prepare",
        "--raw-dir",
        str(raw),
        "--config",
        str(config_path),
        "--manifests-dir",
        str(manifests),
        "--processed-dir",
        str(processed),
        "--reports-dir",
        str(tmp_path / "reports"),
    ]
    assert main(args) == 0
    assert main(args) == 0  # Idempotent, byte-stable assignments.
    assert verify(manifests)["final_test_ready"] is False
    doc = json.loads((processed / "source_documents.jsonl").read_text())
    assert set(doc) == {"document_id", "title", "text", "source_url", "source"}
    assert doc["source"] == "original_technotes"
    assert "SYNTH_INSPECT" not in (manifests / "train.jsonl").read_text()
