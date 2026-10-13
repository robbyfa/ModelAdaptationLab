"""Phase 1 CLI. Missing prerequisites are errors, never synthetic fallbacks."""

import argparse
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

from model_adaptation_lab.data.audit import run_audit, write_report
from model_adaptation_lab.data.ingest import (
    fetch_sources,
    load_config,
    original_documents,
    original_questions,
)
from model_adaptation_lab.data.protocol import load_protocol, prepare_protocol
from model_adaptation_lab.data.splits import freeze, verify
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, jsonl


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Model Adaptation Lab — Phase 1 data foundation")
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Download, audit, freeze, or verify data")
    actions = data.add_subparsers(dest="action", required=True)
    for name, help_text in (
        ("download", "Download pinned sources (~2.96 GB) and extract the official document subset"),
        ("audit", "Audit local sources; writes reports even when data are missing (exit 2)"),
        ("prepare", "Audit, save separated records/corpus, and freeze official assignments"),
        ("verify", "Verify existing frozen manifest hashes"),
        ("protocol", "Build the versioned eligible protocol and seal its internal final holdout"),
    ):
        cmd = actions.add_parser(name, help=help_text, description=help_text)
        cmd.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
        cmd.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
        cmd.add_argument("--reports-dir", type=Path, default=Path("reports"))
        suffix = "/techqa-internal-v1" if name == "protocol" else ""
        cmd.add_argument("--manifests-dir", type=Path, default=Path("data/manifests" + suffix))
        cmd.add_argument("--processed-dir", type=Path, default=Path("data/processed" + suffix))
        if name == "protocol":
            cmd.add_argument("--protocol", type=Path, default=Path("configs/protocol-v1.yaml"))
            cmd.add_argument("--source-manifests-dir", type=Path, default=Path("data/manifests"))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "verify":
            lock = verify(args.manifests_dir)
            print(
                f"Manifest hashes verified. Final test assigned: {lock['final_test_ready']}; "
                f"access: {lock.get('final_test_access', 'not_released')}"
            )
            return 0
        config = load_config(args.config)
        if args.action == "protocol":
            rows = original_questions(args.raw_dir / "original/training_Q_A.json", "train")
            rows += original_questions(args.raw_dir / "original/dev_Q_A.json", "validation")
            summary = prepare_protocol(
                rows,
                args.raw_dir / "original/training_dev_technotes.json",
                load_protocol(args.protocol),
                args.source_manifests_dir,
                args.manifests_dir,
                args.processed_dir,
                digest(canonical(config)),
            )
            print(f"Protocol frozen: {args.manifests_dir}; splits: {summary['splits']}")
            print("Train/validation ready. Internal final holdout sealed until Phase 5.")
            return 0
        if args.action == "download":
            fetch_sources(config, args.raw_dir)
            print("Pinned sources downloaded; run model-lab data audit")
            return 0
        report, rows = run_audit(args.raw_dir, config)
        write_report(report, args.reports_dir)
        print(f"Audit saved to {args.reports_dir / 'dataset_audit.md'}")
        complete = all(report["splits"][s]["rows"] for s in ("train", "validation"))
        complete = complete and report["evidence"]["status"] == "available"
        complete = complete and report["derivative"].get("rows", 0) > 0
        if not complete:
            raise LabError("Audit incomplete: missing source data; see the saved report")
        if args.action == "prepare":
            freeze(rows, report, args.manifests_dir)
            # Annotation files are separate from corpus. Never export NVIDIA contexts here.
            for split in ("train", "validation"):
                atomic_write(
                    args.processed_dir / f"{split}_annotations.jsonl",
                    jsonl(
                        [
                            q.record()
                            for q in sorted(rows, key=lambda q: q.question_id)
                            if q.split == split
                        ]
                    ),
                )
            docs = original_documents(args.raw_dir / "original" / "training_dev_technotes.json")
            atomic_write(
                args.processed_dir / "source_documents.jsonl",
                jsonl([asdict(docs[k]) for k in sorted(docs)]),
            )
            print("Official train/development assignments frozen. Final test remains pending.")
        for issue in report["issues"]:
            print(f"LIMITATION: {issue}")
        return 0
    except (LabError, OSError, tarfile.TarError, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
