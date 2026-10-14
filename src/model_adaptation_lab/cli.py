"""Dataset and validation inference CLI; no mock substitution or final-test access."""

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
from model_adaptation_lab.evaluation.runner import run_evaluation
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, jsonl
from model_adaptation_lab.models.loader import yaml_config
from model_adaptation_lab.retrieval.indexing import build_index, verify_index


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Model Adaptation Lab — audited data and validation inference"
    )
    commands = root.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Run real-model validation inference only")
    evaluate.add_argument(
        "--strategy",
        choices=["base", "prompt", "both", "rag", "available", "lora", "lora-rag", "adapted"],
        required=True,
    )
    evaluate.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    evaluate.add_argument("--generation-config", type=Path, default=Path("configs/generation.yaml"))
    evaluate.add_argument("--prompt-config", type=Path, default=Path("configs/prompts.yaml"))
    evaluate.add_argument(
        "--manifests-dir", type=Path, default=Path("data/manifests/techqa-internal-v1")
    )
    evaluate.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/techqa-internal-v1")
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; previous attempts are never overwritten",
    )
    evaluate.add_argument("--cache-dir", type=Path, default=Path("outputs/model-cache"))
    evaluate.add_argument("--device", choices=["cuda", "cpu", "mps"])
    evaluate.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    evaluate.add_argument("--local-files-only", action="store_true")
    evaluate.add_argument(
        "--smoke",
        action="store_true",
        help="Two balanced validation rows and at most 64 new tokens",
    )
    evaluate.add_argument(
        "--limit", type=int, help="Validation subset size; recorded in run metadata"
    )
    evaluate.add_argument("--retrieval-config", type=Path, default=Path("configs/retrieval.yaml"))
    evaluate.add_argument("--index-dir", type=Path, help="Required for rag or available strategies")
    evaluate.add_argument("--adapter-dir", type=Path, help="Completed training run directory")
    evaluate.add_argument("--embedding-device", choices=["cpu", "cuda", "mps"])
    retrieval = commands.add_parser(
        "retrieval", help="Build/verify a frozen source-document FAISS index"
    )
    retrieval_actions = retrieval.add_subparsers(dest="retrieval_action", required=True)
    for action in ("build", "verify"):
        cmd = retrieval_actions.add_parser(action)
        cmd.add_argument("--index-dir", type=Path, required=True)
        cmd.add_argument(
            "--manifests-dir", type=Path, default=Path("data/manifests/techqa-internal-v1")
        )
        if action == "build":
            cmd.add_argument(
                "--processed-dir", type=Path, default=Path("data/processed/techqa-internal-v1")
            )
            cmd.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
            cmd.add_argument("--cache-dir", type=Path, default=Path("outputs/model-cache"))
            cmd.add_argument("--embedding-device", choices=["cpu", "cuda", "mps"])
            cmd.add_argument(
                "--max-documents",
                type=int,
                help="Explicit QA-independent smoke subset; never a primary index",
            )
            cmd.add_argument("--local-files-only", action="store_true")
    train = commands.add_parser("train", help="Fit real LoRA on guarded training data")
    train.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    train.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    train.add_argument("--prompt-config", type=Path, default=Path("configs/prompts.yaml"))
    train.add_argument("--generation-config", type=Path, default=Path("configs/generation.yaml"))
    train.add_argument(
        "--manifests-dir", type=Path, default=Path("data/manifests/techqa-internal-v1")
    )
    train.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/techqa-internal-v1")
    )
    train.add_argument("--cache-dir", type=Path, default=Path("outputs/model-cache"))
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", choices=["cuda", "mps", "cpu"])
    train.add_argument("--fraction", type=float, choices=[0.25, 0.5, 1.0], default=1.0)
    train.add_argument(
        "--smoke", action="store_true", help="Real two-step training on four records per split"
    )
    train.add_argument("--local-files-only", action="store_true")
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
        if args.command == "train":
            from model_adaptation_lab.models.lora import train_lora

            model = yaml_config(args.model_config)
            if args.device:
                model["device"] = args.device
            state = train_lora(
                config=yaml_config(args.config),
                model_config=model,
                generation=yaml_config(args.generation_config),
                prompts=yaml_config(args.prompt_config),
                manifests=args.manifests_dir,
                processed=args.processed_dir,
                output=args.output_dir,
                cache=args.cache_dir,
                local_files_only=args.local_files_only,
                smoke=args.smoke,
                fraction=args.fraction,
            )
            print(f"Training {state['status']}: {args.output_dir / 'training.json'}")
            return 0
        if args.command == "retrieval":
            if args.retrieval_action == "verify":
                result = verify_index(args.index_dir, args.manifests_dir)
            else:
                config = yaml_config(args.config)
                if args.embedding_device:
                    config["embedding"]["device"] = args.embedding_device
                result = build_index(
                    manifests=args.manifests_dir,
                    processed=args.processed_dir,
                    config=config,
                    output=args.index_dir,
                    cache=args.cache_dir,
                    local_files_only=args.local_files_only,
                    max_documents=args.max_documents,
                )
            print(
                f"Index {result['status']}: {result['documents_indexed']} documents, "
                f"{result['chunks_indexed']} chunks; {result['corpus_scope']}"
            )
            return 0
        if args.command == "evaluate":
            model = yaml_config(args.model_config)
            generation = yaml_config(args.generation_config)
            if args.device:
                model["device"] = args.device
            if args.dtype:
                model["dtype"] = args.dtype
            if args.smoke:
                generation["max_new_tokens"] = min(generation.get("max_new_tokens", 64), 64)
            if args.limit is not None and args.limit <= 0:
                raise LabError("--limit must be positive")
            names = {
                "both": ["base", "prompt"],
                "available": ["base", "prompt", "rag"],
                "adapted": ["lora", "lora-rag"],
            }.get(args.strategy, [args.strategy])
            retrieval_config = (
                yaml_config(args.retrieval_config) if set(names) & {"rag", "lora-rag"} else None
            )
            if retrieval_config is not None and args.embedding_device:
                retrieval_config["embedding"]["device"] = args.embedding_device
            state = run_evaluation(
                strategy_names=names,
                model=model,
                generation=generation,
                prompts=yaml_config(args.prompt_config),
                manifests=args.manifests_dir,
                processed=args.processed_dir,
                output=args.output_dir,
                cache=args.cache_dir,
                smoke=args.smoke,
                limit=args.limit,
                local_files_only=args.local_files_only,
                retrieval_config=retrieval_config,
                index_path=args.index_dir,
                adapter_path=args.adapter_dir,
            )
            print(f"Validation run {state['status']}: {args.output_dir / 'run.json'}")
            return 0 if state["status"] == "completed" else 1
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
    except KeyboardInterrupt:
        print("error: Interrupted; partial run artifacts have been preserved", file=sys.stderr)
        return 130
    except (LabError, OSError, tarfile.TarError, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
