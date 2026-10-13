"""One validation runner for all Phase 2 strategies, preserving every attempt."""

import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from model_adaptation_lab.data.splits import load_annotations, verify
from model_adaptation_lab.evaluation.metrics import diagnostics, summarize
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash
from model_adaptation_lab.models.loader import (
    GenerationBackend,
    TransformersBackend,
    validate_configs,
    validate_prompts,
)
from model_adaptation_lab.strategies.base import AnsweringStrategy, BaseStrategy
from model_adaptation_lab.strategies.few_shot import FewShotStrategy


def validation_sample(
    rows: list[dict[str, Any]], limit: int | None, seed: int
) -> list[dict[str, Any]]:
    if any(r["split"] != "validation" for r in rows):
        raise LabError("Phase 2 evaluation accepts validation only")
    if limit is not None and limit <= 0:
        raise LabError("Evaluation limit must be positive")
    # Alternate deterministically ordered strata to include both in a two-question smoke.
    strata = [[r for r in rows if r["answerable"] == flag] for flag in (True, False)]
    for group in strata:
        group.sort(key=lambda r: digest(f"{seed}:{r['question_id']}".encode()))
    ordered = []
    for i in range(max((len(group) for group in strata), default=0)):
        for group in strata:
            if i < len(group):
                ordered.append(group[i])
    return ordered[:limit]


def capture_environment() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    hardware: dict[str, str | None] = {}
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            try:
                hardware[key] = subprocess.check_output(
                    ["sysctl", "-n", key], stderr=subprocess.DEVNULL, text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                hardware[key] = None
    return {
        "host_hardware": hardware,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "tracked_diff_sha256": digest((git("diff", "HEAD") or "").encode()),
        "uv_lock_sha256": file_hash(Path("uv.lock")) if Path("uv.lock").exists() else None,
        "installed_packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
    }


def snapshot_code(output: Path) -> dict[str, str]:
    """Preserve code/config bytes for dirty development runs, without any dataset text."""
    files = sorted(Path("src/model_adaptation_lab").rglob("*.py"))
    files += sorted(Path("configs").glob("*.yaml"))
    files += [p for p in (Path("pyproject.toml"), Path("uv.lock")) if p.exists()]
    hashes = {}
    for path in files:
        content = path.read_bytes()
        atomic_write(output / "code_snapshot" / path, content)
        hashes[str(path)] = digest(content)
    return hashes


def evaluate_questions(
    strategy: AnsweringStrategy,
    rows: list[dict[str, Any]],
    output: Path,
    abstention: str,
) -> list[dict[str, Any]]:
    results = []
    # Exclusive creation prevents accidental replacement of failed or completed attempts.
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            started = time.perf_counter()
            try:
                answer = replace(strategy.answer(row["question"]), question_id=row["question_id"])
                record = asdict(answer) | {
                    "status": "ok",
                    "error": None,
                    "metrics": diagnostics(
                        answer.answer, row["answer"], row["answerable"], abstention
                    ),
                }
            except (LabError, RuntimeError) as exc:
                record = {
                    "question_id": row["question_id"],
                    "strategy": strategy.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "answer": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "metrics": None,
                }
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(record)
    return results


def run_evaluation(
    *,
    strategy_names: list[str],
    model: dict[str, Any],
    generation: dict[str, Any],
    prompts: dict[str, Any],
    manifests: Path,
    processed: Path,
    output: Path,
    cache: Path,
    smoke: bool,
    limit: int | None,
    local_files_only: bool,
    backend: GenerationBackend | None = None,
) -> dict[str, Any]:
    if not strategy_names or any(s not in {"base", "prompt"} for s in strategy_names):
        raise LabError("Only base and prompt strategies are implemented in Phase 2")
    if len(set(strategy_names)) != len(strategy_names):
        raise LabError("Repeated strategies are not allowed")
    validate_configs(model, generation)
    validate_prompts(prompts)
    lock = verify(manifests)
    if lock.get("protocol_id") != "techqa-internal-v1":
        raise LabError("Phase 2 requires the eligible techqa-internal-v1 protocol")
    rows = validation_sample(
        load_annotations(manifests, processed, "validation", purpose="selection"),
        (limit or 2) if smoke else limit,
        generation["seed"],
    )
    if not rows:
        raise LabError("No eligible validation questions")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise LabError(
            "Run directory exists; use a new directory to preserve previous attempts"
        ) from exc
    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": output.name,
        "code_snapshot_sha256": snapshot_code(output),
        "status": "initializing",
        "scope": "validation_only",
        "smoke": smoke,
        "backend_type": "injected_test" if backend else "real_transformers",
        "started_unix": time.time(),
        "model_config": model,
        "generation_config": generation,
        "prompt_config": prompts,
        "strategy_order": strategy_names,
        "validation_ids": [r["question_id"] for r in rows],
        "validation_records_sha256": digest(canonical(rows)),
        "freeze_sha256": file_hash(manifests / "freeze.json"),
        "protocol_lock": lock,
        "environment": capture_environment(),
        "model_metrics": None,
        "warmup": None,
        "results": {},
    }
    state_path = output / "run.json"
    atomic_write(state_path, canonical(state))
    try:
        if backend is None:
            backend = TransformersBackend(
                model, generation, cache, local_files_only=local_files_only
            )
        state["model_metrics"] = backend.metadata
        state["effective_generation_parameters"] = backend.generation_parameters
        strategies: list[BaseStrategy] = [
            BaseStrategy(backend, prompts)
            if name == "base"
            else FewShotStrategy(backend, prompts, manifests, processed)
            for name in strategy_names
        ]
        state["demonstration_ids"] = {
            s.name: [r["question_id"] for r in s.demonstrations] for s in strategies
        }
        state["demonstration_record_hashes"] = {
            s.name: [digest(canonical(r)) for r in s.demonstrations] for s in strategies
        }
        state["status"] = "running"
        atomic_write(state_path, canonical(state))
        # Same fixed, non-dataset warmup; record separately from measured questions.
        state["warmup"] = asdict(
            backend.generate([{"role": "user", "content": "Reply with the single word Ready."}])
        )
        for strategy in strategies:
            path = output / f"{strategy.name}.jsonl"
            results = evaluate_questions(strategy, rows, path, prompts["abstention"])
            state["results"][strategy.name] = summarize(results) | {"file_sha256": file_hash(path)}
            atomic_write(state_path, canonical(state))
        failures = sum(s["failed"] for s in state["results"].values())
        state["status"] = "partial_failure" if failures else "completed"
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["finished_unix"] = time.time()
        atomic_write(state_path, canonical(state))
