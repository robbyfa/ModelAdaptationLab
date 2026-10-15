"""Record a development measurement run in explicitly local MLflow."""

import importlib
from pathlib import Path
from typing import Any

from model_adaptation_lab.evaluation.quality.packet import verify_packet
from model_adaptation_lab.io import LabError, atomic_write, canonical, file_hash, read_json


def track_quality(
    *,
    packet: Path,
    scores: Path,
    output: Path,
    tracking_uri: str = "sqlite:///outputs/mlflow.db",
    semantic: Path | None = None,
) -> dict[str, Any]:
    if not tracking_uri.startswith("sqlite:///"):
        raise LabError("Quality tracking requires an explicit local SQLite store")
    if output.exists():
        raise LabError("Tracking receipt exists; preserve prior runs")
    manifest, cases = verify_packet(packet)
    result = read_json(scores)
    if result.get("packet_sha256") != file_hash(packet / "manifest.json"):
        raise LabError("Scores do not belong to this packet")
    if "semantic_sha256" in result:
        semantic = semantic or packet / "semantic.json"
        if file_hash(semantic) != result["semantic_sha256"]:
            raise LabError("Semantic diagnostics changed after scoring")
    for path, expected in result["review_files"].items():
        if file_hash(Path(path)) != expected:
            raise LabError("Review file changed after scoring")
    for source in manifest["source_runs"]:
        path = Path(source["path"])
        if file_hash(path / "run.json") != source["run_sha256"]:
            raise LabError("Source run changed after packet creation")
        for strategy, expected in source["results"].items():
            if file_hash(path / f"{strategy}.jsonl") != expected:
                raise LabError("Source answers changed after packet creation")
    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as exc:
        raise LabError("Install the training extra for local MLflow tracking") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("model-adaptation-lab-quality")
    with mlflow.start_run(run_name=packet.name) as run:
        mlflow.log_params(
            {
                "scope": "validation_only",
                "packet_sha256": file_hash(packet / "manifest.json"),
                "rubric_sha256": manifest["files"]["rubric.md"],
                "freeze_sha256": manifest["freeze_sha256"],
                "human_review_coverage": result["human_review_coverage"],
            }
        )
        mlflow.set_tag("measurement_validity", result["measurement_validity"])
        mlflow.log_metrics(
            {
                "outputs": len(cases),
                "questions": len({c["question_id"] for c in cases}),
                **{
                    k: result[k]
                    for k in (
                        "failed_outputs",
                        "empty_outputs",
                        "truncated_outputs",
                        "independent_human_reviewers",
                        "jointly_reviewed_items",
                    )
                },
            }
        )
        for name in (*manifest["files"], "manifest.json"):
            mlflow.log_artifact(str(packet / name), artifact_path="packet")
        mlflow.log_artifact(str(scores), artifact_path="assessment")
        if semantic is not None:
            mlflow.log_artifact(str(semantic), artifact_path="assessment")
        reproduction = Path(result["scorer_reproduction_directory"])
        for name, expected in result["scorer_code_snapshot_sha256"].items():
            if file_hash(reproduction / "code_snapshot" / name) != expected:
                raise LabError("Scorer code snapshot changed")
        mlflow.log_artifacts(
            str(reproduction / "code_snapshot"), artifact_path="scorer_code_snapshot"
        )
        for index, source in enumerate(manifest["source_runs"]):
            directory = Path(source["path"])
            source_state = read_json(directory / "run.json")
            for name, expected in source_state.get("code_snapshot_sha256", {}).items():
                if file_hash(directory / "code_snapshot" / name) != expected:
                    raise LabError("Source inference code snapshot changed")
            for name in ("run.json", *(s + ".jsonl" for s in source["results"])):
                mlflow.log_artifact(str(directory / name), artifact_path=f"source_runs/{index}")
            if (directory / "code_snapshot").exists():
                mlflow.log_artifacts(
                    str(directory / "code_snapshot"),
                    artifact_path=f"source_runs/{index}/code_snapshot",
                )
        for path in result["review_files"]:
            mlflow.log_artifact(path, artifact_path="ratings")
        receipt = {
            "mlflow_run_id": run.info.run_id,
            "tracking_uri": tracking_uri,
            "packet_sha256": file_hash(packet / "manifest.json"),
            "scores_sha256": file_hash(scores),
            "measurement_validity": result["measurement_validity"],
        }
    atomic_write(output, canonical(receipt))
    return receipt
