"""Verify adapter provenance before attaching it to the unchanged pinned base."""

import importlib
from pathlib import Path
from typing import Any

from model_adaptation_lab.io import LabError, file_hash, read_json
from model_adaptation_lab.models.loader import TransformersBackend


def verify_adapter(
    path: Path, model: dict[str, Any], manifests: Path, *, smoke: bool
) -> dict[str, Any]:
    state = read_json(path / "training.json")
    if state.get("status") != "completed":
        raise LabError("Adapter training did not complete")
    if state.get("scope") == "smoke" and not smoke:
        raise LabError("Smoke adapters cannot be used for normal evaluation")
    if state.get("freeze_sha256") != file_hash(manifests / "freeze.json"):
        raise LabError("Adapter dataset protocol mismatch")
    for key in ("model_id", "revision", "tokenizer_revision"):
        if state["model_config"][key] != model[key]:
            raise LabError(f"Adapter base mismatch: {key}")
    files = state.get("adapter_files", {})
    if not {"adapter_config.json", "adapter_model.safetensors"} <= set(files):
        raise LabError("Missing adapter weights/configuration")
    for name, expected in files.items():
        if Path(name).name != name or file_hash(path / "adapter" / name) != expected:
            raise LabError("Adapter artifact hash mismatch")
    return dict(state)


def attach_adapter(
    backend: TransformersBackend, path: Path, manifests: Path, *, smoke: bool
) -> None:
    state = verify_adapter(path, backend.model_config, manifests, smoke=smoke)
    try:
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise LabError("Adapter inference requires the training dependency extra") from exc
    backend.model = (
        peft.PeftModel.from_pretrained(
            backend.model, str(path / "adapter"), is_trainable=False, local_files_only=True
        )
        .to(backend.device)
        .eval()
    )
    backend.metadata["adapter"] = {
        "training_metadata_sha256": file_hash(path / "training.json"),
        "files": state["adapter_files"],
        "mlflow_run_id": state.get("mlflow_run_id"),
        "format": state["config"]["format"],
        "scope": state["scope"],
        "weights_merged": False,
    }
