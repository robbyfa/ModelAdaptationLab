"""Few-shot candidate; no claim of validation optimization in Phase 2."""

from pathlib import Path
from typing import Any

from model_adaptation_lab.models.loader import GenerationBackend
from model_adaptation_lab.models.prompting import choose_demonstrations
from model_adaptation_lab.strategies.base import BaseStrategy


class FewShotStrategy(BaseStrategy):
    name = "prompt"

    def __init__(
        self,
        backend: GenerationBackend,
        prompts: dict[str, Any],
        manifests: Path,
        processed: Path,
    ) -> None:
        super().__init__(backend, prompts)
        self.demonstrations = choose_demonstrations(manifests, processed, prompts)
        self.prompt_version = str(prompts["few_shot_version"])
