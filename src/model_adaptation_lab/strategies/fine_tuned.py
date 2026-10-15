"""Same zero-shot inference protocol with the shared trained adapter attached."""

from model_adaptation_lab.strategies.base import BaseStrategy


class FineTunedStrategy(BaseStrategy):
    name = "lora"
