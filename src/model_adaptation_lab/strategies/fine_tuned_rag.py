"""Same retrieval settings and same adapter as the standalone LoRA configuration."""

from model_adaptation_lab.strategies.rag import RAGStrategy


class FineTunedRAGStrategy(RAGStrategy):
    name = "lora-rag"
