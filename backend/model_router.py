"""
Model routing layer (clean NTU council version - LLM ONLY, SEQUENTIAL SAFE).
"""

from typing import Any, Optional

from .model_loader import run_unified_model
from .config import MODEL_BACKENDS, OLLAMA_MODELS


# =====================================================
# MODEL ROUTER
# =====================================================

class ModelRouter:
    """
    Unified inference interface:

    BACKENDS:
    - HF (TxGemma, MedGemma, BioMistral)
    - Ollama (Qwen / LLaMA etc.)
    """

    # =====================================================
    # OLLAMA BACKEND
    # =====================================================
    @staticmethod
    def _run_ollama(prompt: str, model_name: str) -> str:
        """
        Ollama handles its own memory lifecycle.
        No HF loading involved.
        """

        import requests

        # map logical name → actual ollama model id
        model_id = OLLAMA_MODELS.get(model_name, model_name)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()
            return response.json().get("response", "").strip()

        except Exception as e:
            print(f"[Ollama Error] {model_name}: {e}")
            return "ERROR"

    # =====================================================
    # HF BACKEND (SEQUENTIAL SAFE)
    # =====================================================
    @staticmethod
    def _run_hf(model_name: str, prompt: str) -> str:
        """
        STRICT MEMORY SAFE MODE:
        - load model
        - run inference
        - unload immediately
        """

        return run_unified_model(
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=64
        )

    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================
    @classmethod
    async def run_async(cls, model_name: str, input_data: Any) -> Optional[str]:
        """
        Async wrapper (pipeline compatibility only).
        Execution is still strictly sequential per model.
        """

        backend = MODEL_BACKENDS.get(model_name)

        if backend == "ollama":
            return cls._run_ollama(input_data, model_name)

        if backend == "hf":
            return cls._run_hf(model_name, input_data)

        raise ValueError(f"Unknown backend for model: {model_name}")