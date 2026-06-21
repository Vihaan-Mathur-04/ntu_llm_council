"""
Model routing layer
(NTU Council - Sequential Safe Version)
"""

from typing import Any, Optional

from .model_loader import run_unified_model
from .config import MODEL_BACKENDS, OLLAMA_MODELS


# =====================================================
# MODEL ROUTER
# =====================================================

class ModelRouter:
    """
    Unified inference interface.

    BACKENDS:
    - hf
    - ollama

    Memory-safe:
    Models are loaded, used, and released
    by model_loader.py
    """

    # =====================================================
    # OLLAMA BACKEND
    # =====================================================
    @staticmethod
    def _run_ollama(
        prompt: str,
        model_name: str,
        max_new_tokens: int = 64
    ) -> str:

        import requests
        import traceback

        model_id = OLLAMA_MODELS.get(
            model_name,
            model_name
        )

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0,
                        "top_p": 1
                    }
                },
                timeout=300
            )

            response.raise_for_status()

            data = response.json()

            print("\n====================")
            print(f"MODEL: {model_name}")
            print("====================")
            print(data)
            print("====================\n")

            # -------------------------
            # Standard Ollama output
            # -------------------------
            if (
                "response" in data
                and data["response"]
            ):
                return data["response"].strip()

            # -------------------------
            # Chat-style output
            # -------------------------
            if (
                "message" in data
                and isinstance(data["message"], dict)
            ):
                content = data["message"].get(
                    "content",
                    ""
                )

                if content:
                    return content.strip()

            return ""

        except Exception:

            print(
                f"\n[OLLAMA ERROR - {model_name}]"
            )

            traceback.print_exc()

            return "ERROR"

    # =====================================================
    # HF BACKEND
    # =====================================================
    @staticmethod
    def _run_hf(
        model_name: str,
        prompt: str
    ) -> str:

        return run_unified_model(
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=64
        )

    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================
    @classmethod
    async def run_async(
        cls,
        model_name: str,
        input_data: Any
    ) -> Optional[str]:

        backend = MODEL_BACKENDS.get(
            model_name
        )

        if backend == "ollama":

            return cls._run_ollama(
                input_data,
                model_name
            )

        if backend == "hf":

            return cls._run_hf(
                model_name,
                input_data
            )

        raise ValueError(
            f"Unknown backend for model: {model_name}"
        )