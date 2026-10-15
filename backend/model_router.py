"""
Model routing layer
(NTU Council - Sequential Safe Version)
"""

from typing import Any, Optional

from .model_loader import run_unified_model
from .config import MODEL_BACKENDS, OLLAMA_MODELS


# =====================================================
# TOKEN BUDGETS
# =====================================================

HF_TOKEN_BUDGETS = {
    "phi": 750,
    "llama": 750,
    "txgemma": 750,
    "medgemma": 1800,      # Chairman
}

DEFAULT_HF_TOKENS = 700


# =====================================================
# MODEL ROUTER
# =====================================================

class ModelRouter:

    # =====================================================
    # OLLAMA BACKEND
    # =====================================================

    @staticmethod
    def _run_ollama(
        prompt: str,
        model_name: str
    ) -> str:

        import requests
        import traceback

        model_id = OLLAMA_MODELS.get(model_name, model_name)

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1000,
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

            if "response" in data and data["response"]:
                return data["response"].strip()

            if (
                "message" in data
                and isinstance(data["message"], dict)
            ):
                content = data["message"].get("content", "")
                if content:
                    return content.strip()

            return ""

        except Exception:

            print(f"\n[OLLAMA ERROR - {model_name}]")

            traceback.print_exc()

            return "ERROR"

    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================

    @classmethod
    async def run_async(
        cls,
        model_name: str,
        input_data: Any
    ) -> Optional[str]:

        backend = MODEL_BACKENDS.get(model_name)

        # ----------------------------
        # Ollama
        # ----------------------------

        if backend == "ollama":

            return cls._run_ollama(
                input_data,
                model_name
            )

        # ----------------------------
        # HuggingFace
        # ----------------------------

        if backend == "hf":

            token_budget = HF_TOKEN_BUDGETS.get(
                model_name,
                DEFAULT_HF_TOKENS
            )

            print(
                f"[TOKEN BUDGET] "
                f"{model_name}: {token_budget}"
            )

            return run_unified_model(
                model_name=model_name,
                prompt=input_data,
                max_new_tokens=token_budget
            )

        raise ValueError(
            f"Unknown backend for model: {model_name}"
        )