"""
Centralized model loading for LLM Council
SEQUENTIAL EXECUTION MODE

HF models:
- txgemma
- biomistral
- medgemma

Ollama models are NOT handled here.
They are handled inside model_router.py.
"""

from typing import Any, Tuple
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import HF_MODELS


# =========================================================
# DEVICE
# =========================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()

def run_unified_model(model_name: str, prompt: str, max_new_tokens: int = 64) -> str:
    """
    TRUE DISPATCH LAYER (SEQUENTIAL SAFE, NO CACHING, NO RECURSION)

    - HF models: load → generate → unload immediately
    - Ollama models: API call
    """

    from .config import MODEL_BACKENDS, OLLAMA_MODELS
    from .model_loader import load_hf_model, DEVICE
    import torch
    import gc
    import requests

    backend = MODEL_BACKENDS.get(model_name)

    # =====================================================
    # HF MODELS (TxGemma, BioMistral)
    # =====================================================
    if backend == "hf":

        model, tokenizer = load_hf_model(model_name)

        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False
                )

            input_len = inputs["input_ids"].shape[1]
            result = tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )

            return result.strip()

        finally:
            # CRITICAL MEMORY CLEANUP
            del model
            del tokenizer
            gc.collect()

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    # =====================================================
    # OLLAMA MODELS (Qwen, MedGemma)
    # =====================================================
    if backend == "ollama":

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
            return None

    raise ValueError(f"Unknown backend for model: {model_name}")
# =========================================================
# LOAD HF MODEL
# =========================================================

def load_hf_model(model_name: str):
    """
    STRICT HF LOADER (SEQUENTIAL SAFE)

    MUST always return (model, tokenizer)
    NEVER returns None silently
    """

    from .config import HF_MODELS
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    if model_name not in HF_MODELS:
        raise ValueError(f"Unknown HF model: {model_name}")

    model_id = HF_MODELS[model_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        model.to(DEVICE)
        model.eval()

        return model, tokenizer

    except Exception as e:
        print(f"[HF LOAD ERROR] {model_name}: {e}")
        raise  # 🔥 IMPORTANT: DO NOT SILENCE FAILURES


# =========================================================
# RUN INFERENCE
# =========================================================

def run_loaded_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]

    generated = outputs[0][input_len:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return response.strip()


# =========================================================
# UNLOAD HF MODEL
# =========================================================

def unload_model(model=None, tokenizer=None):

    print("[UNLOAD] Releasing HF model")

    try:
        if model is not None:
            model.to("cpu")
    except Exception:
        pass

    try:
        del model
    except Exception:
        pass

    try:
        del tokenizer
    except Exception:
        pass

    gc.collect()

    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# =========================================================
# DEVICE INFO
# =========================================================

def get_device_info():

    return {
        "device": str(DEVICE),
        "mps_available": torch.backends.mps.is_available()
    }