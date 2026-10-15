"""
Centralized model loading for LLM Council
SEQUENTIAL EXECUTION MODE

HF models:
- txgemma
- llama
- medgemma

Ollama models are NOT handled here.
They are handled inside model_router.py.
"""

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
# =========================================================
# HF MODEL CACHE
# =========================================================

MODEL_CACHE = {}
TOKENIZER_CACHE = {}

def run_unified_model(model_name: str, prompt: str, max_new_tokens: int = 400) -> str:
    """
    TRUE DISPATCH LAYER (SEQUENTIAL SAFE, NO CACHING, NO RECURSION)

    - HF models: load → generate → unload immediately
    - Ollama models: API call
    """

    from .config import MODEL_BACKENDS, OLLAMA_MODELS
    import torch
    import gc
    import requests

    backend = MODEL_BACKENDS.get(model_name)

    # =====================================================
    # HF MODELS (TxGemma, BioMistral)
    # =====================================================
    if backend == "hf":

        model, tokenizer = load_hf_model(model_name)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            max_length=2048
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]

        generated = outputs[0][input_len:]

        result = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        print("\n[HF RAW OUTPUT]")
        print(result)
        print("====================\n")

        print("=" * 80)
        print(f"MODEL: {model_name}")
        print(f"INPUT TOKENS : {input_len}")
        print(f"OUTPUT TOKENS: {generated.shape[-1]}")
        print(f"TOTAL TOKENS : {input_len + generated.shape[-1]}")
        print("=" * 80)
        if generated.shape[-1] == 0:
            print(f"[WARNING] {model_name} generated ZERO new tokens.")

        print("=" * 80)
        print(model_name)
        print("Prompt tokens:", input_len)
        print("Generated tokens:", outputs.shape[-1])
        print("New tokens:", generated.shape[-1])
        print("=" * 80)

        return result

    # =====================================================
    # OLLAMA MODELS (Phi, MedGemma)
    # =====================================================
    if backend == "ollama":

        model_id = OLLAMA_MODELS.get(model_name, model_name)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 1024,
                        "temperature": 0,
                        "top_p": 1
                    }
                },
                timeout=300
            )
            response.raise_for_status()

            data = response.json()

            print(data.keys())

            response_text = data.get("response", "")

            message_text = ""
            if "message" in data and isinstance(data["message"], dict):
                message_text = data["message"].get("content", "")

            thinking_text = data.get("thinking", "")

            content = "\n".join(
                x for x in [
                    response_text,
                    message_text,
                    thinking_text
                ] if x
            ).strip()

            print("\n[OLLAMA RAW OUTPUT]")
            print(data)
            print("====================\n")

            return content

        except Exception as e:
            print(f"[Ollama Error] {model_name}: {e}")
            return None

    raise ValueError(f"Unknown backend for model: {model_name}")
# =========================================================
# LOAD HF MODEL
# =========================================================

def load_hf_model(model_name: str):
    """
    Load a Hugging Face model only once.
    Subsequent calls reuse the cached model.
    """

    if model_name not in HF_MODELS:
        raise ValueError(f"Unknown HF model: {model_name}")

    # -------------------------------
    # Already loaded
    # -------------------------------

    if model_name in MODEL_CACHE:

        return (
            MODEL_CACHE[model_name],
            TOKENIZER_CACHE[model_name]
        )

    # -------------------------------
    # First load
    # -------------------------------

    model_id = HF_MODELS[model_name]

    print(f"\n[HF] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(
            torch.float16
            if DEVICE.type == "mps"
            else torch.float32
        ),
        low_cpu_mem_usage=True
    )

    model.to(DEVICE)
    model.eval()

    MODEL_CACHE[model_name] = model
    TOKENIZER_CACHE[model_name] = tokenizer

    print(f"[HF] {model_name} loaded and cached.")

    return model, tokenizer


# =========================================================
# RUN INFERENCE
# =========================================================

def run_loaded_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 400
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
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
            use_cache=True,
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