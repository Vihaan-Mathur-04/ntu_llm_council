import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

class ModelManager:
    _models = {}

    # -------------------------
    # MEDGEMMA
    # -------------------------
    @classmethod
    def get_medgemma(cls):
        if "medgemma" not in cls._models:
            print("Loading MedGemma 4B IT...")

            processor = AutoProcessor.from_pretrained(
                "google/medgemma-4b-it"
            )

            model = AutoModelForCausalLM.from_pretrained(
                "google/medgemma-4b-it",
                device_map="auto",
                torch_dtype=torch.float16
            )

            cls._models["medgemma"] = {
                "processor": processor,
                "model": model
            }

        return cls._models["medgemma"]

    # -------------------------
    # TXGEMMA
    # -------------------------
    @classmethod
    def get_txgemma(cls):
        if "txgemma" not in cls._models:
            print("Loading TxGemma 9B...")

            tokenizer = AutoTokenizer.from_pretrained(
                "google/txgemma-9b-chat"
            )

            model = AutoModelForCausalLM.from_pretrained(
                "google/txgemma-9b-chat",
                device_map="auto",
                torch_dtype=torch.float16
            )

            cls._models["txgemma"] = {
                "tokenizer": tokenizer,
                "model": model
            }

        return cls._models["txgemma"]

    # -------------------------
    # SCBERT (placeholder loader)
    # -------------------------
    @classmethod
    def get_scbert(cls):
        if "scbert" not in cls._models:
            print("Loading scBERT...")

            # IMPORTANT: you will replace this based on repo code
            model = None  

            cls._models["scbert"] = model

        return cls._models["scbert"]