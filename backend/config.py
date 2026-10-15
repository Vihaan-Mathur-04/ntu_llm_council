"""
Configuration for the LLM Council (CLEAN + CELLSTAR READY VERSION).
"""

from dotenv import load_dotenv

load_dotenv()


# =========================================================
# DATA PATHS
# =========================================================

DATA_DIR = "data/conversations"


# =========================================================
# OLLAMA MODELS (LOCAL LIGHTWEIGHT MODELS)
# =========================================================

OLLAMA_MODELS = {
    "medgemma": "medgemma",
    "phi": "phi4-mini:3.8b",
    "llama": "llama3.1:8b"
    }


# =========================================================
# HUGGINGFACE MODELS (HEAVY SCIENTIFIC MODELS)
# =========================================================

HF_MODELS = {
    "txgemma":"google/txgemma-9b-chat"
}


# =========================================================
# COUNCIL MEMBERS (ACTIVE VOTERS)
# =========================================================

COUNCIL_MEMBERS = [
    "txgemma",
    "llama",
    "phi"
]


# =========================================================
# CHAIRMAN MODEL (FINAL SYNTHESIS)
# =========================================================

CHAIRMAN_MODEL = "medgemma"


# =========================================================
# EXECUTION BACKENDS
# =========================================================

MODEL_BACKENDS = {
    # Ollama-backed models
    "phi": "ollama",        # ✅ FIXED
    "medgemma": "ollama",

    # HF-backed models
    "llama": "ollama",
    "txgemma": "hf"
}


# =========================================================
# MODEL ROLES (PROMPT ENGINEERING ONLY)
# =========================================================

MODEL_ROLES = {
    "phi": "simple_reasoner",
    "llama": "strong_reasoner",
    "txgemma": "biological_expert",
    "medgemma": "chairman_synthesizer"
}


# =========================================================
# CELLSTAR / RAG SETTINGS (FUTURE READY)
# =========================================================

CELLSTAR_SETTINGS = {
    "use_marker_gene_context": True,
    "max_marker_genes": 30,
    "include_reference_atlas": True
}