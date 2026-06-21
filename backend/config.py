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
    "qwen": "qwen3.5:4b"   # ✅ FIXED: new primary lightweight model
}


# =========================================================
# HUGGINGFACE MODELS (HEAVY SCIENTIFIC MODELS)
# =========================================================

HF_MODELS = {
    "meditron": "meditron:7b"
}


# =========================================================
# COUNCIL MEMBERS (ACTIVE VOTERS)
# =========================================================

COUNCIL_MEMBERS = [
    "qwen",
    "meditron"
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
    "qwen": "ollama",        # ✅ FIXED
    "medgemma": "ollama",

    # HF-backed models
    "meditron": "ollama"
}


# =========================================================
# MODEL ROLES (PROMPT ENGINEERING ONLY)
# =========================================================

MODEL_ROLES = {
    "qwen": "general_reasoner",
    "meditron": "clinical_reasoner",
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