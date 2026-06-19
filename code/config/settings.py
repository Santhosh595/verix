"""
Central run configuration. Reads secrets from environment variables only
(never hardcode keys). See .env.example for the variables this expects.
"""

import os
from pathlib import Path

# Load .env file if it exists, so env vars are available when this module is imported
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Repo-relative paths -- keep these as the single source of truth so
# every module resolves dataset/output paths the same way.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_SAMPLE_DIR = DATASET_DIR / "images" / "sample"
IMAGES_TEST_DIR = DATASET_DIR / "images" / "test"

OUTPUT_CSV = REPO_ROOT / "code" / "output.csv"
EVAL_REPORT_MD = REPO_ROOT / "code" / "evaluation" / "evaluation_report.md"
CACHE_DIR = REPO_ROOT / "code" / ".cache"

# Provider/model selection.
# Supported: gemini | groq | openai | anthropic | openrouter
# For OpenRouter, use :free model suffixes for $0 cost.
# Recommended free models:
#   openai/gpt-oss-120b:free
#   deepseek/deepseek-chat:free
#   qwen/qwen3-coder:free
#   meta-llama/llama-3.3-70b-instruct:free
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "openrouter")
VLM_MODEL = os.getenv("VLM_MODEL", "openai/gpt-oss-120b:free")
VLM_FALLBACK_MODEL = os.getenv("VLM_FALLBACK_MODEL", "deepseek/deepseek-chat:free")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Determinism / rate-limit knobs. TODO: tune once provider is picked.
TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

# Simple TPM/RPM throttling knobs -- wire into utils/caching.py or a
# dedicated rate limiter in core/vision_analyzer.py.
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
