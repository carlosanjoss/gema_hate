from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOCIAL_PIPELINE_ENV = PROJECT_DIR.parent / "social_data_pipeline" / ".env"


def _load_environment() -> None:
    load_dotenv(PROJECT_DIR / ".env", override=False)
    load_dotenv(SOCIAL_PIPELINE_ENV, override=False)


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} deve ser maior que zero.")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} não pode ser negativo.")
    return value


def _optional_non_negative_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} não pode ser negativo.")
    return value


@dataclass(frozen=True)
class Settings:
    postgres_uri: str
    inference_backend: str
    ollama_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_max_retries: int
    ollama_retry_delay_seconds: int
    ollama_num_ctx: int
    ollama_num_predict: int
    ollama_detail_num_predict: int
    ollama_num_gpu: int
    prompt_version: str
    max_text_chars: int
    database_batch_size: int
    inference_batch_size: int
    inference_batch_max_chars: int
    max_consecutive_errors: int
    hf_model: str
    hf_token: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_environment()
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise RuntimeError(
                "POSTGRES_URI não foi definida no .env deste projeto nem em "
                "../social_data_pipeline/.env."
            )

        inference_backend = os.getenv("INFERENCE_BACKEND", "ollama").strip().lower()
        if inference_backend not in {"ollama", "huggingface"}:
            raise ValueError("INFERENCE_BACKEND deve ser 'ollama' ou 'huggingface'.")
        hf_model = os.getenv("HF_MODEL", "google/gemma-3-4b-it")

        return cls(
            postgres_uri=postgres_uri,
            inference_backend=inference_backend,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            ollama_model=(
                hf_model
                if inference_backend == "huggingface"
                else os.getenv("OLLAMA_MODEL", "gemma3:4b")
            ),
            ollama_timeout_seconds=_positive_int("OLLAMA_TIMEOUT_SECONDS", 300),
            ollama_max_retries=_positive_int("OLLAMA_MAX_RETRIES", 3),
            ollama_retry_delay_seconds=_positive_int("OLLAMA_RETRY_DELAY_SECONDS", 5),
            ollama_num_ctx=_positive_int("OLLAMA_NUM_CTX", 2304),
            ollama_num_predict=_positive_int("OLLAMA_NUM_PREDICT", 300),
            ollama_detail_num_predict=_positive_int(
                "OLLAMA_DETAIL_NUM_PREDICT", 512
            ),
            ollama_num_gpu=_non_negative_int("OLLAMA_NUM_GPU", 999),
            prompt_version=os.getenv(
                "PROMPT_VERSION", "gemma3_4b_structured_v2"
            ),
            max_text_chars=_positive_int("MAX_TEXT_CHARS", 1800),
            database_batch_size=_positive_int("DATABASE_BATCH_SIZE", 25),
            inference_batch_size=_positive_int("INFERENCE_BATCH_SIZE", 16),
            inference_batch_max_chars=_positive_int(
                "INFERENCE_BATCH_MAX_CHARS", 3600
            ),
            max_consecutive_errors=_positive_int("MAX_CONSECUTIVE_ERRORS", 25),
            hf_model=hf_model,
            hf_token=os.getenv("HF_TOKEN") or None,
        )
