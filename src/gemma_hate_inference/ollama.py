from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

import requests

from gemma_hate_inference.config import Settings


T = TypeVar("T")


@dataclass(frozen=True)
class InferenceResult:
    value: Any
    raw_response: str
    attempts: int
    elapsed_seconds: float


def extract_json(text: str) -> dict[str, Any]:
    cleaned = str(text).strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"JSON não encontrado na resposta: {cleaned[:200]}")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("A resposta JSON não é um objeto.")
    return parsed


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        validator: Callable[[dict[str, Any]], T],
        response_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> InferenceResult:
        options: dict[str, Any] = {
            "temperature": 0,
            "num_ctx": self.settings.ollama_num_ctx,
            "num_predict": num_predict or self.settings.ollama_num_predict,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
        if self.settings.ollama_num_gpu is not None:
            options["num_gpu"] = self.settings.ollama_num_gpu

        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": response_schema or "json",
            "keep_alive": -1,
            "options": options,
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(1, self.settings.ollama_max_retries + 1):
            try:
                response = self.session.post(
                    self.settings.ollama_url,
                    json=payload,
                    timeout=self.settings.ollama_timeout_seconds,
                )
                if not response.ok:
                    raise RuntimeError(
                        f"Ollama HTTP {response.status_code}: {response.text[:1000]}"
                    )
                body = response.json()
                last_raw = str(body["response"])
                value = validator(extract_json(last_raw))
                return InferenceResult(
                    value=value,
                    raw_response=last_raw,
                    attempts=attempt,
                    elapsed_seconds=time.perf_counter() - started,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.ollama_max_retries:
                    time.sleep(self.settings.ollama_retry_delay_seconds)

        elapsed = time.perf_counter() - started
        error = RuntimeError(
            f"Falha após {self.settings.ollama_max_retries} tentativas em "
            f"{elapsed:.1f}s: {last_error}"
        )
        error.raw_response = last_raw  # type: ignore[attr-defined]
        error.attempts = self.settings.ollama_max_retries  # type: ignore[attr-defined]
        error.elapsed_seconds = elapsed  # type: ignore[attr-defined]
        raise error


class InferenceClient(Protocol):
    def generate(
        self,
        prompt: str,
        validator: Callable[[dict[str, Any]], T],
        response_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> InferenceResult: ...


class HuggingFaceClient:
    """Cliente local do Transformers; carregado somente quando solicitado."""

    def __init__(self, settings: Settings):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Backend Hugging Face requer torch, transformers e accelerate. "
                "Instale: pip install -r requirements-huggingface.txt"
            ) from exc

        self.settings = settings
        self.torch = torch
        load_options: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
        if settings.hf_token:
            load_options["token"] = settings.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(settings.hf_model, **load_options)
        self.model = AutoModelForCausalLM.from_pretrained(settings.hf_model, **load_options)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        validator: Callable[[dict[str, Any]], T],
        response_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> InferenceResult:
        del response_schema  # O contrato JSON já está explícito no prompt.
        started = time.perf_counter()
        last_error: Exception | None = None
        last_raw = ""
        max_new_tokens = num_predict or self.settings.ollama_num_predict

        for attempt in range(1, self.settings.ollama_max_retries + 1):
            try:
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = inputs.to(self.model.device)
                with self.torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generated = output[0][inputs["input_ids"].shape[1] :]
                last_raw = self.tokenizer.decode(generated, skip_special_tokens=True)
                value = validator(extract_json(last_raw))
                return InferenceResult(
                    value=value,
                    raw_response=last_raw,
                    attempts=attempt,
                    elapsed_seconds=time.perf_counter() - started,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.ollama_max_retries:
                    time.sleep(self.settings.ollama_retry_delay_seconds)

        elapsed = time.perf_counter() - started
        error = RuntimeError(
            f"Falha Hugging Face após {self.settings.ollama_max_retries} tentativas em "
            f"{elapsed:.1f}s: {last_error}"
        )
        error.raw_response = last_raw  # type: ignore[attr-defined]
        error.attempts = self.settings.ollama_max_retries  # type: ignore[attr-defined]
        error.elapsed_seconds = elapsed  # type: ignore[attr-defined]
        raise error


def create_inference_client(settings: Settings) -> InferenceClient:
    if settings.inference_backend == "huggingface":
        return HuggingFaceClient(settings)
    return OllamaClient(settings)
