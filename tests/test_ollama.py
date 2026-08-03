from __future__ import annotations

from types import SimpleNamespace

from gemma_hate_inference.models import PREDICTION_JSON_SCHEMA
from gemma_hate_inference.ollama import OllamaClient, extract_json


def _settings():
    return SimpleNamespace(
        ollama_model="gemma3:4b",
        ollama_url="http://localhost:11434/api/generate",
        ollama_timeout_seconds=30,
        ollama_max_retries=1,
        ollama_retry_delay_seconds=1,
        ollama_num_ctx=4096,
        ollama_num_predict=300,
        ollama_num_gpu=0,
    )


def test_extract_json_accepts_markdown_fence_and_surrounding_text() -> None:
    parsed = extract_json('resposta\n```json\n{"pred_label":"hate"}\n```')
    assert parsed == {"pred_label": "hate"}


def test_generate_sends_prediction_schema_when_provided() -> None:
    client = OllamaClient(_settings())
    captured = {}

    class FakeResponse:
        ok = True

        @staticmethod
        def json():
            return {"response": '{"pred_label":"nao_hate"}'}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    client.session = SimpleNamespace(post=fake_post)
    result = client.generate(
        "prompt",
        lambda obj: obj,
        response_schema=PREDICTION_JSON_SCHEMA,
    )

    assert captured["format"] == PREDICTION_JSON_SCHEMA
    assert result.value == {"pred_label": "nao_hate"}


def test_generate_keeps_free_json_for_preflight() -> None:
    client = OllamaClient(_settings())
    captured = {}

    class FakeResponse:
        ok = True

        @staticmethod
        def json():
            return {"response": '{"status":"ok"}'}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    client.session = SimpleNamespace(post=fake_post)
    client.generate("preflight", lambda obj: obj)

    assert captured["format"] == "json"


def test_generate_lets_ollama_choose_gpu_when_not_forced() -> None:
    settings = _settings()
    settings.ollama_num_gpu = None
    client = OllamaClient(settings)
    captured = {}

    class FakeResponse:
        ok = True

        @staticmethod
        def json():
            return {"response": '{"status":"ok"}'}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    client.session = SimpleNamespace(post=fake_post)
    client.generate("preflight", lambda obj: obj)

    assert "num_gpu" not in captured["options"]
    assert captured["keep_alive"] == -1
