from gemma_hate_inference.config import Settings


def test_default_ollama_model_is_gemma3_4b(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://example.invalid/test")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_GPU", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("OLLAMA_DETAIL_NUM_PREDICT", raising=False)
    monkeypatch.delenv("PROMPT_VERSION", raising=False)

    settings = Settings.from_env()

    assert settings.ollama_model == "gemma3:4b"
    assert settings.ollama_num_gpu == 999
    assert settings.ollama_num_ctx == 2304
    assert settings.ollama_detail_num_predict == 512
    assert settings.inference_batch_size == 16
    assert settings.max_consecutive_errors == 25
    assert settings.prompt_version == "gemma3_4b_structured_v2"
