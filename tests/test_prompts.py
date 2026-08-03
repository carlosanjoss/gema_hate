from __future__ import annotations

from gemma_hate_inference.prompts import (
    build_binary_batch_prompt,
    build_binary_prompt,
    build_detailed_prompt,
    build_prompt,
)


def test_prompt_contains_only_the_two_requested_steps_and_truncates_text() -> None:
    prompt = build_prompt("abcdefghij", max_chars=5)
    assert "ETAPA 1 — Classificação binária" in prompt
    assert "ETAPA 2 — Categoria" in prompt
    assert "CAMADA 3" not in prompt
    assert "hostility" not in prompt
    assert "dimensions" not in prompt
    assert '"h": 1' in prompt
    assert '"h": 0' in prompt
    assert "sempre as sete chaves compactas" in prompt
    assert '"""abcde"""' in prompt
    assert "abcdefghij" not in prompt


def test_binary_prompt_keeps_rules_but_requests_only_first_stage() -> None:
    prompt = build_binary_prompt("texto")

    assert "DECISÃO BINÁRIA OBRIGATÓRIA" in prompt
    assert "movimentos identitários" in prompt
    assert "manosfera/red pill" in prompt
    assert "Hostilidade generalizante contra grupo político" in prompt
    assert "Retorne somente h, p e j" in prompt
    assert "potential_group_attack" in prompt
    assert '"""texto"""' in prompt


def test_binary_batch_prompt_keeps_rubric_and_order_contract() -> None:
    prompt = build_binary_batch_prompt(["primeiro", "segundo"])

    assert "movimentos identitários" in prompt
    assert "exatamente 2 resultados" in prompt
    assert 'ITEM 0:\n"""primeiro"""' in prompt
    assert 'ITEM 1:\n"""segundo"""' in prompt


def test_detailed_prompt_keeps_binary_rules_and_complete_output() -> None:
    prompt = build_detailed_prompt("texto")

    assert "movimentos identitários" in prompt
    assert "manosfera/red pill" in prompt
    assert "Se h=0: c=null" in prompt
    assert "CATEGORIAS VÁLIDAS PARA h=1" in prompt
    assert "Retorne somente h, p, c, q, t, e e j" in prompt
    assert "nunca copie o texto inteiro" in prompt
    assert '"""texto"""' in prompt
