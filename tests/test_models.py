from __future__ import annotations

import pytest

from gemma_hate_inference.models import (
    PREDICTION_JSON_SCHEMA,
    binary_batch_json_schema,
    normalize_binary_batch,
    normalize_prediction,
)
from gemma_hate_inference.taxonomy import CATEGORIES


def test_non_hate_has_no_category() -> None:
    prediction = normalize_prediction(
        {
            "pred_label": "nao_hate",
            "pred_hate": 0,
            "hate_probability": 0.2,
            "pred_category": None,
            "category_probability": None,
            "alvo_identificado": None,
            "evidencia_textual": "trecho que denuncia preconceito",
            "justificativa_curta": "O texto denuncia preconceito sem endossá-lo.",
        }
    )

    assert prediction.is_hate is False
    assert prediction.hate_probability == 0.2
    assert prediction.categories == ()


def test_hate_supports_one_or_more_categories_from_prompt_format() -> None:
    prediction = normalize_prediction(
        {
            "pred_label": "hate",
            "pred_hate": 1,
            "hate_probability": 0.91,
            "pred_category": "misogyny; lgbtphobia",
            "category_probability": 0.84,
            "alvo_identificado": "mulheres LGBT+",
            "evidencia_textual": "trecho",
            "justificativa_curta": "Ataque baseado em gênero e sexualidade.",
        }
    )

    assert prediction.is_hate is True
    assert [category.code for category in prediction.categories] == [
        "lgbtphobia",
        "misogyny",
    ]
    assert all(category.probability == 0.84 for category in prediction.categories)


def test_unknown_hate_category_falls_back_to_other() -> None:
    prediction = normalize_prediction(
        {
            "pred_label": "hate",
            "pred_hate": 1,
            "pred_category": "categoria_inventada",
            "justificativa_curta": "Há ataque identitário fora das categorias específicas.",
        }
    )
    assert prediction.categories[0].code == "other"


def test_inconsistent_binary_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistentes"):
        normalize_prediction(
            {
                "pred_label": "hate",
                "pred_hate": 0,
                "justificativa_curta": "Resposta inconsistente.",
            }
        )


def test_justification_is_required() -> None:
    with pytest.raises(ValueError, match="justificativa_curta"):
        normalize_prediction({"pred_label": "nao_hate", "pred_hate": 0})


def test_compact_wire_format_is_normalized() -> None:
    prediction = normalize_prediction(
        {
            "h": 1,
            "p": 0.92,
            "c": ["misogyny", "lgbtphobia"],
            "q": 0.87,
            "t": "mulheres LGBT+",
            "e": "trecho",
            "j": "potential_group_attack",
        }
    )

    assert prediction.is_hate is True
    assert [category.code for category in prediction.categories] == [
        "lgbtphobia",
        "misogyny",
    ]


def test_binary_reason_code_becomes_auditable_justification() -> None:
    prediction = normalize_prediction(
        {"h": 0, "p": 0.1, "j": "no_group_attack"}
    )

    assert prediction.justification == (
        "Não há ataque direcionado a grupo social ou identitário."
    )


def test_binary_batch_schema_and_normalization_preserve_order() -> None:
    schema = binary_batch_json_schema(2)
    assert schema["properties"]["r"]["minItems"] == 2
    assert schema["properties"]["r"]["maxItems"] == 2

    predictions = normalize_binary_batch(
        {
            "r": [
                {"h": 0, "p": 0.1, "j": "neutral_or_irrelevant"},
                {"h": 1, "p": 0.9, "j": "potential_group_attack"},
            ]
        },
        expected_count=2,
    )
    assert [prediction.is_hate for prediction in predictions] == [False, True]


def test_binary_batch_rejects_wrong_result_count() -> None:
    with pytest.raises(ValueError, match="exatamente 2"):
        normalize_binary_batch(
            {"r": [{"h": 0, "p": 0.1, "j": "neutral_or_irrelevant"}]},
            expected_count=2,
        )


def test_detail_schema_only_accepts_known_categories() -> None:
    category_schema = PREDICTION_JSON_SCHEMA["properties"]["c"]["anyOf"]
    assert category_schema[0]["enum"] == list(CATEGORIES)
    assert category_schema[1]["items"]["enum"] == list(CATEGORIES)
    assert category_schema[1]["maxItems"] == 4
