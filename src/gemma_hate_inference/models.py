from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gemma_hate_inference.taxonomy import CATEGORIES


PREDICTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "h": {"type": "integer", "enum": [0, 1], "description": "pred_hate"},
        "p": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "hate_probability",
        },
        "c": {
            "description": "pred_category",
            "anyOf": [
                {"type": "string", "enum": list(CATEGORIES)},
                {
                    "type": "array",
                    "items": {"type": "string", "enum": list(CATEGORIES)},
                    "minItems": 1,
                    "maxItems": 4,
                    "uniqueItems": True,
                },
                {"type": "null"},
            ]
        },
        "q": {
            "description": "category_probability",
            "anyOf": [
                {"type": "number", "minimum": 0, "maximum": 1},
                {"type": "null"},
            ]
        },
        "t": {
            "type": ["string", "null"],
            "maxLength": 100,
            "description": "alvo_identificado",
        },
        "e": {
            "type": ["string", "null"],
            "maxLength": 180,
            "description": "evidencia_textual",
        },
        "j": {
            "type": "string",
            "minLength": 1,
            "maxLength": 220,
            "description": "justificativa_curta",
        },
    },
    "required": ["h", "p", "c", "q", "t", "e", "j"],
    "additionalProperties": False,
}

BINARY_PREDICTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "h": {"type": "integer", "enum": [0, 1], "description": "pred_hate"},
        "p": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "hate_probability",
        },
        "j": {
            "type": "string",
            "enum": [
                "no_group_attack",
                "reported_or_discussed_hate",
                "individual_offense",
                "ordinary_political_criticism",
                "profanity_without_group",
                "neutral_or_irrelevant",
                "potential_group_attack",
            ],
            "description": "codigo_da_justificativa",
        },
    },
    "required": ["h", "p", "j"],
    "additionalProperties": False,
}

BINARY_JUSTIFICATIONS = {
    "no_group_attack": "Não há ataque direcionado a grupo social ou identitário.",
    "reported_or_discussed_hate": "O texto relata ou discute preconceito sem endossá-lo.",
    "individual_offense": "Há ofensa individual sem ataque claro a grupo social.",
    "ordinary_political_criticism": "Há crítica política comum sem hostilidade generalizante.",
    "profanity_without_group": "Há linguagem ofensiva, mas sem alvo de grupo social.",
    "neutral_or_irrelevant": "O texto é neutro ou irrelevante para discurso de ódio.",
    "potential_group_attack": "Possível ataque a grupo; encaminhado à categorização detalhada.",
}


def binary_batch_json_schema(batch_size: int) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")
    return {
        "type": "object",
        "properties": {
            "r": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": BINARY_PREDICTION_JSON_SCHEMA,
            }
        },
        "required": ["r"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class CategoryPrediction:
    code: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    is_hate: bool
    hate_probability: float
    categories: tuple[CategoryPrediction, ...]
    target_identified: str | None
    textual_evidence: str | None
    justification: str


def clamp_probability(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(min(max(number, 0.0), 1.0), 6)


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "n/a", "não se aplica", "nao se aplica"}:
        return None
    return text


def _parse_categories(
    obj: dict[str, Any], is_hate: bool, hate_probability: float
) -> tuple[CategoryPrediction, ...]:
    if not is_hate:
        return ()

    raw = obj.get("pred_category", obj.get("categories"))
    default_probability = clamp_probability(
        obj.get("category_probability"), hate_probability
    )
    parsed: dict[str, float] = {}

    if isinstance(raw, str):
        values: list[Any] = raw.replace(",", ";").split(";")
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    elif isinstance(raw, dict):
        values = [{"code": code, "probability": probability} for code, probability in raw.items()]
    else:
        values = []

    for item in values:
        if isinstance(item, dict):
            code = str(item.get("code", item.get("category", ""))).strip().lower()
            probability = clamp_probability(item.get("probability"), default_probability)
        else:
            code = str(item).strip().lower()
            probability = default_probability
        if code in CATEGORIES:
            parsed[code] = max(parsed.get(code, 0.0), probability)

    if not parsed:
        parsed["other"] = default_probability

    return tuple(
        CategoryPrediction(code=code, probability=probability)
        for code, probability in sorted(parsed.items())
    )


def normalize_prediction(obj: dict[str, Any]) -> Prediction:
    if not isinstance(obj, dict):
        raise ValueError("A resposta do modelo deve ser um objeto JSON.")

    if "h" in obj:
        try:
            compact_is_hate = int(obj.get("h"))
        except (TypeError, ValueError) as exc:
            raise ValueError("h deve ser 0 ou 1.") from exc
        if compact_is_hate not in {0, 1}:
            raise ValueError("h deve ser 0 ou 1.")
        justification = obj.get("j")
        justification = BINARY_JUSTIFICATIONS.get(str(justification), justification)
        obj = {
            "pred_label": "hate" if compact_is_hate else "nao_hate",
            "pred_hate": compact_is_hate,
            "hate_probability": obj.get("p"),
            "pred_category": obj.get("c"),
            "category_probability": obj.get("q"),
            "alvo_identificado": obj.get("t"),
            "evidencia_textual": obj.get("e"),
            "justificativa_curta": justification,
        }

    label = str(obj.get("pred_label", "")).strip().lower()
    label = label.replace("-", "_").replace(" ", "_")
    if label == "hate":
        is_hate = True
    elif label in {"nao_hate", "não_hate"}:
        is_hate = False
    else:
        raise ValueError("pred_label deve ser 'hate' ou 'nao_hate'.")

    expected_pred_hate = 1 if is_hate else 0
    raw_pred_hate = obj.get("pred_hate")
    if raw_pred_hate is not None:
        try:
            parsed_pred_hate = int(raw_pred_hate)
        except (TypeError, ValueError) as exc:
            raise ValueError("pred_hate deve ser 0 ou 1.") from exc
        if parsed_pred_hate != expected_pred_hate:
            raise ValueError("pred_label e pred_hate são inconsistentes.")

    hate_probability = clamp_probability(
        obj.get("hate_probability"), 0.70 if is_hate else 0.30
    )
    if not is_hate and hate_probability > 0.49:
        hate_probability = 0.40

    justification = _clean_optional(obj.get("justificativa_curta"))
    if not justification:
        raise ValueError("A resposta não contém justificativa_curta.")

    return Prediction(
        is_hate=is_hate,
        hate_probability=hate_probability,
        categories=_parse_categories(obj, is_hate, hate_probability),
        target_identified=_clean_optional(obj.get("alvo_identificado")),
        textual_evidence=_clean_optional(obj.get("evidencia_textual")),
        justification=justification,
    )


def normalize_binary_batch(
    obj: dict[str, Any], expected_count: int
) -> tuple[Prediction, ...]:
    if not isinstance(obj, dict):
        raise ValueError("A resposta em lote deve ser um objeto JSON.")
    rows = obj.get("r")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(
            f"O lote deve conter exatamente {expected_count} resultados em r."
        )
    return tuple(normalize_prediction(row) for row in rows)
