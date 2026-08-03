from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from gemma_hate_inference.config import Settings
from gemma_hate_inference.database import ContentItem, Database
from gemma_hate_inference.models import (
    BINARY_PREDICTION_JSON_SCHEMA,
    PREDICTION_JSON_SCHEMA,
    binary_batch_json_schema,
    normalize_binary_batch,
    normalize_prediction,
)
from gemma_hate_inference.ollama import (
    InferenceClient,
    InferenceResult,
    create_inference_client,
)
from gemma_hate_inference.platforms import SUPPORTED_PLATFORMS
from gemma_hate_inference.progress import format_progress
from gemma_hate_inference.prompts import (
    build_binary_batch_prompt,
    build_binary_prompt,
    build_detailed_prompt,
)


@dataclass(frozen=True)
class RunOptions:
    content: str = "all"
    platforms: Sequence[str] = SUPPORTED_PLATFORMS
    only_usable: bool = False
    limit: int | None = None
    dry_run: bool = False
    run_name: str | None = None
    skip_migrate: bool = False


def _content_kinds(content: str) -> tuple[str, ...]:
    if content == "all":
        # Posts de mensageria concentram mais duplicatas; processá-los primeiro
        # maximiza a quantidade de linhas concluídas por chamada ao modelo.
        return "post", "comment"
    if content == "posts":
        return ("post",)
    if content == "comments":
        return ("comment",)
    raise ValueError(f"Conteúdo inválido: {content}")


def _selected_counts(
    db: Database,
    settings: Settings,
    kinds: Sequence[str],
    options: RunOptions,
) -> tuple[dict[str, int], int]:
    pending = {
        kind: db.count_pending(
            kind=kind,
            model_name=settings.ollama_model,
            prompt_version=settings.prompt_version,
            platforms=options.platforms,
            only_usable=options.only_usable,
        )
        for kind in kinds
    }
    remaining = options.limit
    selected: dict[str, int] = {}
    for kind in kinds:
        available = pending[kind]
        amount = available if remaining is None else min(available, remaining)
        selected[kind] = amount
        if remaining is not None:
            remaining -= amount
    return selected, sum(selected.values())


def _infer_binary_items(
    client: InferenceClient,
    settings: Settings,
    items: Sequence[ContentItem],
) -> list[InferenceResult | Exception]:
    if len(items) == 1:
        try:
            return [
                client.generate(
                    build_binary_prompt(items[0].text, settings.max_text_chars),
                    normalize_prediction,
                    response_schema=BINARY_PREDICTION_JSON_SCHEMA,
                )
            ]
        except Exception as exc:
            return [exc]

    try:
        batch_result = client.generate(
            build_binary_batch_prompt(
                [item.text for item in items], settings.max_text_chars
            ),
            lambda obj: normalize_binary_batch(obj, len(items)),
            response_schema=binary_batch_json_schema(len(items)),
            num_predict=max(settings.ollama_num_predict, len(items) * 64),
        )
        elapsed_per_item = batch_result.elapsed_seconds / len(items)
        return [
            InferenceResult(
                value=prediction,
                raw_response=batch_result.raw_response,
                attempts=batch_result.attempts,
                elapsed_seconds=elapsed_per_item,
            )
            for prediction in batch_result.value
        ]
    except Exception:
        # Se o contrato do lote falhar, isola cada texto. Assim um único item
        # problemático não invalida os demais nem é convertido em nao_hate.
        results: list[InferenceResult | Exception] = []
        for item in items:
            try:
                results.append(
                    client.generate(
                        build_binary_prompt(item.text, settings.max_text_chars),
                        normalize_prediction,
                        response_schema=BINARY_PREDICTION_JSON_SCHEMA,
                    )
                )
            except Exception as exc:
                results.append(exc)
        return results


def run_inference(settings: Settings, options: RunOptions) -> int:
    db = Database(settings.postgres_uri)
    try:
        if not options.skip_migrate:
            db.migrate()
        kinds = _content_kinds(options.content)
        selected, total = _selected_counts(db, settings, kinds, options)
        print(
            "Selecionados: "
            + ", ".join(f"{kind}={count}" for kind, count in selected.items())
            + f"; total={total}"
        )

        if options.dry_run or total == 0:
            if total == 0:
                print("Não há conteúdo pendente para este modelo/prompt e filtros.")
            return 0

        run_name = options.run_name or datetime.now().astimezone().strftime(
            "gemma_hate_%Y%m%d_%H%M%S_%f"
        )
        filters = {
            "content": options.content,
            "platforms": list(options.platforms),
            "only_usable": options.only_usable,
            "limit": options.limit,
            "skip_migrate": options.skip_migrate,
        }
        run_id = db.create_run(
            run_name=run_name,
            model_name=settings.ollama_model,
            prompt_version=settings.prompt_version,
            filters=filters,
            total_selected=total,
        )
        print(f"Execução iniciada: run_id={run_id}, nome={run_name}", flush=True)
        print(
            f"Andamento será exibido a cada {settings.database_batch_size} itens. "
            "Use `python run.py status` em outro terminal para acompanhar o progresso gravado.",
            flush=True,
        )
        started_at = datetime.now().astimezone()

        success = 0
        errors = 0
        reused = 0
        consecutive_errors = 0
        last_reported = 0
        client = create_inference_client(settings)

        try:
            client.generate(
                'Retorne somente JSON válido, sem markdown: {"status":"ok"}',
                lambda obj: obj,
            )
        except Exception as exc:
            message = f"Falha no preflight do backend/modelo: {exc}"
            db.finish_run(run_id, "failed", success, errors, message, reused=reused)
            print(message)
            return 3

        try:
            for kind in kinds:
                kind_limit = selected[kind]
                kind_processed = 0
                db.prepare_pending_queue(
                    kind=kind,
                    model_name=settings.ollama_model,
                    prompt_version=settings.prompt_version,
                    platforms=options.platforms,
                    only_usable=options.only_usable,
                    limit=kind_limit,
                )
                db.commit()
                queue = iter(
                    db.iter_queued_pending(
                        kind=kind,
                        model_name=settings.ollama_model,
                        prompt_version=settings.prompt_version,
                        page_size=max(settings.database_batch_size, 1000),
                    )
                )
                carry: ContentItem | None = None
                queue_exhausted = False

                while kind_processed < kind_limit:
                    model_items: list[ContentItem] = []
                    model_keys: set[tuple[str, str]] = set()
                    batch_chars = 0

                    while len(model_items) < settings.inference_batch_size:
                        if carry is not None:
                            item = carry
                            carry = None
                        else:
                            try:
                                item = next(queue)
                            except StopIteration:
                                queue_exhausted = True
                                break

                        if kind_processed >= kind_limit:
                            break
                        item_key = (item.text_hash, item.text)
                        if item_key in model_keys:
                            continue
                        # Uma duplicata anterior pode ter preenchido esta linha
                        # depois que a página de pendências foi carregada.
                        if db.has_success(
                            item,
                            settings.ollama_model,
                            settings.prompt_version,
                        ):
                            continue

                        item_chars = min(
                            len(item.text), settings.max_text_chars
                        )
                        if (
                            model_items
                            and batch_chars + item_chars
                            > settings.inference_batch_max_chars
                        ):
                            carry = item
                            break
                        model_items.append(item)
                        model_keys.add(item_key)
                        batch_chars += item_chars

                    if consecutive_errors >= settings.max_consecutive_errors:
                        raise RuntimeError(
                            "Circuit breaker acionado após "
                            f"{consecutive_errors} erros consecutivos. "
                            "Verifique o Ollama e as respostas do modelo antes de retomar."
                        )

                    if not model_items:
                        if queue_exhausted:
                            break
                        continue

                    binary_results = _infer_binary_items(
                        client, settings, model_items
                    )
                    for item, binary_outcome in zip(
                        model_items, binary_results, strict=True
                    ):
                        try:
                            if isinstance(binary_outcome, Exception):
                                raise binary_outcome
                            result = binary_outcome
                            if binary_outcome.value.is_hate:
                                detail_result = client.generate(
                                    build_detailed_prompt(
                                        item.text, settings.max_text_chars
                                    ),
                                    normalize_prediction,
                                    response_schema=PREDICTION_JSON_SCHEMA,
                                    num_predict=settings.ollama_detail_num_predict,
                                )
                                result = InferenceResult(
                                    value=detail_result.value,
                                    raw_response=detail_result.raw_response,
                                    attempts=(
                                        binary_outcome.attempts
                                        + detail_result.attempts
                                    ),
                                    elapsed_seconds=(
                                        binary_outcome.elapsed_seconds
                                        + detail_result.elapsed_seconds
                                    ),
                                )

                            db.save_success(
                                run_id=run_id,
                                item=item,
                                model_name=settings.ollama_model,
                                prompt_version=settings.prompt_version,
                                prediction=result.value,
                                raw_response=result.raw_response,
                                attempts=result.attempts,
                                elapsed_seconds=result.elapsed_seconds,
                            )
                            success += 1
                            kind_processed += 1
                            consecutive_errors = 0
                        except Exception as exc:
                            db.rollback()
                            db.save_error(
                                run_id=run_id,
                                item=item,
                                model_name=settings.ollama_model,
                                prompt_version=settings.prompt_version,
                                error_message=str(exc),
                                raw_response=str(
                                    getattr(exc, "raw_response", "")
                                ),
                                attempts=int(getattr(exc, "attempts", 1)),
                                elapsed_seconds=float(
                                    getattr(exc, "elapsed_seconds", 0.0)
                                ),
                            )
                            errors += 1
                            kind_processed += 1
                            consecutive_errors += 1

                        processed = success + errors
                        db.update_run_progress(run_id, success, errors, reused)
                        db.commit()

                        if (
                            consecutive_errors
                            >= settings.max_consecutive_errors
                        ):
                            raise RuntimeError(
                                "Circuit breaker acionado após "
                                f"{consecutive_errors} erros consecutivos. "
                                "Verifique o Ollama e as respostas do modelo antes de retomar."
                            )

                        if (
                            processed - last_reported
                            >= settings.database_batch_size
                        ):
                            print(
                                f"Processados {processed}/{total} "
                                f"(sucesso={success}, reutilizados={reused}, "
                                f"modelo={success - reused}, erros={errors}) | "
                                f"{format_progress(total, processed, started_at)}",
                                flush=True,
                            )
                            last_reported = processed

            db.finish_run(run_id, "completed", success, errors, reused=reused)
            print(
                f"Execução concluída: run_id={run_id}, processados={success + errors}, "
                f"sucesso={success}, reutilizados={reused}, "
                f"modelo={success - reused}, erros={errors} | "
                f"{format_progress(total, success + errors, started_at)}"
            )
            return 0 if errors == 0 else 2

        except KeyboardInterrupt:
            db.rollback()
            db.finish_run(
                run_id,
                "interrupted",
                success,
                errors,
                "Interrompido pelo usuário.",
                reused=reused,
            )
            print(f"Execução interrompida: run_id={run_id}")
            return 130
        except Exception as exc:
            db.rollback()
            db.finish_run(run_id, "failed", success, errors, str(exc), reused=reused)
            raise
    finally:
        db.close()
