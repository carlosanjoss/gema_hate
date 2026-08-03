from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime

from gemma_hate_inference.config import Settings
from gemma_hate_inference.database import Database
from gemma_hate_inference.platforms import SUPPORTED_PLATFORMS
from gemma_hate_inference.service import RunOptions, run_inference
from gemma_hate_inference.progress import format_finished_progress, format_progress


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser maior que zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inferência de discurso de ódio em duas etapas com Gemma/Ollama."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("migrate", help="Cria/atualiza as tabelas aditivas no PostgreSQL.")

    run = subparsers.add_parser("run", help="Executa inferência nos posts/comentários pendentes.")
    run.add_argument(
        "--content",
        choices=["all", "posts", "comments"],
        default="all",
        help="Tipo de conteúdo (padrão: all).",
    )
    run.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        default=[],
        choices=SUPPORTED_PLATFORMS,
        type=str.lower,
        help=(
            "Filtra plataforma; pode ser repetido. Sem este argumento, processa "
            "reddit, twitter, youtube, facebook, tiktok, whatsapp e telegram."
        ),
    )
    run.add_argument("--limit", type=_positive, help="Limite global de itens.")
    run.add_argument(
        "--only-usable",
        action="store_true",
        help="Processa somente is_usable=true. Sem a opção, usa todo texto não vazio.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Conta os itens pendentes sem chamar o modelo nem gravar análises.",
    )
    run.add_argument("--run-name", help="Nome único opcional para a execução.")
    run.add_argument(
        "--skip-migrate",
        action="store_true",
        help="Não reaplica migrações; use somente em workers adicionais.",
    )

    status = subparsers.add_parser("status", help="Mostra as execuções mais recentes.")
    status.add_argument("--limit", type=_positive, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "migrate":
        db = Database(settings.postgres_uri)
        try:
            db.migrate()
            print("Migração aplicada com sucesso.")
        finally:
            db.close()
        return 0

    if args.command == "status":
        db = Database(settings.postgres_uri)
        try:
            overall = db.overall_progress(
                settings.ollama_model,
                settings.prompt_version,
                SUPPORTED_PLATFORMS,
            )
            rows = db.recent_runs(args.limit)
        finally:
            db.close()
        total = overall["total"]
        completed = overall["completed"]
        target_70 = math.ceil(total * 0.70)
        print(
            f"GERAL sucesso={completed}/{total} ({completed / total:.2%}) "
            f"meta_70={target_70} faltam_para_70={max(target_70 - completed, 0)}"
        )
        if not rows:
            print("Nenhuma execução registrada.")
            return 0
        for row in rows:
            started_at = row["started_at"]
            finished_at = row["finished_at"]
            if finished_at:
                progress = format_finished_progress(
                    int(row["total_selected"]),
                    int(row["processed_count"]),
                    started_at,
                    finished_at,
                )
            else:
                progress = format_progress(
                    int(row["total_selected"]),
                    int(row["processed_count"]),
                    started_at,
                    datetime.now(tz=started_at.tzinfo),
                )
            print(
                f"run_id={row['run_id']} status={row['status']} "
                f"processados={row['processed_count']}/{row['total_selected']} "
                f"sucesso={row['success_count']} reutilizados={row['reused_count']} "
                f"erros={row['error_count']} "
                f"modelo={row['model_name']} nome={row['run_name']} | {progress}"
            )
        return 0

    if args.command == "run":
        return run_inference(
            settings,
            RunOptions(
                content=args.content,
                platforms=tuple(args.platforms) or SUPPORTED_PLATFORMS,
                only_usable=args.only_usable,
                limit=args.limit,
                dry_run=args.dry_run,
                run_name=args.run_name,
                skip_migrate=args.skip_migrate,
            ),
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
