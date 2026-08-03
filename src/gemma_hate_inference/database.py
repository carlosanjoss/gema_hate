from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from gemma_hate_inference.models import Prediction


PROJECT_DIR = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = PROJECT_DIR / "sql"


@dataclass(frozen=True)
class ContentItem:
    content_kind: str
    source_id: int
    platform: str
    external_id: str
    text: str
    text_hash: str


class Database:
    def __init__(self, uri: str):
        self.conn: Connection = psycopg.connect(uri, row_factory=dict_row)

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            self.conn.execute(migration_path.read_text(encoding="utf-8"))
        self.conn.commit()

    @staticmethod
    def _source(kind: str) -> tuple[str, str]:
        if kind == "post":
            return (
                "posts",
                """CASE
                    WHEN NULLIF(BTRIM(s.title), '') IS NOT NULL
                     AND NULLIF(BTRIM(s.text), '') IS NOT NULL
                        THEN BTRIM(s.title) || E'\\n\\n' || BTRIM(s.text)
                    ELSE COALESCE(
                        NULLIF(BTRIM(s.title), ''),
                        NULLIF(BTRIM(s.text), ''),
                        ''
                    )
                END""",
            )
        if kind == "comment":
            return "comments", "BTRIM(s.text)"
        raise ValueError(f"Tipo de conteúdo inválido: {kind}")

    @staticmethod
    def _useful_content_filter(text_expression: str) -> str:
        """Remove da inferência conteúdos técnicos sem texto rotulável."""
        return rf"""NOT (
            {text_expression} ~* '(https?://|www\.)'
            OR {text_expression} ~*
                '(^|[[:space:]<(])([[:alnum:]-]+\.)+(com\.br|org\.br|gov\.br|com|org|net|io|me|tv|app)(/[^[:space:]<]*)?($|[[:space:]>)])'
            OR (
                (
                    {text_expression} ~*
                        '(consulta.{{0,20}}(cpf|nome)|/cpf[0-9]*|dados[[:space:]]+b[aá]sicos)'
                    AND {text_expression} ~* 'cpf'
                )
                OR (
                    {text_expression} ~* 'cpf[[:space:]:•]*[0-9]{{10,14}}'
                    AND {text_expression} ~*
                        '(pis|t[ií]tulo[[:space:]]+eleitoral|data[[:space:]]+consulta[[:space:]]+receita|poder[[:space:]]+aquisitivo)'
                )
                OR (
                    {text_expression} ~* 'resultado[[:space:]:]*[0-9]+'
                    AND {text_expression} ~*
                        'cpf[[:space:]:•]*[0-9]{{9,14}}'
                    AND {text_expression} ~* '(rua|nascimento|sexo)'
                )
            )
            OR (
                {text_expression} ~*
                    '^[[:space:]*#_-]*(boletim\s+(informativo|de\s+not[ií]cias)|boletim\s+not[ií]cias|not[ií]cias\s+do\s+dia)'
                OR LEFT({text_expression}, 250) ~*
                    '(jrmunews|not[ií]cias\s+do\s+brasil\s+e\s+do\s+mundo|resumo(-[0-9]+)?\s+de\s+not[ií]cias|not[ií]cias\s+do\s+dia)'
            )
        )"""

    def count_pending(
        self,
        kind: str,
        model_name: str,
        prompt_version: str,
        platforms: Sequence[str],
        only_usable: bool,
    ) -> int:
        table, text_expression = self._source(kind)
        filters = [
            f"NULLIF({text_expression}, '') IS NOT NULL",
            self._useful_content_filter(text_expression),
        ]
        params: list[object] = []
        if platforms:
            filters.append("s.platform = ANY(%s)")
            params.append(list(platforms))
        if only_usable:
            filters.append("s.is_usable IS TRUE")
        filters.append(
            f"""NOT EXISTS (
                SELECT 1
                FROM gemma_hate_analyses a
                WHERE a.model_name = %s
                  AND a.prompt_version = %s
                  AND a.text_hash = MD5({text_expression})
                  AND a.text_snapshot = {text_expression}
                  AND a.status = 'success'
            )"""
        )
        params.extend([model_name, prompt_version])
        sql = f"""
            SELECT COUNT(*) AS total
            FROM (
                SELECT MD5({text_expression}), {text_expression}
                FROM {table} s
                WHERE {' AND '.join(filters)}
                GROUP BY MD5({text_expression}), {text_expression}
            ) unique_texts
        """
        row = self.conn.execute(sql, params).fetchone()
        return int(row["total"])

    def iter_pending(
        self,
        kind: str,
        model_name: str,
        prompt_version: str,
        platforms: Sequence[str],
        only_usable: bool,
        page_size: int,
        limit: int | None,
    ) -> Iterator[ContentItem]:
        table, text_expression = self._source(kind)
        last_id = 0
        emitted = 0
        seen_texts: set[tuple[str, str]] = set()

        while limit is None or emitted < limit:
            batch_limit = page_size if limit is None else min(page_size, limit - emitted)
            filters = [
                "s.id > %s",
                f"NULLIF({text_expression}, '') IS NOT NULL",
                self._useful_content_filter(text_expression),
            ]
            params: list[object] = [last_id]
            if platforms:
                filters.append("s.platform = ANY(%s)")
                params.append(list(platforms))
            if only_usable:
                filters.append("s.is_usable IS TRUE")
            filters.append(
                f"""NOT EXISTS (
                    SELECT 1
                    FROM gemma_hate_analyses a
                    WHERE a.model_name = %s
                      AND a.prompt_version = %s
                      AND a.text_hash = MD5({text_expression})
                      AND a.text_snapshot = {text_expression}
                      AND a.status = 'success'
                )"""
            )
            params.extend([model_name, prompt_version, batch_limit])
            sql = f"""
                SELECT
                    s.id AS source_id,
                    s.platform,
                    s.external_id,
                    {text_expression} AS source_text,
                    MD5({text_expression}) AS text_hash
                FROM {table} s
                WHERE {' AND '.join(filters)}
                ORDER BY s.id
                LIMIT %s
            """
            rows = self.conn.execute(sql, params).fetchall()
            if not rows:
                return

            for row in rows:
                last_id = int(row["source_id"])
                text_key = (str(row["text_hash"]), str(row["source_text"]))
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                emitted += 1
                yield ContentItem(
                    content_kind=kind,
                    source_id=last_id,
                    platform=str(row["platform"]),
                    external_id=str(row["external_id"]),
                    text=str(row["source_text"]),
                    text_hash=str(row["text_hash"]),
                )

    def prepare_pending_queue(
        self,
        kind: str,
        model_name: str,
        prompt_version: str,
        platforms: Sequence[str],
        only_usable: bool,
        limit: int | None,
    ) -> None:
        table, text_expression = self._source(kind)
        filters = [
            f"NULLIF({text_expression}, '') IS NOT NULL",
            self._useful_content_filter(text_expression),
        ]
        params: list[object] = []
        if platforms:
            filters.append("s.platform = ANY(%s)")
            params.append(list(platforms))
        if only_usable:
            filters.append("s.is_usable IS TRUE")
        filters.append(
            f"""NOT EXISTS (
                SELECT 1
                FROM gemma_hate_analyses a
                WHERE a.model_name = %s
                  AND a.prompt_version = %s
                  AND a.text_hash = MD5({text_expression})
                  AND a.text_snapshot = {text_expression}
                  AND a.status = 'success'
            )"""
        )
        params.extend([model_name, prompt_version])
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT %s"
            params.append(limit)

        self.conn.execute("DROP TABLE IF EXISTS pg_temp.gemma_pending_queue")
        self.conn.execute(
            f"""
            CREATE TEMP TABLE gemma_pending_queue
            ON COMMIT PRESERVE ROWS AS
            WITH candidates AS (
                SELECT
                    s.id AS source_id,
                    CHAR_LENGTH({text_expression}) AS text_length,
                    COUNT(*) OVER (
                        PARTITION BY MD5({text_expression}), {text_expression}
                    ) AS duplicate_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY MD5({text_expression}), {text_expression}
                        ORDER BY s.id
                    ) AS content_position
                FROM {table} s
                WHERE {' AND '.join(filters)}
            ),
            unique_candidates AS (
                SELECT source_id, text_length, duplicate_count
                FROM candidates
                WHERE content_position = 1
            )
            SELECT
                ROW_NUMBER() OVER (
                    ORDER BY duplicate_count DESC, text_length, source_id
                ) AS position,
                source_id
            FROM unique_candidates
            ORDER BY duplicate_count DESC, text_length, source_id
            {limit_sql}
            """,
            params,
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX ON gemma_pending_queue (position)"
        )

    def iter_queued_pending(
        self,
        kind: str,
        model_name: str,
        prompt_version: str,
        page_size: int,
    ) -> Iterator[ContentItem]:
        table, text_expression = self._source(kind)
        last_position = 0

        while True:
            rows = self.conn.execute(
                f"""
                SELECT
                    q.position,
                    s.id AS source_id,
                    s.platform,
                    s.external_id,
                    {text_expression} AS source_text,
                    MD5({text_expression}) AS text_hash
                FROM gemma_pending_queue q
                JOIN {table} s ON s.id = q.source_id
                WHERE q.position > %s
                  AND NOT EXISTS (
                      SELECT 1
                      FROM gemma_hate_analyses a
                      WHERE a.model_name = %s
                        AND a.prompt_version = %s
                        AND a.text_hash = MD5({text_expression})
                        AND a.text_snapshot = {text_expression}
                        AND a.status = 'success'
                  )
                ORDER BY q.position
                LIMIT %s
                """,
                (
                    last_position,
                    model_name,
                    prompt_version,
                    page_size,
                ),
            ).fetchall()
            if not rows:
                return

            for row in rows:
                last_position = int(row["position"])
                yield ContentItem(
                    content_kind=kind,
                    source_id=int(row["source_id"]),
                    platform=str(row["platform"]),
                    external_id=str(row["external_id"]),
                    text=str(row["source_text"]),
                    text_hash=str(row["text_hash"]),
                )

    def has_success(
        self,
        item: ContentItem,
        model_name: str,
        prompt_version: str,
    ) -> bool:
        row = self.conn.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM gemma_hate_analyses
                WHERE model_name = %s
                  AND prompt_version = %s
                  AND text_hash = %s
                  AND text_snapshot = %s
                  AND status = 'success'
            ) AS found
            """,
            (
                model_name,
                prompt_version,
                item.text_hash,
                item.text,
            ),
        ).fetchone()
        return bool(row["found"])

    def create_run(
        self,
        run_name: str,
        model_name: str,
        prompt_version: str,
        filters: dict[str, object],
        total_selected: int,
    ) -> int:
        row = self.conn.execute(
            """
            INSERT INTO gemma_hate_runs
                (run_name, model_name, prompt_version, filters, total_selected)
            VALUES (%s, %s, %s, %s::jsonb, %s)
            RETURNING run_id
            """,
            (run_name, model_name, prompt_version, json.dumps(filters), total_selected),
        ).fetchone()
        self.conn.commit()
        return int(row["run_id"])

    def save_success(
        self,
        run_id: int,
        item: ContentItem,
        model_name: str,
        prompt_version: str,
        prediction: Prediction,
        raw_response: str,
        attempts: int,
        elapsed_seconds: float,
    ) -> int:
        post_id = item.source_id if item.content_kind == "post" else None
        comment_id = item.source_id if item.content_kind == "comment" else None
        row = self.conn.execute(
            """
            INSERT INTO gemma_hate_analyses (
                run_id, content_kind, post_id, comment_id, platform, external_id,
                text_hash, text_snapshot, model_name, prompt_version, status,
                is_hate, hate_probability, hate_types,
                target_identified, textual_evidence, justification, raw_response,
                attempt_count, inference_seconds
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, 'success',
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (
                model_name, prompt_version, text_hash
            ) WHERE status = 'success'
            DO NOTHING
            RETURNING analysis_id
            """,
            (
                run_id, item.content_kind, post_id, comment_id, item.platform,
                item.external_id, item.text_hash, item.text, model_name, prompt_version,
                prediction.is_hate, prediction.hate_probability,
                [category.code for category in prediction.categories],
                prediction.target_identified, prediction.textual_evidence,
                prediction.justification, raw_response, attempts, elapsed_seconds,
            ),
        ).fetchone()
        if row is None:
            existing = self.conn.execute(
                """
                SELECT analysis_id
                FROM gemma_hate_analyses
                WHERE model_name = %s
                  AND prompt_version = %s
                  AND text_hash = %s
                  AND text_snapshot = %s
                  AND status = 'success'
                """,
                (model_name, prompt_version, item.text_hash, item.text),
            ).fetchone()
            if existing is None:
                raise RuntimeError("Conflito de hash textual sem análise canônica")
            return int(existing["analysis_id"])

        analysis_id = int(row["analysis_id"])

        if prediction.categories:
            with self.conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO gemma_hate_analysis_categories
                        (analysis_id, category_code, probability)
                    VALUES (%s, %s, %s)
                    """,
                    [
                        (analysis_id, category.code, category.probability)
                        for category in prediction.categories
                    ],
                )
        return analysis_id

    def save_error(
        self,
        run_id: int,
        item: ContentItem,
        model_name: str,
        prompt_version: str,
        error_message: str,
        raw_response: str,
        attempts: int,
        elapsed_seconds: float,
    ) -> int:
        post_id = item.source_id if item.content_kind == "post" else None
        comment_id = item.source_id if item.content_kind == "comment" else None
        row = self.conn.execute(
            """
            INSERT INTO gemma_hate_analyses (
                run_id, content_kind, post_id, comment_id, platform, external_id,
                text_hash, text_snapshot, model_name, prompt_version, status,
                raw_response, error_message, attempt_count, inference_seconds
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'error', %s, %s, %s, %s)
            RETURNING analysis_id
            """,
            (
                run_id, item.content_kind, post_id, comment_id, item.platform,
                item.external_id, item.text_hash, item.text, model_name, prompt_version,
                raw_response, error_message[:4000], attempts, elapsed_seconds,
            ),
        ).fetchone()
        return int(row["analysis_id"])

    def update_run_progress(
        self, run_id: int, success: int, errors: int, reused: int = 0
    ) -> None:
        self.conn.execute(
            """
            UPDATE gemma_hate_runs
            SET processed_count = %s, success_count = %s, error_count = %s,
                reused_count = %s
            WHERE run_id = %s
            """,
            (success + errors, success, errors, reused, run_id),
        )

    def finish_run(
        self,
        run_id: int,
        status: str,
        success: int,
        errors: int,
        error_message: str | None = None,
        reused: int = 0,
    ) -> None:
        self.conn.execute(
            """
            UPDATE gemma_hate_runs
            SET status = %s,
                processed_count = %s,
                success_count = %s,
                error_count = %s,
                reused_count = %s,
                finished_at = NOW(),
                error_message = %s
            WHERE run_id = %s
            """,
            (status, success + errors, success, errors, reused, error_message, run_id),
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def recent_runs(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.conn.execute(
            """
            SELECT run_id, run_name, model_name, prompt_version, status,
                   total_selected, processed_count, success_count, error_count, reused_count,
                   started_at, finished_at
            FROM gemma_hate_runs
            ORDER BY run_id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def overall_progress(
        self,
        model_name: str,
        prompt_version: str,
        platforms: Sequence[str],
    ) -> dict[str, int]:
        post_text_expression = """CASE
            WHEN NULLIF(BTRIM(p.title), '') IS NOT NULL
             AND NULLIF(BTRIM(p.text), '') IS NOT NULL
              THEN BTRIM(p.title) || E'\n\n' || BTRIM(p.text)
            ELSE COALESCE(
              NULLIF(BTRIM(p.title), ''),
              NULLIF(BTRIM(p.text), ''),
              ''
            )
        END"""
        comment_text_expression = "BTRIM(c.text)"
        row = self.conn.execute(
            f"""
            WITH source_texts AS (
                SELECT
                    MD5({post_text_expression}) AS text_hash,
                    {post_text_expression} AS text_snapshot
                FROM posts p
                WHERE p.platform = ANY(%s)
                  AND {self._useful_content_filter(post_text_expression)}
                UNION ALL
                SELECT
                    MD5({comment_text_expression}),
                    {comment_text_expression}
                FROM comments c
                WHERE c.platform = ANY(%s)
                  AND {self._useful_content_filter(comment_text_expression)}
            ), eligible AS (
                SELECT text_hash, text_snapshot
                FROM source_texts
                WHERE NULLIF(text_snapshot, '') IS NOT NULL
                GROUP BY text_hash, text_snapshot
            ), completed AS (
                SELECT DISTINCT text_hash, text_snapshot
                FROM gemma_hate_analyses
                WHERE model_name = %s
                  AND prompt_version = %s
                  AND status = 'success'
            )
            SELECT COUNT(*) AS total,
                   COUNT(c.text_hash) AS completed
            FROM eligible e
            LEFT JOIN completed c
              ON c.text_hash = e.text_hash
             AND c.text_snapshot = e.text_snapshot
            """,
            (list(platforms), list(platforms), model_name, prompt_version),
        ).fetchone()
        return {"total": int(row["total"]), "completed": int(row["completed"])}
