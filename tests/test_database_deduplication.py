from __future__ import annotations

from gemma_hate_inference.database import ContentItem, Database


class _Result:
    def __init__(self, row: dict[str, object] | None = None):
        self.row = row

    def fetchone(self) -> dict[str, object] | None:
        return self.row


class _FakeConnection:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def execute(self, statement: str, params: object = None) -> _Result:
        self.calls.append((statement, params))
        if "COUNT(c.text_hash) AS completed" in statement:
            return _Result({"total": 10, "completed": 4})
        if "COUNT(*) AS total" in statement:
            return _Result({"total": 7})
        if "SELECT EXISTS" in statement:
            return _Result({"found": True})
        return _Result()


def _database() -> tuple[Database, _FakeConnection]:
    connection = _FakeConnection()
    database = Database.__new__(Database)
    database.conn = connection  # type: ignore[assignment]
    return database, connection


def test_pending_count_groups_equal_texts_and_uses_global_success() -> None:
    database, connection = _database()

    assert database.count_pending("comment", "model", "v1", (), False) == 7

    normalized = " ".join(connection.calls[-1][0].split())
    assert "GROUP BY MD5(BTRIM(s.text)), BTRIM(s.text)" in normalized
    assert "a.text_snapshot = BTRIM(s.text)" in normalized
    assert "a.comment_id = s.id" not in normalized


def test_non_labelable_technical_content_is_excluded() -> None:
    filter_sql = Database._useful_content_filter("BTRIM(s.text)")

    assert "~* '(https?://|www\\.)'" in filter_sql
    assert "com\\.br|org\\.br|gov\\.br" in filter_sql
    assert "[[:alnum:]-]+\\." in filter_sql
    assert "\\S+\\s*$" not in filter_sql
    assert "consulta.{0,20}(cpf|nome)" in filter_sql
    assert "dados[[:space:]]+b[aá]sicos" in filter_sql
    assert "boletim\\s+(informativo|de\\s+not[ií]cias)" in filter_sql
    assert "jrmunews" in filter_sql


def test_pending_queue_keeps_one_canonical_source_per_text() -> None:
    database, connection = _database()

    database.prepare_pending_queue("post", "model", "v1", (), False, None)

    queue_sql = " ".join(connection.calls[1][0].split())
    assert "PARTITION BY MD5(" in queue_sql
    assert "AS content_position" in queue_sql
    assert "WHERE content_position = 1" in queue_sql


def test_success_lookup_is_by_text_not_source_id() -> None:
    database, connection = _database()
    item = ContentItem("comment", 99, "reddit", "abc", "texto", "hash")

    assert database.has_success(item, "model", "v1") is True

    statement, params = connection.calls[-1]
    normalized = " ".join(statement.split())
    assert "text_hash = %s" in normalized
    assert "text_snapshot = %s" in normalized
    assert "comment_id" not in normalized
    assert params == ("model", "v1", "hash", "texto")


def test_overall_progress_counts_unique_texts() -> None:
    database, connection = _database()

    assert database.overall_progress("model", "v1", ("reddit",)) == {
        "total": 10,
        "completed": 4,
    }

    normalized = " ".join(connection.calls[-1][0].split())
    assert "GROUP BY text_hash, text_snapshot" in normalized
    assert "SELECT DISTINCT text_hash, text_snapshot" in normalized
    assert "c.text_snapshot = e.text_snapshot" in normalized
