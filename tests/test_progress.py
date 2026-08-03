from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gemma_hate_inference.progress import (
    estimate_progress,
    format_duration,
    format_finished_progress,
    format_progress,
)


def test_estimate_progress_calculates_rate_eta_and_finish() -> None:
    started = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
    now = started + timedelta(seconds=100)

    estimate = estimate_progress(total=100, processed=20, started_at=started, now=now)

    assert estimate.elapsed_seconds == 100
    assert estimate.items_per_second == 0.2
    assert estimate.remaining_seconds == 400
    assert estimate.estimated_finish == started + timedelta(seconds=500)


def test_progress_display_handles_zero_processed() -> None:
    started = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
    message = format_progress(total=10, processed=0, started_at=started, now=started)

    assert "velocidade=calculando" in message
    assert "restante=calculando" in message
    assert format_duration(3661) == "01:01:01"


def test_finished_progress_uses_real_finish_time_not_eta() -> None:
    started = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
    finished = started + timedelta(minutes=5)

    message = format_finished_progress(10, 10, started, finished)

    assert "término=15/07 10:05" in message
    assert "término_estimado" not in message
