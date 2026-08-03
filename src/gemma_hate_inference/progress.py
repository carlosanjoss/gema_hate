from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ProgressEstimate:
    elapsed_seconds: float
    items_per_second: float | None
    remaining_seconds: float | None
    estimated_finish: datetime | None


def estimate_progress(
    total: int,
    processed: int,
    started_at: datetime,
    now: datetime | None = None,
) -> ProgressEstimate:
    if now is None:
        now = datetime.now(tz=started_at.tzinfo)

    elapsed_seconds = max((now - started_at).total_seconds(), 0.0)
    if processed <= 0 or elapsed_seconds <= 0:
        return ProgressEstimate(elapsed_seconds, None, None, None)

    items_per_second = processed / elapsed_seconds
    remaining_items = max(total - processed, 0)
    remaining_seconds = remaining_items / items_per_second
    return ProgressEstimate(
        elapsed_seconds=elapsed_seconds,
        items_per_second=items_per_second,
        remaining_seconds=remaining_seconds,
        estimated_finish=now + timedelta(seconds=remaining_seconds),
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculando"
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_progress(
    total: int,
    processed: int,
    started_at: datetime,
    now: datetime | None = None,
) -> str:
    estimate = estimate_progress(total, processed, started_at, now)
    rate = "calculando" if estimate.items_per_second is None else f"{estimate.items_per_second:.3f} item/s"
    finish = (
        "calculando"
        if estimate.estimated_finish is None
        else estimate.estimated_finish.strftime("%d/%m %H:%M")
    )
    return (
        f"tempo={format_duration(estimate.elapsed_seconds)} "
        f"velocidade={rate} "
        f"restante={format_duration(estimate.remaining_seconds)} "
        f"término_estimado={finish}"
    )


def format_finished_progress(
    total: int,
    processed: int,
    started_at: datetime,
    finished_at: datetime,
) -> str:
    estimate = estimate_progress(total, processed, started_at, finished_at)
    rate = "calculando" if estimate.items_per_second is None else f"{estimate.items_per_second:.3f} item/s"
    return (
        f"tempo={format_duration(estimate.elapsed_seconds)} "
        f"velocidade={rate} "
        f"término={finished_at.strftime('%d/%m %H:%M')}"
    )
