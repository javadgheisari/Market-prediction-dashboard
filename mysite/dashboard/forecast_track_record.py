"""Record issued forecasts and score them against later closed candles."""

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from django.db.models import Count, Q
from django.utils import timezone

from dashboard import market_data_service
from dashboard.models import ForecastRecord


TEHRAN = ZoneInfo("Asia/Tehran")
RECHECK_AFTER = timedelta(minutes=30)
MODEL_LABELS = {
    "xgboost": "XGBoost Portfolio",
    "chronos": "Chronos-2",
    "gru": "GRU",
    "lstm": "LSTM",
}
SOURCE_LABELS = {
    "forexsb": "Downloaded history",
    "bundled": "Project CSV",
    "upload": "Uploaded CSV",
}


def _utc_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    return parsed.astimezone(dt_timezone.utc)


def _positive_price(value: object) -> float:
    price = float(value)
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Forecast price must be positive and finite.")
    return price


def record_prediction(
    symbol: str,
    model_name: str,
    chart: dict,
    *,
    price_source: str,
    used_news: bool,
    horizon_steps: int,
    issued_at: datetime | None = None,
) -> ForecastRecord | None:
    """Save a real future forecast; never retroactively score a historical chart."""
    try:
        origin_at = _utc_timestamp(chart["origin_at_utc"])
        target_at = _utc_timestamp(chart["target_at_utc"])
        interval_minutes = int(chart["interval_minutes"])
        origin_close = _positive_price(chart["actual_prices"][-1])
        predicted_close = _positive_price(chart["future_predictions"][-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    issued_at = issued_at or timezone.now()
    if timezone.is_naive(issued_at):
        issued_at = timezone.make_aware(issued_at, dt_timezone.utc)
    issued_at = issued_at.astimezone(dt_timezone.utc)
    if not (origin_at <= issued_at < target_at) or interval_minutes not in market_data_service.INTERVALS:
        return None
    if horizon_steps < 1 or len(chart["future_predictions"]) < horizon_steps:
        return None

    return ForecastRecord.objects.create(
        symbol=symbol.upper(), model_name=model_name,
        price_source=price_source, used_news=used_news,
        issued_at=issued_at, origin_at=origin_at, target_at=target_at,
        interval_minutes=interval_minutes, horizon_steps=horizon_steps,
        origin_close=origin_close, predicted_close=predicted_close,
    )


def _direction(price: float, origin: float) -> int:
    if math.isclose(price, origin, rel_tol=1e-10, abs_tol=1e-12):
        return 0
    return 1 if price > origin else -1


def _settle(record: ForecastRecord, actual_close: float) -> None:
    record.actual_close = actual_close
    record.actual_at = record.target_at + timedelta(minutes=record.interval_minutes)
    predicted_direction = _direction(record.predicted_close, record.origin_close)
    actual_direction = _direction(actual_close, record.origin_close)
    if predicted_direction == 0:
        record.status = ForecastRecord.Status.NO_DIRECTION
    elif predicted_direction == actual_direction:
        record.status = ForecastRecord.Status.CORRECT
    else:
        record.status = ForecastRecord.Status.INCORRECT


def refresh_due_records(
    symbol: str, *, now: datetime | None = None, session=None
) -> bool:
    """Check matured predictions against BiQuote's matching closed candles.

    Return whether verification was temporarily unavailable. Missing candles
    remain pending; they are never inferred from a nearby price or backtest.
    """
    now = now or timezone.now()
    pending = list(
        ForecastRecord.objects.filter(
            symbol=symbol.upper(), status=ForecastRecord.Status.PENDING,
            target_at__lte=now,
        ).filter(
            Q(last_checked_at__isnull=True)
            | Q(last_checked_at__lte=now - RECHECK_AFTER)
        ).order_by("target_at")[:100]
    )
    groups = defaultdict(list)
    for record in pending:
        closed_after = record.target_at + timedelta(minutes=record.interval_minutes)
        if now < closed_after:
            continue
        groups[record.interval_minutes].append(record)

    unavailable = False
    for minutes, records in groups.items():
        interval = market_data_service.INTERVALS.get(minutes)
        if interval is None:
            continue
        try:
            options = {"session": session} if session is not None else {}
            bars = market_data_service.fetch_closed_bars(
                symbol, interval, pd.Timestamp(min(r.origin_at for r in records)),
                **options,
            )
        except (requests.RequestException, RuntimeError, ValueError):
            unavailable = True
            bars = []

        by_time = {}
        for bar in bars:
            if bar.get("isOpen"):
                continue
            try:
                timestamp = _utc_timestamp(bar["openTime"])
                price = _positive_price(bar["close"])
            except (KeyError, TypeError, ValueError):
                continue
            by_time[timestamp] = price

        for record in records:
            record.last_checked_at = now
            if record.target_at in by_time:
                _settle(record, by_time[record.target_at])
        ForecastRecord.objects.bulk_update(
            records, ["last_checked_at", "actual_close", "actual_at", "status"]
        )
    return unavailable


def track_record_for_symbol(symbol: str, *, limit: int = 20) -> dict:
    """Return per-model counts and the latest issued forecasts for display."""
    records = ForecastRecord.objects.filter(symbol=symbol.upper())
    groups = records.values("model_name").annotate(
        correct=Count("id", filter=Q(status=ForecastRecord.Status.CORRECT)),
        incorrect=Count("id", filter=Q(status=ForecastRecord.Status.INCORRECT)),
        pending=Count("id", filter=Q(status=ForecastRecord.Status.PENDING)),
        no_direction=Count("id", filter=Q(status=ForecastRecord.Status.NO_DIRECTION)),
    ).order_by("model_name")
    summary = []
    for group in groups:
        scored = group["correct"] + group["incorrect"]
        summary.append({
            "model": MODEL_LABELS.get(group["model_name"], group["model_name"].upper()),
            **group,
            "scored": scored,
            "accuracy": round(100 * group["correct"] / scored, 1) if scored else None,
        })

    recent = []
    for record in records.order_by("-issued_at", "-pk")[:limit]:
        recent.append({
            "model": MODEL_LABELS.get(record.model_name, record.model_name.upper()),
            "source": SOURCE_LABELS.get(record.price_source, record.price_source),
            "used_news": record.used_news,
            "issued_at": record.issued_at.astimezone(TEHRAN).strftime("%Y-%m-%d %H:%M"),
            "target_at": record.target_at.astimezone(TEHRAN).strftime("%Y-%m-%d %H:%M"),
            "horizon_steps": record.horizon_steps,
            "origin_close": record.origin_close,
            "predicted_close": record.predicted_close,
            "actual_close": record.actual_close,
            "status": record.status,
            "status_label": record.get_status_display(),
        })
    return {"summary": summary, "recent": recent, "total": records.count()}
