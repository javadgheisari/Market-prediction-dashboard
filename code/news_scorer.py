"""Optional structured AI scoring for asset-related news headlines.

The default crawler remains fully local. This module is imported only when
``--scorer openai`` (or ``NEWS_SCORER=openai``) is selected.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
EVENT_TYPES = {
    "earnings",
    "macro",
    "monetary_policy",
    "geopolitical",
    "regulation",
    "supply_demand",
    "technical",
    "company",
    "other",
}


def _bounded(value: Any, lower: float, upper: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"AI scorer returned a non-numeric {name}.") from error
    if not lower <= number <= upper:
        raise ValueError(
            f"AI scorer returned {name}={number}; expected {lower}..{upper}."
        )
    return number


def _output_text(payload: dict[str, Any]) -> str:
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    raise ValueError("OpenAI response did not contain structured output text.")


def score_headlines_openai(
    symbol: str,
    asset_description: str,
    headlines: list[dict[str, str]],
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    timeout: float = 45.0,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, dict[str, str]]:
    """Score headlines with the Responses API and strict JSON Schema."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is required when the news scorer is 'openai'."
        )
    if not headlines:
        return {}

    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_id": {"type": "string"},
                        "asset_impact": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                        },
                        "relevance": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "novelty": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "impact_horizon_hours": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 720,
                        },
                        "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
                    },
                    "required": [
                        "article_id", "asset_impact", "relevance", "importance",
                        "confidence", "novelty", "impact_horizon_hours", "event_type",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    instructions = (
        "Score each financial-news headline for its expected causal effect on the "
        f"price of {symbol} ({asset_description}). Asset impact is -1 strongly "
        "bearish to +1 strongly bullish. Distinguish asset impact from generic "
        "sentiment. Use only information present in the headline, lower confidence "
        "when ambiguous, and return exactly one score for every supplied article_id."
    )
    request_body = {
        "model": model,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps({"asset": symbol, "headlines": headlines}),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "asset_news_scores",
                "strict": True,
                "schema": schema,
            }
        },
    }
    client = session or requests.Session()
    response = client.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=timeout,
    )
    response.raise_for_status()
    parsed = json.loads(_output_text(response.json()))

    expected_ids = {item["article_id"] for item in headlines}
    result: dict[str, dict[str, str]] = {}
    for item in parsed.get("scores", []):
        article_id = str(item.get("article_id", ""))
        if article_id not in expected_ids or article_id in result:
            raise ValueError("AI scorer returned an unknown or duplicate article_id.")
        event_type = str(item.get("event_type", "other"))
        if event_type not in EVENT_TYPES:
            raise ValueError(f"AI scorer returned unsupported event_type '{event_type}'.")
        result[article_id] = {
            "asset_impact": f"{_bounded(item['asset_impact'], -1, 1, 'asset_impact'):.4f}",
            "relevance": f"{_bounded(item['relevance'], 0, 1, 'relevance'):.4f}",
            "importance": f"{_bounded(item['importance'], 0, 1, 'importance'):.4f}",
            "confidence": f"{_bounded(item['confidence'], 0, 1, 'confidence'):.4f}",
            "novelty": f"{_bounded(item['novelty'], 0, 1, 'novelty'):.4f}",
            "impact_horizon_hours": (
                f"{_bounded(item['impact_horizon_hours'], 1, 720, 'impact_horizon_hours'):.2f}"
            ),
            "event_type": event_type,
            "scoring_method": f"openai:{model}",
        }
    missing = expected_ids.difference(result)
    if missing:
        raise ValueError(f"AI scorer omitted {len(missing)} headline(s).")
    return result
