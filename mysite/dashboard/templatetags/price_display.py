"""Format approximate market prices without losing small token prices."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from django import template


register = template.Library()


@register.filter
def approximate_price(value: object) -> str:
    try:
        price = Decimal(str(value))
        if not price.is_finite() or price <= 0:
            return "—"
        places = min(12, max(2, 5 - price.adjusted()))
        rounded = price.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        if rounded == 0:
            return f"{price:.5E}"
        formatted = f"{rounded:,.{places}f}"
        return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
    except (InvalidOperation, TypeError, ValueError):
        return "—"
