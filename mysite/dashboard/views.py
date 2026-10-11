import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from django.contrib import messages
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.shortcuts import render
from django.utils.text import slugify
from django.views import View

from dashboard import (
    forecaster,
    forexsb_service,
    market_context_service,
    market_data_service,
    news_service,
)
from dashboard.data_paths import has_bundled_data, list_available_assets
from dashboard.forms import NameForm
from dashboard.price_export_service import latest_price_csv


SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.-]+$")


def normalize_symbol(symbol_name):
    symbol = symbol_name.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("The asset symbol contains unsupported characters.")
    return symbol


def available_symbols():
    return {
        symbol
        for symbols in list_available_assets().values()
        for symbol in symbols
    }


@contextmanager
def temporary_price_csv(uploaded_file, symbol):
    if uploaded_file is None:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="market-dashboard-") as directory:
        path = Path(directory) / f"{symbol}_uploaded.csv"
        with path.open("wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        yield path


class DashboardView(View):
    template_name = "dashboard.html"

    def get(self, request):
        return render(
            request,
            self.template_name,
            {"asset_groups": list_available_assets()},
        )


class LatestPricesView(View):
    def get(self, request):
        symbols = sorted(available_symbols())
        try:
            prices = market_data_service.fetch_latest_prices(symbols)
        except (OSError, RuntimeError, ValueError) as exc:
            return JsonResponse({"error": str(exc), "prices": {}}, status=502)
        return JsonResponse({"prices": prices})


class PredictView(View):
    template_name = "predict.html"
    form_class = NameForm

    def get_symbol(self, symbol_name):
        return normalize_symbol(symbol_name)

    def base_context(self, symbol, form, news_page=1):
        return {
            "symbol_name": symbol,
            "form": form,
            "bundled_available": has_bundled_data(symbol),
            "forexsb_available": forexsb_service.supports_symbol(symbol),
            "asset_news": news_service.load_asset_news(symbol, page=news_page),
            "context_status": market_context_service.load_context_status(symbol),
        }

    def get(self, request, symbol_name):
        try:
            symbol = self.get_symbol(symbol_name)
            form = self.form_class(
                initial={"predict_steps": 24},
                bundled_available=has_bundled_data(symbol),
                forexsb_available=forexsb_service.supports_symbol(symbol),
            )
            error = None
        except ValueError as exc:
            symbol = symbol_name
            form = self.form_class()
            error = str(exc)

        return render(
            request,
            self.template_name,
            {
                **self.base_context(
                    symbol, form, news_page=request.GET.get("news_page", 1)
                ),
                "error": error,
            },
        )

    def post(self, request, symbol_name):
        symbol = symbol_name.strip().upper()
        form = self.form_class(
            request.POST,
            request.FILES,
            bundled_available=has_bundled_data(symbol),
            forexsb_available=forexsb_service.supports_symbol(symbol),
        )
        context = self.base_context(
            symbol, form, news_page=request.GET.get("news_page", 1)
        )

        if form.is_valid():
            try:
                symbol = self.get_symbol(symbol_name)
                model_name = form.cleaned_data["model_name"]
                model_label = dict(form.fields["model_name"].choices).get(
                    model_name, model_name.upper()
                )
                if model_name == "news_buy":
                    context.update(
                        {
                            "symbol_name": symbol,
                            "model_name": model_label,
                            "news_decision": news_service.predict_news_buy_decision(
                                symbol
                            ),
                        }
                    )
                    return render(request, self.template_name, context)
                data_source = form.cleaned_data["data_source"]
                predict_steps = form.cleaned_data["predict_steps"]
                uploaded_csv = (
                    form.cleaned_data.get("market_csv")
                    if form.cleaned_data["price_source"] == "upload"
                    else None
                )
                with temporary_price_csv(uploaded_csv, symbol) as price_csv:
                    chart_data = forecaster.run_dashboard_forecast(
                        symbol=symbol,
                        model_name=model_name,
                        predict_steps=predict_steps,
                        price_csv=price_csv,
                        price_source=form.cleaned_data["price_source"],
                        use_news=data_source in {"with_news", "with_news_context"},
                        use_context=data_source in {"with_context", "with_news_context"},
                        advanced_config=form.advanced_config(),
                    )
                context.update(
                    {
                        "symbol_name": symbol,
                        "model_name": model_label,
                        "data_source": data_source,
                        "price_source": form.cleaned_data["price_source"],
                        "predict_steps": predict_steps,
                        "chart_data": chart_data,
                    }
                )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                context["error"] = str(exc)

        return render(request, self.template_name, context)


class NewsCrawlView(View):
    def post(self, request, symbol_name):
        symbol = symbol_name.strip().upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            messages.error(request, "The asset symbol contains unsupported characters.")
            return redirect("dashboard")

        try:
            result = news_service.crawl_asset_news(symbol)
        except Exception as exc:
            messages.error(request, f"Could not crawl news for {symbol}: {exc}")
        else:
            messages.success(
                request,
                f"{symbol} news updated: fetched {result['fetched']} relevant "
                f"articles; {result['saved']} total saved.",
            )
        return redirect("predict", symbol_name=symbol)


class NewsClearView(View):
    def post(self, request, symbol_name):
        try:
            symbol = normalize_symbol(symbol_name)
        except ValueError as exc:
            messages.error(request, str(exc))
            return redirect("dashboard")

        try:
            removed = news_service.clear_asset_news(symbol)
        except OSError as exc:
            messages.error(request, f"Could not clear news for {symbol}: {exc}")
        else:
            if removed:
                messages.success(request, f"Saved news for {symbol} was cleared.")
            else:
                messages.info(request, f"There was no saved news to clear for {symbol}.")
        return redirect("predict", symbol_name=symbol)


class MarketContextCollectView(View):
    def post(self, request, symbol_name):
        try:
            symbol = normalize_symbol(symbol_name)
        except ValueError as exc:
            messages.error(request, str(exc))
            return redirect("dashboard")
        if symbol not in available_symbols():
            raise Http404("Unknown dashboard asset.")

        try:
            result = market_context_service.collect_asset_context(symbol)
        except (OSError, RuntimeError, ValueError) as exc:
            messages.error(
                request, f"Could not collect market intelligence for {symbol}: {exc}"
            )
        else:
            source_count = len(result["sources"])
            error_count = len(result["errors"])
            if result["saved"]:
                messages.success(
                    request,
                    f"{symbol} market intelligence updated: {result['fetched']} new "
                    f"observations from {source_count} sources; {result['saved']} saved.",
                )
            else:
                messages.warning(
                    request,
                    f"No market intelligence was collected for {symbol}. Configure "
                    "a supported provider or API key and try again.",
                )
            if error_count:
                failed = ", ".join(sorted(result["errors"]))
                messages.warning(
                    request,
                    f"Some {symbol} providers were unavailable: {failed}.",
                )
        return redirect("predict", symbol_name=symbol)


class MarketRecommendationsView(View):
    template_name = "market_recommendations.html"

    def get_market(self, market_slug):
        for market_name, symbols in list_available_assets().items():
            if slugify(market_name) == market_slug:
                return market_name, symbols
        raise Http404("Unknown dashboard market.")

    def post(self, request, market_slug):
        market_name, symbols = self.get_market(market_slug)
        try:
            result = news_service.refresh_market_news(symbols)
        except (OSError, RuntimeError, ValueError) as exc:
            messages.error(
                request,
                f"Could not refresh news for {market_name}: {exc}",
            )
            return redirect("dashboard")

        return render(
            request,
            self.template_name,
            {
                "market_name": market_name,
                **result,
            },
        )


class PriceCsvDownloadView(View):
    def get(self, request, symbol_name):
        try:
            symbol = normalize_symbol(symbol_name)
        except ValueError as exc:
            raise Http404(str(exc)) from exc
        if symbol not in available_symbols():
            raise Http404("Unknown dashboard asset.")

        try:
            with latest_price_csv(symbol) as path:
                csv_content = path.read_bytes()
        except (OSError, RuntimeError, ValueError) as exc:
            messages.error(request, f"Could not download prices for {symbol}: {exc}")
            return redirect("dashboard")

        response = HttpResponse(csv_content, content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = (
            f'attachment; filename="{symbol}_latest_H1.csv"'
        )
        return response
