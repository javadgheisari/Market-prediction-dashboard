import re
import tempfile
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import redirect
from django.shortcuts import render
from django.utils.text import get_valid_filename
from django.views import View

from dashboard import forecaster, market_data_service, news_service
from dashboard.data_paths import has_bundled_data, list_available_assets
from dashboard.forms import CompleteCsvForm, NameForm


SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.-]+$")


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
            {
                "asset_groups": list_available_assets(),
                "complete_csv_form": CompleteCsvForm(),
            },
        )


class CompleteCsvView(View):
    """Complete an uploaded price CSV and return it as a download."""

    def post(self, request):
        form = CompleteCsvForm(request.POST, request.FILES)
        if not form.is_valid():
            errors = "; ".join(
                message
                for messages_list in form.errors.values()
                for message in messages_list
            )
            messages.error(request, errors or "Choose a valid CSV and symbol.")
            return redirect("dashboard")

        symbol = form.cleaned_data["symbol"]
        uploaded_file = form.cleaned_data["market_csv"]
        try:
            with temporary_price_csv(uploaded_file, symbol) as path:
                frame = market_data_service.merge_latest_prices(path, symbol)
        except (OSError, RuntimeError, ValueError) as exc:
            messages.error(request, f"Could not complete the CSV: {exc}")
            return redirect("dashboard")

        output = StringIO()
        frame.to_csv(output, index=False, lineterminator="\n")
        filename = get_valid_filename(f"{Path(uploaded_file.name).stem}_completed.csv")
        response = HttpResponse(
            output.getvalue(), content_type="text/csv; charset=utf-8"
        )
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class PredictView(View):
    template_name = "predict.html"
    form_class = NameForm

    def get_symbol(self, symbol_name):
        symbol = symbol_name.strip().upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError("The asset symbol contains unsupported characters.")
        return symbol

    def base_context(self, symbol, form):
        return {
            "symbol_name": symbol,
            "form": form,
            "bundled_available": has_bundled_data(symbol),
            "asset_news": news_service.load_asset_news(symbol),
        }

    def get(self, request, symbol_name):
        try:
            symbol = self.get_symbol(symbol_name)
            form = self.form_class(
                initial={"predict_steps": 24},
                bundled_available=has_bundled_data(symbol),
            )
            error = None
        except ValueError as exc:
            symbol = symbol_name
            form = self.form_class()
            error = str(exc)

        return render(
            request,
            self.template_name,
            {**self.base_context(symbol, form), "error": error},
        )

    def post(self, request, symbol_name):
        symbol = symbol_name.strip().upper()
        form = self.form_class(
            request.POST,
            request.FILES,
            bundled_available=has_bundled_data(symbol),
        )
        context = self.base_context(symbol, form)

        if form.is_valid():
            try:
                symbol = self.get_symbol(symbol_name)
                model_name = form.cleaned_data["model_name"]
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
                        use_news=data_source == "with_news",
                    )
                context.update(
                    {
                        "symbol_name": symbol,
                        "model_name": model_name.upper(),
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
