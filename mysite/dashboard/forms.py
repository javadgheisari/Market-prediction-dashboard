from django import forms
from django.utils.text import slugify

MODEL_CHOICES = (
    ("xgboost", "XGBoost Portfolio (recommended)"),
    ("news_buy", "News-only buy decision"),
    ("chronos", "Chronos-2 (optional download)"),
    ("gru", "Improved GRU"),
    ("lstm", "Improved LSTM"),
)

RECOMMENDATION_MODEL_CHOICES = (
    ("ensemble", "News + XGBoost + GRU + LSTM (recommended)"),
    ("model_ensemble", "XGBoost + GRU + LSTM (news optional)"),
    ("xgboost", "XGBoost Portfolio"),
    *MODEL_CHOICES[1:],
)

ADVANCED_CONFIG_FIELDS = (
    "window",
    "split_ratio",
    "validation_ratio",
    "portfolio_size",
    "recency_half_life",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "epochs",
    "chronos_model_id",
    "chronos_context_length",
    "chronos_device",
)

DATA_SOURCE_CHOICES = (
    ("price_only", "Price data only"),
    ("with_news", "Price + crawled news"),
)

PRICE_SOURCE_CHOICES = (
    ("bundled", "Use project /data CSV"),
    ("forexsb", "Download H1 history automatically"),
    ("upload", "Choose CSV from this computer"),
)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024


class RecommendationForm(forms.Form):
    market = forms.ChoiceField(
        label="Market",
        widget=forms.Select(
            attrs={"class": "form-select", "data-recommendation-market": "1"}
        ),
    )
    assets = forms.MultipleChoiceField(
        label="Assets",
        required=False,
        widget=forms.SelectMultiple(
            attrs={
                "class": "form-select recommendation-assets-select",
                "size": "9",
                "data-recommendation-assets": "1",
            }
        ),
        help_text="Leave every asset unselected to rank the full market.",
    )
    model_name = forms.ChoiceField(
        choices=RECOMMENDATION_MODEL_CHOICES,
        label="Ranking model",
        initial="ensemble",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    use_news = forms.BooleanField(
        label="Include crawled news features",
        required=False,
        initial=True,
        help_text=(
            "The news ensemble and news-only modes always use news. The "
            "three-model ensemble and individual models can optionally combine "
            "price history with timestamped news features."
        ),
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )
    refresh_news = forms.BooleanField(
        label="Clear and crawl fresh news before ranking",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )
    predict_steps = forms.IntegerField(
        label="Forecast steps",
        min_value=1,
        max_value=48,
        initial=24,
        help_text="Used by price models; one step is one source candle.",
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    def __init__(self, *args, asset_groups=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.asset_groups = asset_groups or {}
        self.market_symbols = {
            slugify(market): list(symbols)
            for market, symbols in self.asset_groups.items()
        }
        self.market_names = {
            slugify(market): market for market in self.asset_groups
        }
        all_symbols = list(
            dict.fromkeys(
                symbol
                for symbols in self.asset_groups.values()
                for symbol in symbols
            )
        )
        self.market_symbols["all-markets"] = all_symbols
        self.market_names["all-markets"] = "All markets"
        self.fields["market"].choices = [
            (market_slug, self.market_names[market_slug])
            for market_slug in self.market_symbols
        ]
        self.fields["assets"].choices = [
            (market, [(symbol, symbol) for symbol in symbols])
            for market, symbols in self.asset_groups.items()
        ]
        if self.asset_groups:
            self.fields["market"].initial = slugify(next(iter(self.asset_groups)))

    def clean(self):
        cleaned_data = super().clean()
        market = cleaned_data.get("market")
        selected = cleaned_data.get("assets") or []
        allowed = set(self.market_symbols.get(market, []))
        invalid = [symbol for symbol in selected if symbol not in allowed]
        if invalid:
            self.add_error(
                "assets",
                "Choose assets that belong to the selected market.",
            )

        if cleaned_data.get("model_name") in {"ensemble", "news_buy"}:
            cleaned_data["use_news"] = True
        if not cleaned_data.get("use_news"):
            cleaned_data["refresh_news"] = False
        return cleaned_data

    def selected_symbols(self):
        if not hasattr(self, "cleaned_data"):
            raise ValueError("Call is_valid() before selected_symbols().")
        market = self.cleaned_data["market"]
        return self.cleaned_data.get("assets") or self.market_symbols[market]

    def selected_market_name(self):
        if not hasattr(self, "cleaned_data"):
            raise ValueError("Call is_valid() before selected_market_name().")
        return self.market_names[self.cleaned_data["market"]]


class NameForm(forms.Form):
    model_name = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Model",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    data_source = forms.ChoiceField(
        choices=DATA_SOURCE_CHOICES,
        label="Training data",
        initial="price_only",
        widget=forms.Select(attrs={"class": "form-select"}),
        help_text=(
            "Crawl news first. Price + crawled news automatically uses prices "
            "from the first saved article onward."
        ),
    )
    price_source = forms.ChoiceField(
        choices=PRICE_SOURCE_CHOICES,
        label="Historical price data",
        initial="bundled",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    market_csv = forms.FileField(
        label="Historical price CSV",
        required=False,
        widget=forms.ClearableFileInput(
            attrs={"class": "form-control", "accept": ".csv,text/csv"}
        ),
        help_text="Required only when choosing a CSV from this computer (maximum 50 MB).",
    )
    predict_steps = forms.IntegerField(
        label="Forecast steps",
        min_value=1,
        max_value=48,
        initial=24,
        help_text="One step equals one candle in the selected asset's CSV.",
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    window = forms.IntegerField(
        label="Input window",
        min_value=20,
        max_value=240,
        initial=60,
        required=False,
        help_text="Historical candles per sample for GRU, LSTM, and XGBoost.",
        widget=forms.NumberInput(
            attrs={"class": "form-control", "data_advanced": "1", "data_models": "xgboost gru lstm"}
        ),
    )
    split_ratio = forms.FloatField(
        label="Train/validation share",
        min_value=0.6,
        max_value=0.9,
        initial=0.8,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "step": "0.05", "data_advanced": "1", "data_models": "all"}
        ),
    )
    validation_ratio = forms.FloatField(
        label="Validation share inside training",
        min_value=0.1,
        max_value=0.4,
        initial=0.2,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "step": "0.05", "data_advanced": "1", "data_models": "xgboost gru lstm"}
        ),
    )
    portfolio_size = forms.IntegerField(
        label="XGBoost portfolio members",
        min_value=1,
        max_value=3,
        initial=3,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "data_advanced": "1", "data_models": "xgboost"}
        ),
    )
    recency_half_life = forms.FloatField(
        label="Recency half-life (dataset fraction)",
        min_value=0.1,
        max_value=2.0,
        initial=0.5,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "step": "0.1", "data_advanced": "1", "data_models": "xgboost"}
        ),
    )
    n_estimators = forms.IntegerField(
        label="Maximum XGBoost trees",
        min_value=50,
        max_value=1000,
        initial=300,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "data_advanced": "1", "data_models": "xgboost"}
        ),
    )
    max_depth = forms.IntegerField(
        label="XGBoost maximum depth",
        min_value=2,
        max_value=8,
        initial=3,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "data_advanced": "1", "data_models": "xgboost"}
        ),
    )
    learning_rate = forms.FloatField(
        label="XGBoost learning rate",
        min_value=0.005,
        max_value=0.2,
        initial=0.03,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "step": "0.005", "data_advanced": "1", "data_models": "xgboost"}
        ),
    )
    epochs = forms.IntegerField(
        label="GRU/LSTM maximum epochs",
        min_value=5,
        max_value=300,
        initial=50,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "data_advanced": "1", "data_models": "gru lstm"}
        ),
    )
    chronos_model_id = forms.ChoiceField(
        label="Chronos checkpoint",
        choices=(
            ("amazon/chronos-2", "Chronos-2 (120M)"),
            ("autogluon/chronos-2-small", "Chronos-2 Small (28M)"),
            ("autogluon/chronos-2-synth", "Chronos-2 Synth (120M)"),
        ),
        initial="amazon/chronos-2",
        required=False,
        widget=forms.Select(
            attrs={"class": "form-select", "data_advanced": "1", "data_models": "chronos"}
        ),
    )
    chronos_context_length = forms.IntegerField(
        label="Chronos context candles",
        min_value=24,
        max_value=4096,
        initial=512,
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "data_advanced": "1", "data_models": "chronos"}
        ),
    )
    chronos_device = forms.ChoiceField(
        label="Chronos device",
        choices=(("cpu", "CPU"), ("auto", "Automatic"), ("cuda", "NVIDIA CUDA")),
        initial="cpu",
        required=False,
        widget=forms.Select(
            attrs={"class": "form-select", "data_advanced": "1", "data_models": "chronos"}
        ),
    )

    def __init__(
        self, *args, bundled_available=True, forexsb_available=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not bundled_available:
            choices = []
            if forexsb_available:
                choices.append(("forexsb", "Download H1 history automatically"))
            choices.append(("upload", "Choose CSV from this computer"))
            self.fields["price_source"].choices = tuple(choices)
            self.fields["price_source"].initial = choices[0][0]
        if forexsb_available:
            self.fields["price_source"].help_text = (
                "Automatic history uses ForexSB one-hour data; BiQuote then adds "
                "the latest closed candles."
            )

    def advanced_config(self):
        """Return validated advanced values, filling omitted optional fields."""
        if not hasattr(self, "cleaned_data"):
            raise ValueError("Call is_valid() before advanced_config().")
        return {
            name: self.cleaned_data.get(name)
            if self.cleaned_data.get(name) not in (None, "")
            else self.fields[name].initial
            for name in ADVANCED_CONFIG_FIELDS
        }

    def clean(self):
        cleaned_data = super().clean()
        uploaded_file = cleaned_data.get("market_csv")
        if (
            cleaned_data.get("model_name") != "news_buy"
            and cleaned_data.get("price_source") == "upload"
        ):
            if uploaded_file is None:
                self.add_error("market_csv", "Choose a historical price CSV file.")
            elif not uploaded_file.name.lower().endswith(".csv"):
                self.add_error("market_csv", "The historical price file must be a CSV.")
            elif uploaded_file.size > MAX_UPLOAD_SIZE:
                self.add_error("market_csv", "The CSV file must be 50 MB or smaller.")
        return cleaned_data
