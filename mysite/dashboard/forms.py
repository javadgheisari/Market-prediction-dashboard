from django import forms

MODEL_CHOICES = (
    ("xgboost", "XGBoost"),
    ("gru", "Improved GRU"),
    ("lstm", "Improved LSTM"),
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

    def clean(self):
        cleaned_data = super().clean()
        uploaded_file = cleaned_data.get("market_csv")
        if cleaned_data.get("price_source") == "upload":
            if uploaded_file is None:
                self.add_error("market_csv", "Choose a historical price CSV file.")
            elif not uploaded_file.name.lower().endswith(".csv"):
                self.add_error("market_csv", "The historical price file must be a CSV.")
            elif uploaded_file.size > MAX_UPLOAD_SIZE:
                self.add_error("market_csv", "The CSV file must be 50 MB or smaller.")
        return cleaned_data
