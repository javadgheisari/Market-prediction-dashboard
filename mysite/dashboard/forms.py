from django import forms

MODEL_CHOICES =(
    ("lstm", "lstm"),
    ("gru", "gru"),
)

class NameForm(forms.Form):
    model_name = forms.ChoiceField(choices=MODEL_CHOICES, label="model",)
    predict_days = forms.IntegerField(label='predict_days',)