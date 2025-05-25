from django import forms

DATASETS = [
    ('FB-AUTO', 'FB-AUTO'),
    ('WP-IND', 'WP-IND'),
]

EVAL_TYPES = [
    ('raw', 'Raw'),
    ('filtered', 'Filtered'),
    ('both', 'Both'),
]

class PredictionForm(forms.Form):
    dataset = forms.ChoiceField(
        choices=DATASETS,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    epochs = forms.IntegerField(
        min_value=1,
        max_value=20,
        initial=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    eval_type = forms.ChoiceField(
        choices=EVAL_TYPES,
        initial='both',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3
        }),
        help_text="Optional notes for the run."
    )