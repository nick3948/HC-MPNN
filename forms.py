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
    use_triton = forms.BooleanField(
        required=False,
        initial=False,
        label="Use Triton Acceleration",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    num_layer = forms.IntegerField(
        initial=1,
        min_value=1,
        max_value=5,
        label="Number of GNN Layers",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )