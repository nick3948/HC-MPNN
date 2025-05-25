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
    dataset = forms.ChoiceField(choices=DATASETS)
    epochs = forms.IntegerField(min_value=1, max_value=20, initial=5)
    eval_type = forms.ChoiceField(choices=EVAL_TYPES, initial='both')
    notes = forms.CharField(required=False, widget=forms.Textarea, help_text="Optional notes for the run.")
