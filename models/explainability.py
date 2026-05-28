import shap
import pandas as pd
import numpy as np

def generate_shap_values(model, X_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_data)
    return explainer, shap_values


def _extract_vals(shap_values_instance):
    """
    Safe SHAP extractor for all SHAP formats.
    """
    vals = shap_values_instance.values if hasattr(shap_values_instance, "values") else shap_values_instance

    vals = np.array(vals)

    # multiclass fix
    if vals.ndim == 3:
        vals = vals[:, :, 1]  # take positive class

    return vals


def get_natural_language_explanation(shap_values_instance, feature_names, is_approved):

    vals = _extract_vals(shap_values_instance)

    # safety check (IMPORTANT)
    if len(feature_names) != vals.shape[-1]:
        raise ValueError(
            f"Feature mismatch: feature_names={len(feature_names)} but SHAP={vals.shape}"
        )

    impacts = pd.DataFrame({
        'feature': feature_names,
        'value': vals
    })

    impacts['abs_value'] = impacts['value'].abs()
    impacts = impacts.sort_values('abs_value', ascending=False)

    top_features = impacts.head(3)['feature'].tolist()
    top_features_clean = [f.replace('_', ' ').title() for f in top_features]

    if is_approved:
        explanation = (
            f"Approved due to strong signals in: {', '.join(top_features_clean)}."
        )
    else:
        explanation = (
            f"Flagged due to risk signals in: {', '.join(top_features_clean)}."
        )

    return explanation, impacts


def calculate_fairness_metrics(y_true, y_pred, df_meta, protected_attribute='gender'):

    df = df_meta.copy()
    df['y_true'] = np.array(y_true)
    df['y_pred'] = np.array(y_pred)

    df['approved'] = (df['y_pred'] == 0).astype(int)

    metrics = df.groupby(protected_attribute).agg(
        total_count=('y_true', 'count'),
        approval_rate=('approved', 'mean'),
        actual_default_rate=('y_true', 'mean')
    ).reset_index()

    return metrics