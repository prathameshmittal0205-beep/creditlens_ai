import shap
import pandas as pd
import numpy as np

def generate_shap_values(model, X_data):
    """
    Generates SHAP explainer and values for the XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_data)
    
    return explainer, shap_values

def get_natural_language_explanation(shap_values_instance, feature_names, is_approved):
    """
    Generates a natural language explanation of the decision based on SHAP values.
    """
    # Create DataFrame of feature impacts for this instance
    impacts = pd.DataFrame({
        'feature': feature_names,
        'value': shap_values_instance.values
    })
    
    # Sort by absolute impact
    impacts['abs_value'] = impacts['value'].abs()
    impacts = impacts.sort_values('abs_value', ascending=False).head(3)
    
    top_features = impacts['feature'].tolist()
    
    # Format nicely
    top_features_clean = [f.replace('_', ' ').title() for f in top_features]
    
    if is_approved:
        return f"This customer was approved primarily due to strong metrics in: {', '.join(top_features_clean)}."
    else:
        return f"This customer was flagged for review/rejection primarily due to risk factors in: {', '.join(top_features_clean)}."

def calculate_fairness_metrics(y_true, y_pred, df_meta, protected_attribute='gender'):
    """
    Calculates fairness metrics (e.g., Approval Rate Parity) across demographics.
    y_true: actual default (1 = default, 0 = paid)
    y_pred: predicted default
    df_meta: contains the protected_attribute column aligned with y_pred index
    """
    # Ensure alignment
    df = df_meta.copy()
    df['y_true'] = y_true.values
    df['y_pred'] = y_pred
    
    # Approval is when we predict NO DEFAULT (y_pred == 0)
    df['approved'] = (df['y_pred'] == 0).astype(int)
    
    # Group by protected attribute
    metrics = df.groupby(protected_attribute).agg(
        total_count=('y_true', 'count'),
        approval_rate=('approved', 'mean'),
        actual_default_rate=('y_true', 'mean')
    ).reset_index()
    
    return metrics
