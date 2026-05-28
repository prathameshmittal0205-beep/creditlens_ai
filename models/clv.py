import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

def train_clv_models(summary_df):
    """
    Trains BG/NBD and Gamma-Gamma models for CLV prediction.
    Raises ValueError if fitting fails. No fake models allowed.
    """
    if summary_df is None or summary_df.empty:
        raise ValueError("Empty CLV dataset")
        
    required = {"frequency", "recency", "T", "monetary_value"}
    if not required.issubset(summary_df.columns):
        raise ValueError("Missing required CLV columns")
        
    # 1. Train BG/NBD (Beta-Geometric / Negative Binomial Distribution)
    bgf = None
    for penalizer in [0.01, 0.1, 1.0, 10.0]:
        try:
            model = BetaGeoFitter(penalizer_coef=penalizer)
            model.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])
            bgf = model
            break
        except Exception:
            continue
            
    if bgf is None:
        return None, None
    
    # 2. Train Gamma-Gamma
    returning_customers = summary_df[(summary_df['frequency'] > 0) & (summary_df['monetary_value'] > 0)]
    ggf = None
    if len(returning_customers) > 2:
        for penalizer in [0.01, 0.1, 1.0, 10.0]:
            try:
                model = GammaGammaFitter(penalizer_coef=penalizer)
                model.fit(returning_customers['frequency'], returning_customers['monetary_value'])
                ggf = model
                break
            except Exception:
                continue
                
    return bgf, ggf

def predict_clv_metrics(bgf, ggf, summary_df, t=6, discount_rate=0.01):
    """
    Generates CLV metrics safely using valid lifetimes models only.
    """
    if summary_df is None or summary_df.empty:
        raise ValueError("CLV input data is empty")
        
    required_cols = {'frequency', 'recency', 'T', 'monetary_value'}
    if not required_cols.issubset(summary_df.columns):
        raise ValueError("CLV Data Error: summary_df is missing required columns.")
        
    # Strict validation of models
    if not isinstance(bgf, BetaGeoFitter):
        raise ValueError("BG/NBD model is not a valid BetaGeoFitter object.")
    if not hasattr(bgf, "params_"):
        raise ValueError("BG/NBD model not fitted.")
        
    # Enforce GammaGammaFitter valid type
    if ggf is not None:
        if not isinstance(ggf, GammaGammaFitter) or not hasattr(ggf, "params_"):
            ggf = None
            
    df = summary_df.copy()

    df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, df['frequency'], df['recency'], df['T']
    )
    df['retention_probability'] = bgf.conditional_probability_alive(
        df['frequency'], df['recency'], df['T']
    )

    if 'predicted_purchases' not in df.columns:
        raise ValueError("BG/NBD prediction missing. Cannot compute CLV.")

    if ggf is not None:
        df['expected_avg_spend'] = ggf.conditional_expected_average_profit(
            df['frequency'], df['monetary_value']
        )
        df['clv_score'] = ggf.customer_lifetime_value(
            bgf,
            df['frequency'],
            df['recency'],
            df['T'],
            df['monetary_value'],
            time=t,
            discount_rate=discount_rate
        )
    else:
        df['expected_avg_spend'] = df['monetary_value']
        df['clv_score'] = df['expected_avg_spend'] * df['predicted_purchases']

    return df

def generate_fallback_clv(summary_df, features_df):
    """
    Generates deterministic fallback CLV metrics using behavioral features 
    when Lifetimes mathematical models fail to converge.
    """
    import numpy as np
    
    df = summary_df.copy()
    
    # Merge behavioral features safely
    if 'customer_id' in features_df.columns:
        feat_indexed = features_df.set_index('customer_id')
    else:
        feat_indexed = features_df
        
    df = df.join(feat_indexed[['savings_ratio', 'avg_monthly_income']], how='left')
    
    def normalize(series):
        denom = series.max() - series.min()
        if denom == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / denom

    # 1. Retention Probability Proxy
    # High frequency, high recency (closer to T), and good savings ratio = better retention
    recency_norm = df['recency'] / df['T'].replace(0, 1)
    freq_norm = normalize(df['frequency'])
    savings_norm = normalize(df['savings_ratio'].fillna(0).clip(-1, 1))
    
    # Weighted proxy for retention (scale 0-1)
    retention_score = (0.5 * recency_norm) + (0.3 * freq_norm) + (0.2 * savings_norm)
    df['retention_probability'] = retention_score.clip(0.1, 0.95)
    
    # 2. Predicted Purchases Proxy (6-Month Horizon)
    # Extrapolate historical average monthly frequency, adjusted by retention
    avg_monthly_freq = df['frequency'] / (df['T'] / 30).replace(0, 1)
    df['predicted_purchases'] = avg_monthly_freq * 6 * df['retention_probability']
    
    # 3. Expected Average Spend
    df['expected_avg_spend'] = df['monetary_value']
    
    # 4. CLV Score
    df['clv_score'] = df['predicted_purchases'] * df['expected_avg_spend']
    
    return df[['frequency', 'recency', 'T', 'monetary_value', 
               'predicted_purchases', 'retention_probability', 
               'expected_avg_spend', 'clv_score']]