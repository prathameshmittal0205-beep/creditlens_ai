import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

def train_clv_models(summary_df):
    """
    Trains BG/NBD and Gamma-Gamma models for CLV prediction.
    """
    # Filter out customers with zero repeat purchases for Gamma-Gamma
    # Gamma-Gamma requires frequency > 0 and monetary_value > 0
    returning_customers = summary_df[(summary_df['frequency'] > 0) & (summary_df['monetary_value'] > 0)]
    
    # 1. Train BG/NBD (Beta-Geometric / Negative Binomial Distribution)
    # This predicts the number of future transactions
    bgf = None
    for penalizer in [0.0, 0.01, 0.1, 1.0, 10.0]:
        try:
            bgf = BetaGeoFitter(penalizer_coef=penalizer)
            bgf.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])
            break
        except Exception:
            continue
            
    # 2. Train Gamma-Gamma
    # This predicts the average monetary value per transaction
    ggf = None
    if len(returning_customers) > 0:
        for penalizer in [0.0, 0.01, 0.1, 1.0, 10.0]:
            try:
                ggf = GammaGammaFitter(penalizer_coef=penalizer)
                ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
                break
            except Exception:
                continue
                
    return bgf, ggf

def predict_clv_metrics(bgf, ggf, summary_df, t=6, discount_rate=0.01):
    """
    Generates CLV metrics: expected purchases, retention probability, and CLV score.
    t: periods to forecast (e.g., 6 months)
    """
    df = summary_df.copy()
    
    # Expected number of transactions in next t periods
    df['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, df['frequency'], df['recency'], df['T']
    )
    
    # Probability that customer is still "alive" (active)
    df['retention_probability'] = bgf.conditional_probability_alive(
        df['frequency'], df['recency'], df['T']
    )
    
    # Calculate CLV if Gamma-Gamma model was trained
    if ggf is not None:
        # Expected average profit per transaction
        df['expected_avg_spend'] = ggf.conditional_expected_average_profit(
            df['frequency'], df['monetary_value']
        )
        
        # Calculate overall CLV
        df['clv_score'] = ggf.customer_lifetime_value(
            bgf,
            df['frequency'],
            df['recency'],
            df['T'],
            df['monetary_value'],
            time=t, # months
            discount_rate=discount_rate
        )
    else:
        df['expected_avg_spend'] = 0
        df['clv_score'] = 0
        
    return df
