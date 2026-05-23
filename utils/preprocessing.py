import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def engineer_features(df_transactions):
    """
    Engineers behavioral features from transaction data.
    Fixes Pandas frequency issue by using freq='ME'.
    """
    # 1. Monthly aggregated data using freq='ME' to avoid the ValueError
    # Separate incomes and expenses
    incomes = df_transactions[df_transactions['transaction_type'] == 'income']
    expenses = df_transactions[df_transactions['transaction_type'] == 'expense']
    
    monthly_incomes = incomes.groupby(['customer_id', pd.Grouper(key='transaction_date', freq='ME')])['transaction_amount'].sum().reset_index()
    monthly_expenses = expenses.groupby(['customer_id', pd.Grouper(key='transaction_date', freq='ME')])['transaction_amount'].sum().reset_index()
    
    # 2. Behavioral Metrics Calculation per customer
    # Calculate income volatility (std / mean)
    inc_stats = monthly_incomes.groupby('customer_id')['transaction_amount'].agg(['mean', 'std']).reset_index()
    inc_stats['income_volatility'] = inc_stats['std'] / inc_stats['mean'].replace(0, np.nan)
    inc_stats['income_volatility'] = inc_stats['income_volatility'].fillna(0)
    inc_stats.rename(columns={'mean': 'avg_monthly_income'}, inplace=True)
    inc_stats.drop(columns=['std'], inplace=True)
    
    # Calculate expense consistency
    exp_stats = monthly_expenses.groupby('customer_id')['transaction_amount'].agg(['mean', 'std']).reset_index()
    exp_stats['expense_volatility'] = exp_stats['std'] / exp_stats['mean'].replace(0, np.nan)
    exp_stats['expense_volatility'] = exp_stats['expense_volatility'].fillna(0)
    exp_stats.rename(columns={'mean': 'avg_monthly_expense'}, inplace=True)
    exp_stats.drop(columns=['std'], inplace=True)
    
    # 3. RFM Features
    max_date = df_transactions['transaction_date'].max()
    rfm = expenses.groupby('customer_id').agg({
        'transaction_date': lambda x: (max_date - x.max()).days,
        'transaction_amount': ['count', 'sum']
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Merge all features
    features = inc_stats.merge(exp_stats, on='customer_id', how='left').fillna(0)
    features = features.merge(rfm, on='customer_id', how='left').fillna(0)
    
    # Additional Derived Features
    features['savings_ratio'] = (features['avg_monthly_income'] - features['avg_monthly_expense']) / features['avg_monthly_income'].replace(0, 1)
    features['savings_ratio'] = features['savings_ratio'].clip(lower=-1, upper=1)
    
    # Add metadata back
    meta = df_transactions[['customer_id', 'employment_type', 'gender', 'age_group', 'loan_default']].drop_duplicates()
    features = features.merge(meta, on='customer_id', how='left')
    
    return features

def preprocess_for_segmentation(features_df):
    """
    Prepares data for KMeans/PCA by scaling numeric features.
    """
    numeric_cols = ['avg_monthly_income', 'income_volatility', 'avg_monthly_expense', 
                   'expense_volatility', 'recency', 'frequency', 'monetary', 'savings_ratio']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df[numeric_cols])
    
    return scaled_data, scaler, numeric_cols

def preprocess_for_classification(features_df):
    """
    Prepares data for XGBoost. Applies One-Hot Encoding and SMOTE.
    """
    df = features_df.copy()
    df.drop(columns=['customer_id'], inplace=True)
    if 'cluster' in df.columns:
        df.drop(columns=['cluster'], inplace=True)
    
    # One-hot encode categorical variables
    cat_cols = ['employment_type', 'gender', 'age_group']
    if 'segment' in df.columns:
        cat_cols.append('segment')
        
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df.drop(columns=['loan_default'])
    y = df['loan_default']
    
    # SMOTE to handle imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    return X_res, y_res, X.columns

def prepare_lifetimes_data(df_transactions):
    """
    Transforms transaction data into a format suitable for Lifetimes modeling.
    Returns a DataFrame with frequency, recency, T, and monetary_value.
    """
    import lifetimes
    
    # Filter only expenses for CLV
    expenses = df_transactions[df_transactions['transaction_type'] == 'expense']
    
    summary = lifetimes.utils.summary_data_from_transaction_data(
        expenses,
        customer_id_col='customer_id',
        datetime_col='transaction_date',
        monetary_value_col='transaction_amount',
        observation_period_end=expenses['transaction_date'].max()
    )
    return summary
