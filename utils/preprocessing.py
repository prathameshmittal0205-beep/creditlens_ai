import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def process_uploaded_data(df):
    """
    Cleans and auto-maps uploaded datasets to internal schema.
    Infers missing employment_type if needed.
    """
    # 1. Column mapping (heuristics)
    col_map = {}
    lower_cols = [c.lower() for c in df.columns]
    
    # Date mapping
    for c, lc in zip(df.columns, lower_cols):
        if 'date' in lc or 'time' in lc:
            col_map[c] = 'transaction_date'
            break
            
    # Amount mapping
    for c, lc in zip(df.columns, lower_cols):
        if 'amount' in lc or 'value' in lc or 'spend' in lc:
            col_map[c] = 'transaction_amount'
            break
            
    # Type mapping
    for c, lc in zip(df.columns, lower_cols):
        if 'type' in lc or 'category' in lc:
            col_map[c] = 'transaction_type'
            break
            
    # Customer ID mapping
    for c, lc in zip(df.columns, lower_cols):
        if 'user' in lc or 'customer' in lc or 'client' in lc or 'id' in lc:
            col_map[c] = 'customer_id'
            break

    df = df.rename(columns=col_map)
    
    # Ensure mandatory columns exist
    mandatory = ['customer_id', 'transaction_date', 'transaction_amount']
    missing = [c for c in mandatory if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded dataset is missing required columns or could not map them: {missing}")

    # Ensure transaction type exists
    if 'transaction_type' not in df.columns:
        # Infer transaction type: Negative amounts are expenses, positive are incomes
        df['transaction_type'] = np.where(df['transaction_amount'] < 0, 'expense', 'income')
        df['transaction_amount'] = df['transaction_amount'].abs()

    # Drop missing core values
    df.dropna(subset=['customer_id', 'transaction_date', 'transaction_amount'], inplace=True)

    # Parse dates robustly
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date'], inplace=True)
    
    # Clean amount
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    df.dropna(subset=['transaction_amount'], inplace=True)
    
    # Standardize types
    df['transaction_type'] = df['transaction_type'].astype(str).str.lower()
    df['transaction_type'] = np.where(df['transaction_type'].str.contains('income|credit|deposit'), 'income', 'expense')

    # Infer employment_type if missing
    if 'employment_type' not in df.columns:
        # Heuristic: Find number of distinct income transactions per month per user
        incomes = df[df['transaction_type'] == 'income']
        monthly_incomes = incomes.groupby(['customer_id', pd.Grouper(key='transaction_date', freq='ME')])['transaction_amount'].count().reset_index()
        avg_monthly_incomes = monthly_incomes.groupby('customer_id')['transaction_amount'].mean().to_dict()
        
        def infer_employment(cid):
            avg = avg_monthly_incomes.get(cid, 0)
            if avg == 1: return 'salaried'
            elif avg > 3: return 'gig'
            else: return 'self-employed'
            
        df['employment_type'] = df['customer_id'].map(infer_employment)

    # Basic default rate if target is missing
    if 'loan_default' not in df.columns:
        df['loan_default'] = 0 # Will be populated synthetically or assumed 0

    if 'gender' not in df.columns:
        df['gender'] = 'Unknown'
        
    if 'age_group' not in df.columns:
        df['age_group'] = 'Unknown'
        
    return df

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
    Ensures no data leakage by leaving SMOTE for the credit_scoring file to handle after split.
    """
    df = features_df.copy()
    df.drop(columns=['customer_id'], inplace=True)
    if 'cluster' in df.columns:
        df.drop(columns=['cluster'], inplace=True)
    
    # One-hot encode categorical variables
    cat_cols = ['employment_type', 'gender', 'age_group']
    if 'segment' in df.columns:
        cat_cols.append('segment')
        
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    
    # Ensure we have a target
    if 'loan_default' not in df.columns:
        df['loan_default'] = 0
        
    y = df['loan_default']
    
    # Deterministic behavior-driven label regeneration if classes are imbalanced
    if y.nunique() < 2:
        print("--- Regenerating deterministic behavioral labels ---")
        risk_score = np.zeros(len(df))
        
        if 'savings_ratio' in features_df.columns:
            risk_score += np.where(features_df['savings_ratio'] < 0.1, 1, 0)
            
        if 'avg_monthly_expense' in features_df.columns and 'avg_monthly_income' in features_df.columns:
            expense_ratio = features_df['avg_monthly_expense'] / features_df['avg_monthly_income'].replace(0, 1)
            risk_score += np.where(expense_ratio > 0.85, 1, 0)
            
        if 'income_volatility' in features_df.columns:
            med_inc_vol = features_df['income_volatility'].median()
            risk_score += np.where(features_df['income_volatility'] > med_inc_vol, 1, 0)
            
        if 'expense_volatility' in features_df.columns:
            med_exp_vol = features_df['expense_volatility'].median()
            risk_score += np.where(features_df['expense_volatility'] > med_exp_vol, 1, 0)
            
        # Deterministic tie-breaker
        tie_breaker = (np.arange(len(df)) % 100) / 1000.0
        risk_score += tie_breaker
        
        # Guarantee 35%-65% split
        threshold = np.percentile(risk_score, 65)
        df['loan_default'] = (risk_score >= threshold).astype(int)
        y = df['loan_default']
        
    print("--- Diagnostics: Target Distribution ---")
    print(y.value_counts())
    print("----------------------------------------")
    
    if y.nunique() < 2:
        raise ValueError("Target variable y must contain at least two classes (0 and 1). Behavior-driven regeneration failed.")
        
    X = df.drop(columns=['loan_default'])
    return X, y, X.columns

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
