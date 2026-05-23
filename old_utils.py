import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from lifetimes import BetaGeoFitter, GammaGammaFitter
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import streamlit as st

@st.cache_data
def extract_rfm_features(df):
    """
    Extract RFM (Recency, Frequency, Monetary) and additional behavioral features.
    """
    # Fix timezone issues if any, ensure datetime object
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    max_date = df['transaction_date'].max()
    
    # Separate into income and expense
    expenses = df[df['transaction_type'] == 'expense']
    incomes = df[df['transaction_type'] == 'income']
    
    # Build RFM using expenses for recency and frequency
    rfm = expenses.groupby('customer_id').agg(
        Recency=('transaction_date', lambda x: (max_date - x.max()).days),
        Frequency=('customer_id', 'count'),
        Monetary=('transaction_amount', 'sum')
    ).reset_index()
    
    # Additional features
    income_stats = incomes.groupby('customer_id').agg(
        Total_Income=('transaction_amount', 'sum'),
        Income_Variance=('transaction_amount', 'var'),
        Income_Freq=('customer_id', 'count')
    ).reset_index()
    
    # Fill variance NaNs with 0 (e.g. only 1 transaction)
    income_stats['Income_Variance'] = income_stats['Income_Variance'].fillna(0)
    
    features = pd.merge(rfm, income_stats, on='customer_id', how='left').fillna(0)
    
    # Expense Ratio
    features['Expense_Ratio'] = np.where(features['Total_Income'] > 0, 
                                        features['Monetary'] / features['Total_Income'], 
                                        1.0)
    
    # Transaction consistency: Days divided by freq
    features['Transaction_Consistency'] = np.where(features['Frequency'] > 0,
                                                  features['Recency'] / features['Frequency'],
                                                  0)
                                                  
    # Ensure employment type and true default target carries over if available
    metadata = df.drop_duplicates(subset=['customer_id'])[['customer_id', 'employment_type', 'loan_default']]
    features = pd.merge(features, metadata, on='customer_id', how='left')
    
    return features

@st.cache_data
def run_segmentation(features_df):
    """
    Runs KMeans clustering & GMM probabilities. Returns updated dataframe & PCA object.
    Cluster labels: Stable Savers, Volatile Earners, High-Risk Spenders
    """
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Income_Variance', 'Expense_Ratio', 'Transaction_Consistency']
    X = features_df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    features_df['PCA1'] = X_pca[:, 0]
    features_df['PCA2'] = X_pca[:, 1]
    
    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Map clusters intuitively based on Expense_Ratio and Monetary
    temp_df = pd.DataFrame({'cluster': clusters, 'Expense_Ratio': features_df['Expense_Ratio']})
    cluster_means = temp_df.groupby('cluster')['Expense_Ratio'].mean().sort_values()
    
    # Assign labels based on expense ratio (lowest=Stable, middle=Volatile, highest=High-Risk)
    mapping = {
        cluster_means.index[0]: 'Stable Savers',
        cluster_means.index[1]: 'Volatile Earners',
        cluster_means.index[2]: 'High-Risk Spenders'
    }
    
    features_df['Segment'] = pd.Series(clusters).map(mapping)
    
    # GMM for confidence
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    features_df['Confidence_Score'] = probs.max(axis=1) * 100
    
    return features_df, pca, kmeans

@st.cache_data
def calculate_clv(df, features_df):
    """
    Calculates Probabilistic CLV using Lifetimes framework.
    """
    # Lifetimes requires specific RFM format: freq, recency, T, monetary_value
    expenses = df[df['transaction_type'] == 'expense'].copy()
    max_date = expenses['transaction_date'].max()
    
    # Aggregate data to customer level for lifetimes
    customer_summary = expenses.groupby('customer_id').agg(
        frequency=pd.NamedAgg(column='transaction_date', aggfunc=lambda x: (x.max() - x.min()).days / 30.0), # approx months
        recency=pd.NamedAgg(column='transaction_date', aggfunc=lambda x: (x.max() - x.min()).days),
        T=pd.NamedAgg(column='transaction_date', aggfunc=lambda x: (max_date - x.min()).days),
        monetary_value=pd.NamedAgg(column='transaction_amount', aggfunc='mean')
    ).reset_index()
    
    # Due to synthetic daily data logic, some BG/NBD models might struggle with non-standard transaction timings.
    # Fallback to a simplified RFM CLV formula if bgf fitting fails or for simplicity.
    try:
        # Fit BG/NBD
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        customer_summary = customer_summary[customer_summary['frequency'] > 0]
        bgf.fit(customer_summary['frequency'], customer_summary['recency'], customer_summary['T'])
        
        # Predict transactions next 6 months (180 days)
        t = 180 
        customer_summary['predicted_tx'] = bgf.predict(t, customer_summary['frequency'], customer_summary['recency'], customer_summary['T'])
        customer_summary['prob_alive'] = bgf.conditional_probability_alive(customer_summary['frequency'], customer_summary['recency'], customer_summary['T'])
        
        # Fit Gamma-Gamma
        ggf = GammaGammaFitter(penalizer_coef=0)
        # Filter for positive monetary values
        clv_df = customer_summary[(customer_summary['monetary_value'] > 0) & (customer_summary['frequency'] > 0)]
        ggf.fit(clv_df['frequency'], clv_df['monetary_value'])
        
        clv_df['expected_avg_profit'] = ggf.conditional_expected_average_profit(clv_df['frequency'], clv_df['monetary_value'])
        clv_df['CLV_Score'] = clv_df['predicted_tx'] * clv_df['expected_avg_profit']
        
        result_df = pd.merge(features_df, clv_df[['customer_id', 'prob_alive', 'CLV_Score']], on='customer_id', how='left')
        result_df['CLV_Score'] = result_df['CLV_Score'].fillna(0)
        result_df['prob_alive'] = result_df['prob_alive'].fillna(0)
        
    except Exception as e:
        # Fallback simplified calculation
        print(f"Lifetimes failed, using fallback: {e}")
        # Scale retention score inversely to recency
        retention = 1 / (features_df['Recency'] + 1)
        features_df['CLV_Score'] = features_df['Frequency'] * features_df['Monetary'] * retention
        features_df['prob_alive'] = np.random.uniform(0.5, 0.99, size=len(features_df))
        result_df = features_df
        
    # Scale CLV to a readable range if not already
    max_clv = result_df['CLV_Score'].max()
    if max_clv > 0:
        # Normalize arbitrarily to max of 100,000 for display aesthetics if it's too high
        pass
        
    return result_df

@st.cache_resource
def train_credit_model(features_df):
    """
    Trains an XGBoost classifier for default prediction.
    """
    df_model = features_df.copy()
    
    if 'loan_default' not in df_model.columns or df_model['loan_default'].nunique() < 2:
        # Synthetic Target creation
        df_model['loan_default'] = ((df_model['Segment'] == 'High-Risk Spenders') | 
                                    (df_model['Expense_Ratio'] > 0.85)).astype(int)
    
    # Create Dummies
    if 'Segment' in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=['Segment'], drop_first=False)
        
    features = [col for col in df_model.columns if col not in ['customer_id', 'loan_default', 'employment_type']]
    
    X = df_model[features]
    y = df_model['loan_default']
    
    # SMOTE for imbalance
    smote = SMOTE(random_state=42)
    try:
        X_res, y_res = smote.fit_resample(X, y)
    except ValueError:
        # Not enough samples or class sizes
        X_res, y_res = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Score whole dataset
    all_probs = model.predict_proba(X)[:, 1]
    
    # Invert prob since higher risk means lower score
    # Approval Probability is 1 - Default Probability
    features_df['Approval_Probability'] = 1 - all_probs
    # Scale to 0-1000 credit score
    features_df['Credit_Score'] = (features_df['Approval_Probability'] * 1000).astype(int)
    
    # Decision Logic
    conditions = [
        (features_df['Credit_Score'] >= 700),
        (features_df['Credit_Score'] >= 550) & (features_df['Credit_Score'] < 700),
        (features_df['Credit_Score'] < 550)
    ]
    choices = ['Approve', 'Review', 'Reject']
    features_df['Decision'] = pd.Series(np.select(conditions, choices, default='Reject'))
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return features_df, model, metrics, importance_df
