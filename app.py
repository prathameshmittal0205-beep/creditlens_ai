import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from synthetic_data import generate_synthetic_data
from utils import extract_rfm_features, run_segmentation, calculate_clv, train_credit_model

st.set_page_config(page_title="CreditLens AI", page_icon="🏦", layout="wide")

# Custom CSS for fintech aesthetics
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1e3a8a; font-weight: 700;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Initialization
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'features_df' not in st.session_state:
    st.session_state['features_df'] = None
if 'seg_pca' not in st.session_state:
    st.session_state['seg_pca'] = None
if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = None
if 'model_importance' not in st.session_state:
    st.session_state['model_importance'] = None

# Sidebar Navigation
st.sidebar.title("🏦 CreditLens AI")
page = st.sidebar.radio("Navigation", [
    "1. Home Dashboard", 
    "2. Data Upload", 
    "3. Behavioral Segmentation", 
    "4. Probabilistic CLV", 
    "5. Credit Intelligence", 
    "6. User Profile"
])

def process_pipeline():
    with st.spinner("Extracting Behavior Features..."):
        features = extract_rfm_features(st.session_state['raw_data'])
    with st.spinner("Running Segmentation..."):
        features, pca_model, km_model = run_segmentation(features)
        st.session_state['seg_pca'] = pca_model
    with st.spinner("Calculating Lifetime Value..."):
        features = calculate_clv(st.session_state['raw_data'], features)
    with st.spinner("Training Credit ML Engine..."):
        features, model, metrics, importance = train_credit_model(features)
        st.session_state['model_metrics'] = metrics
        st.session_state['model_importance'] = importance
    
    st.session_state['features_df'] = features
    st.success("Pipeline Execution Complete!")

if page == "1. Home Dashboard":
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>CreditLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #64748b;'>AI-Powered Behavioral Credit Scoring for India's Informal Economy</h3>", unsafe_allow_html=True)
    
    st.write("---")
    st.markdown("""
    > **CreditLens AI** evaluates financial behavior rather than employment category to create fairer credit decisions for gig workers and informal earners.
    """)
    
    if st.session_state['raw_data'] is not None and st.session_state['features_df'] is not None:
        df = st.session_state['raw_data']
        feat_df = st.session_state['features_df']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{df['customer_id'].nunique():,}")
        col2.metric("Total Transactions", f"{len(df):,}")
        col3.metric("Avg Monthly Spend", f"₹ {feat_df['Monetary'].mean():,.2f}")
        default_rate = df.drop_duplicates('customer_id')['loan_default'].mean() * 100 if 'loan_default' in df.columns else 0
        col4.metric("Default Rate", f"{default_rate:.1f}%")
    else:
        st.info("Please upload or generate data in the 'Data Upload' section to view KPIs.")

    st.write("### System Flow")
    flow_steps = ["Behavioral Segmentation", "CLV Modeling", "Credit Intelligence Eng.", "Fair Decisions"]
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=["Raw Tx Data"] + flow_steps),
        link=dict(source=[0, 1, 2, 3], target=[1, 2, 3, 4], value=[100, 100, 100, 100])
    )])
    st.plotly_chart(fig, use_container_width=True)

elif page == "2. Data Upload":
    st.header("Data Upload & Generator")
    option = st.radio("Choose Source:", ["Generate Synthetic Data", "Upload CSV File"])
    
    if option == "Generate Synthetic Data":
        if st.button("Generate 1000 Customers (12 Months)"):
            with st.spinner("Simulating..."):
                st.session_state['raw_data'] = generate_synthetic_data(num_customers=1000, months=12)
            process_pipeline()
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            req_cols = ['customer_id', 'transaction_date', 'transaction_amount', 'transaction_type', 'employment_type']
            if all(col in df.columns for col in req_cols):
                st.session_state['raw_data'] = df
                if st.button("Process Data Pipeline"):
                    process_pipeline()
            else:
                st.error(f"Missing required columns. Need: {req_cols}")
                
    if st.session_state['raw_data'] is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state['raw_data'].head())
        
elif page == "3. Behavioral Segmentation":
    st.header("Behavioral Segmentation Engine")
    if st.session_state['features_df'] is not None:
        df = st.session_state['features_df']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(df, x='PCA1', y='PCA2', color='Segment', 
                             title="Customer Segments in PCA Space",
                             hover_data=['customer_id', 'Confidence_Score'],
                             color_discrete_map={'Stable Savers': '#22c55e', 'Volatile Earners': '#eab308', 'High-Risk Spenders': '#ef4444'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### Segment Summary")
            summary = df.groupby('Segment').agg(
                Count=('customer_id', 'count'),
                Avg_Spend=('Monetary', 'mean')
            ).reset_index()
            st.dataframe(summary)
            
        st.write("---")
        st.write("### Average RFM Metrics per Segment")
        radar_df = df.groupby('Segment')[['Recency', 'Frequency', 'Monetary', 'Expense_Ratio']].mean().reset_index()
        
        # Normalize for radar chart
        for col in ['Recency', 'Frequency', 'Monetary', 'Expense_Ratio']:
            radar_df[col] = radar_df[col] / radar_df[col].max()
            
        fig2 = go.Figure()
        for idx, row in radar_df.iterrows():
            fig2.add_trace(go.Scatterpolar(
                r=[row['Recency'], row['Frequency'], row['Monetary'], row['Expense_Ratio']],
                theta=['Recency', 'Frequency', 'Monetary', 'Expense_Ratio'],
                fill='toself',
                name=row['Segment']
            ))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Normalized Metrics")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Please process data first.")

elif page == "4. Probabilistic CLV":
    st.header("Probabilistic CLV Engine")
    if st.session_state['features_df'] is not None:
        df = st.session_state['features_df']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='CLV_Score', nbins=50, title="CLV Distribution", color_discrete_sequence=['#3b82f6'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x='prob_alive', nbins=20, title="Probability Alive Distribution", color_discrete_sequence=['#8b5cf6'])
            st.plotly_chart(fig2, use_container_width=True)
            
        st.write("### Top 20 Customers by CLV")
        st.dataframe(df[['customer_id', 'Segment', 'CLV_Score', 'prob_alive']].sort_values('CLV_Score', ascending=False).head(20))
    else:
        st.warning("Please process data first.")

elif page == "5. Credit Intelligence":
    st.header("Credit Intelligence Engine (XGBoost)")
    if st.session_state['model_metrics'] is not None:
        metrics = st.session_state['model_metrics']
        importance = st.session_state['model_importance']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        c2.metric("Precision", f"{metrics['precision']:.2f}")
        c3.metric("Recall", f"{metrics['recall']:.2f}")
        c4.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            cm = metrics['confusion_matrix']
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual"),
                               x=['No Default', 'Default'], y=['No Default', 'Default'],
                               title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col2:
            fig_imp = px.bar(importance.head(10), x='Importance', y='Feature', orientation='h', 
                             title="Top 10 Feature Importance", color_discrete_sequence=['#10b981'])
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

        st.write("### Credit Decision Distribution")
        df = st.session_state['features_df']
        decision_counts = df['Decision'].value_counts().reset_index()
        decision_counts.columns = ['Decision', 'Count']
        fig_pie = px.pie(decision_counts, values='Count', names='Decision', 
                         color='Decision', color_discrete_map={'Approve':'#22c55e', 'Review':'#eab308', 'Reject':'#ef4444'}, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=False)
        
    else:
        st.warning("Please process data first.")

elif page == "6. User Profile":
    st.header("Individual User Profile")
    if st.session_state['features_df'] is not None:
        df = st.session_state['features_df']
        raw_df = st.session_state['raw_data']
        
        cust_id = st.selectbox("Select Customer ID", df['customer_id'].unique())
        cust = df[df['customer_id'] == cust_id].iloc[0]
        tx_data = raw_df[raw_df['customer_id'] == cust_id]
        
        color_map = {'Approve': 'green', 'Review': 'orange', 'Reject': 'red'}
        dec_color = color_map.get(cust['Decision'], 'black')
        
        st.markdown(f"### {cust_id} Profile")
        st.markdown(f"**Decision:** <span style='color:{dec_color}; font-size: 24px; font-weight: bold;'>{cust['Decision']}</span>", unsafe_allow_html=True)
        
        st.write("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Credit Score", f"{cust['Credit_Score']}/1000")
        c2.metric("Approval Prob", f"{cust['Approval_Probability']*100:.1f}%")
        c3.metric("Segment", f"{cust['Segment']}")
        c4.metric("CLV Score", f"{cust['CLV_Score']:.1f}")
        
        st.write("---")
        # Explanation AI
        risk_level = "low" if cust['Decision'] == 'Approve' else ("moderate" if cust['Decision'] == 'Review' else "high")
        st.info(f"💡 **AI Agent Explanation:** This qualitative assessment shows this customer is classified as a **{cust['Segment']}** with a {cust['Confidence_Score']:.1f}% confidence. Their transaction consistency and CLV projection yield a **{risk_level}** predicted default risk, resulting in a **{cust['Decision']}** action.")

        st.write("### Spending Trends")
        tx_monthly = tx_data.groupby([pd.Grouper(key='transaction_date', freq='M'), 'transaction_type'])['transaction_amount'].sum().reset_index()
        fig = px.line(tx_monthly, x='transaction_date', y='transaction_amount', color='transaction_type', 
                      title="Income vs Expense Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Please process data first.")
