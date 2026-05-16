# CreditLens AI 🏦

**AI-Powered Behavioral Credit Scoring for India's Informal Economy**

## Project Overview
CreditLens AI is a behavior-driven fintech platform designed to evaluate customers based on their transaction behavior rather than traditional metrics like salary slips or employment type. This enables fair credit decisions for gig workers, self-employed individuals, and informal earners.
By combining Behavioral Segmentation, Probabilistic Customer Lifetime Value (CLV) modeling, and an XGBoost-powered Credit Intelligence Engine, the system dynamically assesses spending consistency, income volatility, and projected lifetime profitability to assign an approval decision.

## Quick Start / Installation

1. **Clone the repository** (if applicable) or copy the files.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage Instructions

Upon launching the application, you'll be presented with a sidebar navigation system:
1. **Home Dashboard**: View high-level KPIs and the architectural flow of the system.
2. **Data Upload**: Generate 1000 synthetic customer records with 12 months of realistic transactions, or upload your own CSV file matching the schema.
3. **Behavioral Segmentation**: Explore PCA clusters (Stable Savers, Volatile Earners, High-Risk Spenders) mapping expense ratios and financial habits.
4. **Probabilistic CLV**: Review lifetime value metrics built via the Lifetimes API (BG/NBD & Gamma-Gamma).
5. **Credit Intelligence Engine**: View XGBoost classification metrics, feature importance, and confusion matrices predicting default status.
6. **User Profile**: Deep dive into individual customer metrics, trends, and receive a concise AI explanation for their credit decision.

## Features

- **End-to-end Machine Learning Pipeline**: Includes data preprocessing, clustering, feature engineering, and predictive modeling inside Streamlit.
- **Synthetic Data Generator**: Capable of creating realistic historical datasets matching various employment profiles (gig, self-employed, salaried) out of the box. 
- **Modern Fintech UI**: Responsive CSS styling, interactive Plotly visualizations (radars, sankey diagrams, 2D PCA plots).

## Resume One-Liner

**Built CreditLens AI** — a behavior-driven fintech platform combining RFM segmentation, probabilistic CLV modeling, and XGBoost credit scoring to enable fair credit decisions for gig workers and informal earners.