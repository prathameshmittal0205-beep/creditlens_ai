# CreditLens AI
> **Behavioral Transaction Intelligence for Alternative Credit Scoring**

CreditLens AI is an end-to-end fintech platform designed to bridge the gap in alternative lending. By leveraging advanced machine learning on raw transaction data, the platform provides behavioral credit scoring, customer lifetime value (CLV) predictions, and fairness-monitored explainability for populations outside the traditional credit bureau system.

![Platform Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🛑 Problem Statement
Millions of gig workers, freelancers, and informal earners lack access to traditional credit systems. Legacy credit bureaus rely heavily on historical debt repayment (credit cards, mortgages), effectively penalizing individuals who operate primarily in cash, debit, or non-traditional banking ecosystems, regardless of their actual financial stability or earning potential.

## 💡 The Solution
**Behavioral Transaction Intelligence.** 
CreditLens AI shifts the paradigm from *historical debt* to *behavioral cash-flow*. By analyzing raw income and expense transactions, the platform derives robust financial signals—such as savings ratios, spending volatility, and income stability—to build a highly predictive, fair, and explainable alternative credit profile.

---

## ✨ Features

- 📊 **Executive Dashboard**: A high-level overview of portfolio health, default trends, and employment demographics.
- 🧩 **Behavioral Segmentation**: Unsupervised clustering groups customers into distinct behavioral profiles (e.g., *Stable Savers, High-Risk Spenders*).
- 💎 **CLV Intelligence**: Forward-looking Customer Lifetime Value (CLV) estimation, predicting future purchase frequency and monetary value to segment users into risk bands and value tiers.
- 🎯 **Credit Scoring**: A robust binary classification engine predicting default probability based strictly on behavioral features.
- 🔍 **SHAP Explainability**: Global and local model explainability, ensuring transparent, interpretable credit decisions.
- ⚖️ **Fairness Monitoring**: Automated bias audits evaluating model parity across protected attributes (e.g., gender, age).

---

## 🧠 Machine Learning Architecture

CreditLens AI utilizes a robust, modular machine learning pipeline:

1. **Unsupervised Learning**: `KMeans` clustering (paired with `PCA` for dimensionality reduction) groups users based on behavioral similarities.
2. **Lifetimes Modeling**: `BG/NBD` (Beta-Geometric/Negative Binomial Distribution) predicts future transaction frequency, while the `Gamma-Gamma` model predicts average transaction value to calculate CLV.
3. **Supervised Learning**: An `XGBoost` classifier, balanced via `SMOTE` (Synthetic Minority Over-sampling Technique), evaluates default risk based on the engineered behavioral features.
4. **Explainability**: `SHAP` (SHapley Additive exPlanations) demystifies XGBoost predictions, outputting the exact weight of every behavioral signal per customer.

---

## 🎨 UI/UX Design

The platform was built with a **fintech-grade aesthetic**:
- Deep, modern dark theme leveraging glassmorphism and subtle elevation effects.
- Highly interactive data visualizations via Plotly.
- Dynamic **AI Executive Insights** providing plain-english contextual summaries of the underlying data trends.

---

## 🛠️ Tech Stack

- **Core**: Python 3.9+
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Machine Learning**: XGBoost, Scikit-learn
- **Advanced Analytics**: Lifetimes (BG/NBD, Gamma-Gamma)
- **Explainability**: SHAP
- **Data Manipulation**: Pandas, NumPy

---

## 🧬 Dataset Generation

CreditLens AI ships with a sophisticated synthetic data engine. Rather than relying on static datasets, the platform dynamically generates a realistic behavioral transaction ledger (1,000+ customers, 12-month histories) encompassing both income and expense events. 

The generator models distinct financial personas—from stable salaried workers to highly volatile gig earners—ensuring realistic ML feature extraction, class imbalance, and model metrics. Users can seamlessly swap out the synthetic data for real CSV uploads via the UI.

---

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/creditlens-ai.git
cd creditlens-ai
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the platform**
```bash
streamlit run app.py
```

---

## 📸 Screenshots

| Executive Overview | Behavioral Segmentation |
| :---: | :---: |
| *[Screenshot Placeholder]* | *[Screenshot Placeholder]* |

| CLV Intelligence | Credit Scoring & SHAP |
| :---: | :---: |
| *[Screenshot Placeholder]* | *[Screenshot Placeholder]* |

---

## 📁 Folder Structure

```text
creditlens_ai/
│
├── app.py                              # Main application entry point
├── requirements.txt                    # Project dependencies
│
├── pages/
│   ├── 1_Executive_Overview.py         # Portfolio health dashboard
│   ├── 2_Behavioral_Segmentation.py    # KMeans clustering insights
│   ├── 3_CLV_Intelligence.py           # Lifetimes / Value tiering
│   ├── 4_Credit_Scoring.py             # XGBoost model & performance
│   └── 6_Individual_Customer_Profiler.py # Deep-dive local SHAP explainer
│
├── models/
│   ├── clv.py                          # BG/NBD & Gamma-Gamma logic
│   ├── credit_scoring.py               # XGBoost & SMOTE pipeline
│   ├── explainability.py               # SHAP & Fairness auditing
│   └── segmentation.py                 # KMeans & PCA logic
│
└── utils/
    ├── data_gen.py                     # Synthetic dataset generation
    ├── preprocessing.py                # Feature engineering pipelines
    └── ui_components.py                # Global CSS & layout helpers
```

---

## 🔮 Future Improvements

- **Deep Learning Pipelines**: Integration of LSTMs for sequential transaction modeling.
- **Open Banking API**: Direct integration with Plaid or Tink for live transactional ingestion.
- **Adversarial Robustness**: Enhanced drift detection to monitor behavioral shifts over time.

---

## 🤝 Contributors

- **[Your Name]** - *Lead Architect & Developer*

Contributions, issues, and feature requests are welcome!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.