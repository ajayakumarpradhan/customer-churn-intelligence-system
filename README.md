# 🛒 Customer Churn Intelligence System
### End-to-End Machine Learning | Behavioral Analytics | XGBoost | FastAPI | Streamlit + AI Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?style=flat&logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat)

---

## 🚀 Problem Statement

In the competitive E-commerce landscape, retaining existing customers is 5x more cost-effective than acquiring new ones. Identifying "at-risk" customers before they churn is critical for maintaining revenue stability and long-term growth.

This project builds an **enterprise-grade Customer Behavioral Intelligence System** featuring a real-time risk scoring API, an interactive segmentation dashboard, and an AI-powered diagnostic chatbot—enabling marketing and success teams to intervene with personalized retention strategies.

---

## 🎯 Objective

- Predict binary customer churn with high recall for proactive intervention.
- Analyze purchase behavior patterns (RFM, CLV, Return Rates).
- Build a production-ready API for real-time risk scoring.
- Segment customers into actionable quadrants (Champions vs. At-Risk).
- Enable transparent AI business querying via a Gemini-powered insights chatbot.

---

## 🧠 ML System Pipeline

```
Raw Data → Behavioral Aggregation → RFM & CLV Engineering → Cost-Sensitive Training (XGBoost) → 4-Quadrant Segmentation → API Deployment → Interactive Dashboard
```

---

## 📊 Dataset

| Property         | Detail                                    |
|------------------|-------------------------------------------|
| Records          | ~250,000 Transactions (49,673 Customers)  |
| Imbalance        | ~20% Churned                              |
| Target Variable  | `churn` (Binary Classification)           |
| Domain           | E-commerce / Retail                       |
| Features         | Demographics, RFM, Returns, CLV, Trends   |

---

## 🔍 Key Insights (EDA)

- **Return Behavior:** Customers with high return rates (>30%) show a 2.5x higher probability of churn.
- **Recency Decay:** Probability of churn increases linearly after 60 days of inactivity.
- **CLV Concentration:** Top 10% of customers (Champions) contribute to over 45% of total revenue.

---

## ⚙️ Feature Engineering

Created **advanced behavioral features** to capture the "customer pulse":

```python
# ── RFM & CLV Features ──
# Recency: Days since last purchase
# Frequency: Total transaction count
# Monetary: Total lifetime spend
# CLV Score: (Frequency * Monetary_Avg * Tenure_Months) / 100

# ── Behavioral Health Features ──
return_rate = total_returns / frequency
category_diversity = unique_categories_purchased
purchase_trend = monetary_total / days_as_customer
```

| Feature              | Category    | Description                              |
|----------------------|-------------|------------------------------------------|
| clv_score            | Value       | Estimated customer lifetime value metric  |
| purchase_trend       | Velocity    | Spending speed over the customer tenure  |
| category_diversity   | Engagement  | Number of distinct product categories     |

---

## 🤖 Model Training & Evaluation

### Training Configuration (Weighted XGBoost)
To handle the natural churn imbalance, we utilized XGBoost with `scale_pos_weight` calculated from the negative-to-positive ratio. This ensures the model prioritizes identifying churners (Recall) over simply guessing the majority class.

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    scale_pos_weight=4.0, # Imbalance handling
    learning_rate=0.1,
    random_state=42
)
```

### Evaluation Strategy
- **Primary Metric:** ROC-AUC and F1-Score for the minority (churn) class.
- **Interpretability:** SHAP summary plots used to explain global drivers of churn.

---

## 🌐 Production System

### FastAPI REST API (`api/api.py`)

| Endpoint           | Method | Description                        |
|--------------------|--------|------------------------------------|
| `/predict`         | POST   | Real-time churn risk scoring       |
| `/customer/{id}`  | GET    | Full customer profile & metrics    |
| `/chat`            | POST   | AI-powered business insights       |

```bash
# Example API call
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "recency_days": 45,
    "frequency": 12,
    "monetary_total": 1200.0,
    "return_rate": 0.08
  }'

# Response: {"churn_prediction": 1, "churn_probability": 0.72, "risk_level": "High"}
```

### Streamlit Dashboard (`app/app.py`)

| Feature                  | Description                                         |
|--------------------------|-----------------------------------------------------|
| 📊 KPI Overview         | Total Revenue at Risk, Global Churn, Avg CLV        |
| 🎯 Segmentation Matrix  | 4-Quadrant visual (CLV vs. Churn Probability)       |
| 🔍 Customer Explorer    | Deep-dive into individual customer SHAP drivers     |
| 🤖 AI Insights Chat     | Natural language query engine for data analytics    |

---

## 🤖 AI Insights — Churn AI Diagnostics

The built-in LLM engine (Google Gemini) understands the underlying customer data and responds with grounded business insights:

*Example Response:*
> **AI Insight:** Our current churn rate is **20.1%**. The **At-Risk High Value** segment is our biggest concern, representing $42,000 in potential lost revenue.
> *Recommendation: Prioritize personalized retention emails for the 7,362 customers in this segment, specifically those whose purchase_trend has dropped by more than 50% in the last 30 days.*

---

## 💼 Business Impact

- **Predictive Retention:** Identify at-risk users *before* they stop spending.
- **Budget Optimization:** Focus high-touch retention (discounts/gifts) strictly on High-CLV customers.
- **Explainable AI:** SHAP integration ensures agents understand *why* a customer is flagged.
- **Unified Interface:** One platform for data scientists (metrics) and marketing (segments).

---

## 📁 Project Structure

```
customer-churn-intelligence-system/
│
├── api/
│   └── api.py                  # FastAPI Backend + Chatbot
├── app/
│   └── app.py                  # Streamlit Dashboard (Bright Theme)
├── src/
│   ├── preprocessing.py        # Data cleaning
│   ├── feature_engineering.py  # RFM/CLV Logic
│   ├── train.py                # XGBoost Training
│   ├── segmentation.py         # 4-Quadrant Logic
│   └── utils.py                # Config & Logging
├── model_artifacts/
│   ├── churn_model.pkl         # Trained XGBoost weight
│   ├── scaler.pkl              # Feature scaling artifacts
│   └── feature_list.pkl        # Column metadata
├── data/
│   ├── raw/                    # Raw transaction CSV
│   └── processed/              # Aggregated customer features
├── deployment/
│   ├── Dockerfile              # Containerization
│   └── DEPLOYMENT.md           # Cloud setup guide
├── README.md                   # Project Documentation
└── requirements.txt            # Dependency list
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ajayakumarpradhan/customer-churn-intelligence-system.git
cd customer-churn-intelligence-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Pipeline (Preprocess -> Train -> Segment)
python -m src.preprocessing
python -m src.train
python -m src.segmentation

# 4. Start FastAPI (port 8000)
python -m uvicorn api.api:app --port 8000

# 5. Start Streamlit Dashboard (port 8501)
python -m streamlit run app/app.py
```

---

## 🛠️ Tech Stack

| Category         | Tools                                  |
|------------------|----------------------------------------|
| Language         | Python 3.10+                           |
| ML Framework     | XGBoost, Scikit-learn, SHAP            |
| Data Processing  | Pandas, NumPy                          |
| AI Insights      | Google Gemini API (FastAPI Integration)|
| API Framework    | FastAPI + Uvicorn                      |
| Dashboard        | Streamlit (Light Premium Theme)        |
| Task Type        | Customer Behavioral Intelligence       |

---

## 👤 Author

**Ajaya Kumar Pradhan**
*Data Analyst | Machine Learning Enthusiast*
