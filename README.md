# Customer Churn Prediction

## Executive Summary

This project develops predictive models to identify telecom customers at risk of churn in order to optimize retention outreach strategy.

Two models were evaluated:

- **Logistic Regression** (interpretable baseline)
- **Gradient Boosting (XGBoost)** to capture nonlinear effects and feature interactions

### Key Results

- Logistic Regression ROC-AUC: **0.842**
- XGBoost ROC-AUC: **0.848**
- Average Precision (AP) improved from **0.634 → 0.667** (+0.03)
- Higher precision under a fixed 20% outreach budget
- Improved expected retention value under a simple cost model

XGBoost was selected for deployment due to stronger high-confidence targeting performance and improved business impact.

---

## Business Framing

Customer churn represents lost recurring revenue. Retention campaigns incur outreach costs (discounts, incentives, marketing spend), so identifying high-risk customers efficiently is critical.

Key considerations:

- Contacting all customers is inefficient.
- Threshold selection must balance recall (capture churners) and precision (avoid unnecessary outreach).
- Model evaluation should be based on financial impact, not just ROC-AUC.

Success is defined as increasing expected net retention value under a fixed outreach budget.

---

## Data Overview

- Dataset: Telco Customer Churn
- Observations: ~7,000 customers
- Target: `Churn` (Yes/No)

Features include demographics, contract type, tenure, service usage, billing, and payment behavior.

Data was inspected for missing values, type consistency, and class imbalance prior to modeling.

---

## Modeling Approach

### 1. Baseline Models
- Heuristic baseline (e.g., targeting month-to-month contracts)
- Logistic regression as interpretable performance floor

### 2. Advanced Model
- XGBoost (gradient boosting)
  - Captures nonlinear relationships
  - Handles feature interactions
  - Modest hyperparameter tuning via cross-validation

### 3. Evaluation Framework

Models were compared using:

- ROC-AUC (ranking ability)
- Precision–Recall and Average Precision (imbalanced classification)
- Performance under fixed 20% contact budget
- Financial simulation under explicit cost assumptions

---

## Results

| Model | ROC-AUC | AP | Top-20% Precision | Notes |
|--------|---------|------|-------------------|-------|
| Logistic Regression | 0.842 | 0.634 | Higher than baseline heuristic | Interpretable baseline |
| XGBoost | 0.848 | 0.667 | Higher than logistic | Selected model |

Under a fixed 20% outreach strategy, XGBoost consistently captured more true churners while reducing wasted contacts.

A simple financial simulation demonstrated higher expected retention value using XGBoost predictions.

---

## Interpretation

Model interpretation was performed using:

- Logistic regression coefficients
- SHAP values for XGBoost

Key churn drivers:

- Month-to-month contracts
- Short tenure
- Higher monthly charges
- Specific service combinations

SHAP analysis confirmed intuitive drivers and revealed modest nonlinear effects not captured by the linear model.

---

## Limitations and Future Work

**Limitations:**

- Cross-sectional dataset (no temporal validation)
- Predicts churn risk, not treatment uplift
- Assumed outreach cost and retention value

**Future Improvements:**

- Uplift modeling to estimate treatment effect
- Segment-specific churn models
- Integration with A/B testing framework

---

## Files

- `telco_customer_churn_analysis.ipynb` — full modeling workflow  
- `README.md` — project overview  
- `telco.csv` — dataset  

---

## Project Objective

This project demonstrates a repeatable tabular machine learning framework for business decision modeling, including:

- Structured baseline comparison
- Ranking-based evaluation
- Operating point selection
- Financial impact simulation
- Model interpretability via SHAP

The approach generalizes beyond churn prediction to other tabular classification problems with cost-sensitive decision policies.