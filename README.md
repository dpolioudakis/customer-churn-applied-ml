# Applied ML Playbook: Customer Churn Prediction

This project works through a churn prediction problem for a telecom dataset, structured as a repeatable playbook for applied tabular ML. It covers the full workflow from EDA through business impact quantification, and the approach generalizes to any cost-sensitive binary classification problem.

---

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

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset via Kaggle)
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
- `PLAYBOOK.md` — reusable workflow checklist and EDA reference
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset

---

## Project Objective

This project demonstrates a repeatable tabular machine learning framework for business decision modeling, including:

- Structured baseline comparison
- Ranking-based evaluation
- Operating point selection
- Financial impact simulation
- Model interpretability via SHAP

The approach generalizes beyond churn prediction to other tabular classification problems with cost-sensitive decision policies.

---

## Using This as a Template

The notebook is structured as a reusable framework for tabular binary classification. To adapt it to a new problem:

### 1. Swap in your dataset
- Replace the CSV and update the feature lists in the **Preprocessing** section
- Identify your numeric and categorical columns, the `ColumnTransformer` pipeline handles both automatically
- Check for class imbalance early; it determines which metrics to prioritize

### 2. Define your cost model
The **Business Simulation** section parameterizes three values:
- `contact_cost` — cost of one outreach attempt
- `retention_value` — revenue saved per retained customer
- `retention_rate` — realistic acceptance/conversion rate

Replace these with estimates from your domain to get a problem-specific ROI calculation.

### 3. Set your operating point
The **Threshold Analysis** section evaluates precision/recall tradeoffs across probability cutoffs. If you have a fixed budget (e.g., can only contact N% of customers), use the fixed-budget block to find the optimal threshold.

### 4. Reusable components
| Component | Where in notebook | What to change |
|---|---|---|
| Preprocessing pipeline | Preprocessing section | Column names, imputation strategy |
| Baseline comparison | Logistic Regression section | Swap in domain-relevant heuristic |
| Hyperparameter search | XGBoost Tuning section | Adjust search space for your model |
| SHAP interpretation | SHAP Analysis section | Works as-is for any tree model |
| Cost simulation | Business Simulation section | Replace cost/value assumptions |

### 5. Evaluation choices
This project uses **Average Precision** as the primary metric (not accuracy or ROC-AUC alone) because the target class is imbalanced. If your problem has a different class distribution or decision context, revisit metric selection before tuning.