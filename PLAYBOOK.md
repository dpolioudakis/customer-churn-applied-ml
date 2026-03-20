# Tabular ML Playbook

A reusable reference for structuring tabular binary classification projects. The customer churn notebook follows this framework.

---

## Workflow

1. **Load and sanity-check data**
   - Inspect missing values, column types, target balance

2. **Define target and features**
   - `y = outcome variable`
   - `X = all remaining columns`

3. **Split into train and validation**
   - Stratified split
   - Fixed random seed for reproducibility

4. **Build preprocessing pipeline** *(fit on train only)*
   - Handle missing values
   - Encode categorical variables
   - Scale numeric features if needed

5. **Train a baseline model**
   - Logistic regression
     - Performance floor
     - Breaks easily — exposes data and preprocessing bugs early
     - Fast to train, low variance
     - Interpretable coefficients (direction and rough magnitude)

6. **Train a stronger model**
   - Gradient boosting (e.g., XGBoost)
     - Typically best-in-class for tabular data
     - Captures nonlinearity and feature interactions
     - Requires modest hyperparameter tuning
     - Interpretable post-hoc via SHAP, but less transparent than logistic regression

7. **Evaluate on validation set**
   - ROC-AUC
   - Precision / recall
   - Select operating threshold

8. **Interpret results**
   - Feature importance / SHAP
   - Error analysis

9. **Write up results**
   - README: problem → approach → results → next steps
   - Optional: final test-set evaluation

---

## Comparing interpretability: logistic coefficients vs. SHAP

After fitting both models, compare their explanations:

- **Logistic coefficients** — direction and rough magnitude of each feature's effect (assumes linearity)
- **XGBoost SHAP values** — nonlinear, interaction-aware feature attributions

Decision framework:
- If the two tell the same story, logistic is likely sufficient and easier to defend
- If SHAP reveals structure that coefficients miss (nonlinearity, interactions), that's concrete justification for the more complex model

---

## EDA Checklist

1. **Load and basic shape** — `df.shape`, confirm row and column counts
2. **Preview data** — `df.head()`, sanity-check column names
3. **Schema inspection** — `df.info()`, data types and null counts
4. **Summary statistics** — `df.describe()` for numeric, `value_counts()` for categorical
5. **Missingness analysis** — `df.isna().sum()`, examine patterns if relevant
6. **Target distribution** — overall outcome rate, class imbalance
7. **Univariate distributions** — histograms for numeric, bar charts for categorical
8. **Target vs feature analysis** — outcome rate by category, boxplots for numeric vs target
9. **Correlation / multivariate scan** — correlation matrix, identify redundant features
10. **Initial takeaways** — strong signals, data quality concerns, modeling implications
