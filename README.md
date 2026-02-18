# 🏦 Loan Approval Prediction - Reducing Credit Risk with Machine Learning

> **Can we predict which applicants are likely to default before approving their loan?**
> This project builds a machine learning pipeline that answers that question, and translates the answer into a lending strategy a business can actually act on.


## 💼 The Business Problem

Every loan approval is a calculated risk. Approving a high-risk applicant leads to defaults and financial loss. Rejecting a creditworthy applicant means lost revenue and a poor customer experience.

For **Easy House Finance**, manual underwriting processes are slow, inconsistent, and difficult to scale. An automated, data-driven approval model can:

- Reduce processing time and operational cost
- Minimize human bias in lending decisions
- Surface the riskiest applicants for extra review before approval
- Improve portfolio quality without sacrificing customer experience

## 📊 Dataset

- **~600 historical loan applications**
- **Target variable:** `Loan_Status` -  Approved (1) or Not Approved (0)
- **Features:** Demographics (gender, marital status, education), financials (applicant income, co-applicant income, loan amount, term), credit history, property area

## ⚙️ What I Built

An end-to-end classification pipeline: raw data → cleaned features → engineered signals → balanced training → model comparison → business recommendations.

### Step 1: Data Cleaning
- ~20% of values were missing; imputed using **mode** (categorical) and **median** (numerical)
- Dropped `Loan_ID` - identifier with no predictive value
- One-hot encoded `Gender`, `Education`, `Property_Area.`
- Binary mapped `Married` and `Self_Employed` to 0/1
- Result: **0% missing values with no row loss**

### Step 2: Exploratory Analysis

Key findings before modeling:
- **~68% approved, ~32% not approved**, moderate class imbalance
- **Credit history** was the single strongest visual predictor of approval
- **Semiurban applicants** had the highest approval rate vs. urban and rural
- Income was right-skewed with notable outliers requiring treatment

### Step 3: Feature Engineering & Outlier Handling

Created two financial stability indicators to better capture true repayment ability:

- `Total_Income` = ApplicantIncome + CoapplicantIncome
- `Balance_Income` = Total_Income − (LoanAmount × 1000 / Loan_Amount_Term)

Applied **Z-score > 3** to remove extreme outliers, eliminated ~3.75% of records, reduced model variance.

### Step 4: Handling Class Imbalance

Applied **SMOTE on training data only** (70/30 split) to balance classes to 50/50, preventing the model from over-predicting "Approve" by default.

### Step 5: Model Comparison

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression | **71%** | Transparent baseline; strong interpretability |
| Decision Tree | **~66%** | Captures non-linear patterns; showed signs of overfitting |
| Random Forest | **76%** | Best overall; most robust on unseen data |

## 📈 Results

### Best Model: Random Forest

| Metric | Score |
|---|---|
| **Test Accuracy** | **76%** |
| Recall - Approved (Class 1) | **0.88** |
| Recall - High-Risk/Rejected (Class 0) | **0.53** |

The model correctly predicts loan outcome for **~3 out of 4 applicants**. It is particularly strong at identifying qualified borrowers (88% recall), with moderate ability to flag high-risk applicants, an area for future improvement via threshold tuning.

### Logistic Regression: Interpretability Model

| Metric | Score |
|---|---|
| Training Accuracy | 78% |
| Test Accuracy | 71% |
| Precision | 0.79 |
| Recall | 0.78 |
| F1-Score | 0.78 |

The moderate accuracy drop (78% → 71%) indicates reasonable generalization with no severe overfitting.

## 🔍 Key Drivers of Loan Approval

From logistic regression odds ratios and random forest feature importance:

| Driver | Impact |
|---|---|
| **Credit History** | **41.7× more likely to be approved**, by far the strongest single predictor |
| **Married** | 4.0× higher odds of approval |
| **Self-Employed** | 2.3× higher odds of approval |
| **Balance Income** | 2.2× higher disposable income after EMI significantly boosts approval |
| **Loan Amount** | Higher amounts reduce approval probability |
| **Property Area** | Semiurban > Urban > Rural in approval likelihood |

> ⚠️ Gender, education, and other demographic features showed **minimal predictive value**, relying on them risks compliance violations and fairness issues. They are deliberately excluded from the final scoring model.

## 💡 Business Recommendations

**1. Make credit history the first filter**
With a 41.7× impact on approval odds, credit history should be the non-negotiable first gate. Offer credit-building pathways to near-eligible applicants.

**2. Use Balance Income, not just gross income, for affordability scoring**
Raw income is right-skewed and misleading. Disposable income after estimated EMI is a far stronger repayment signal.

**3. Promote joint applications during onboarding**
Co-applicant income meaningfully improves approval odds. Encourage joint applications where combined income tips the affordability calculation.

**4. Add a manual review tier for high loan amounts**
High-value requests carry disproportionate default risk. Route them to stricter underwriting or require additional collateral.

**5. Design for fairness from the start**
Excluding low-signal demographic features before deployment reduces legal exposure and builds customer trust.

## ⚠️ Limitations & Next Steps

- Dataset size (~600 rows) limits generalizability, model should be validated on larger, real-world loan portfolios
- Historical approvals may embed past bias, requiring fairness auditing before deployment
- Missing financial depth: debt-to-income ratio, savings history, employment tenure, collateral value
- Future: explore **XGBoost / Gradient Boosting** for improved high-risk detection; implement dynamic cutoff tuning to optimize the precision-recall tradeoff per business tolerance

## 🛠 Tools & Libraries

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `imbalanced-learn (SMOTE)`

## 📁 Files in This Repo

| File | Description |
|---|---|
| `Predicting_Loan_Approval.ipynb` | Full end-to-end notebook: cleaning → EDA → modeling → evaluation |
| `CreditRisk_Data_File.csv` | Raw dataset |
| `Predicting_Loan_Approval_PPT.pptx` | Presentation deck with full business recommendations |


