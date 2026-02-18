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

The moderate drop in accuracy (78% → 71%) indicates reasonable generalization without severe overfitting.

**Top odds ratios (for business explainability):**
- Credit History: **41.7× higher approval odds** for applicants with clean history
- Married: 4.0× | Self-Employed: 2.3× | Balance Income: 2.2×

## 🔍 Key Drivers of Loan Approval
From **Random Forest feature importance** (best model):

<img width="1313" height="1089" alt="image" src="https://github.com/user-attachments/assets/38c72f66-9fec-488a-985c-059cdda5cabf" />


|Feature | Importance Score | What It Means |
|---|---|---|
| **Credit_History** | **0.27** | Dominates all other features, nearly 2× the next driver |
| **LoanAmount** | 0.14 | Larger loans = higher default risk = lower approval odds |
| **ApplicantIncome** | 0.13 | Primary income signal for affordability |
| **CoapplicantIncome** | 0.10 | Joint income meaningfully improves approval probability |
| **Dependents** | 0.08 | More dependents = higher financial burden |
| **Property_Area_Semiurban** | 0.07 | Semiurban applicants have higher approval likelihood |
| **Loan_Amount_Term** | 0.06 | Longer terms improve affordability and approval odds |
| **Married** | 0.04 | Modest positive signal |
| **Property_Area_Urban** | 0.03 | Urban also positive but less than semiurban |
| **Education, Gender, Self_Employed** | ≤ 0.02 each | Minimal predictive value |

> ⚠️ Gender, education, and self-employment status each contribute **≤2% of predictive importance** in the Random Forest model, yet carry real compliance and fairness risk. They are deliberately de-weighted in any production scoring decision.

## 💡 Business Recommendations

**1. Make credit history a hard gate (Importance: 0.27)**
Credit history alone accounts for 27% of predictive power, nearly double the next feature. It should be the non-negotiable first screen. For borderline applicants, offer structured credit-building pathways before re-application.

**2. Set loan amount thresholds by income tier (Importance: 0.14)**
LoanAmount is the #2 driver of rejection risk. Pair it with applicant income to define clear affordability bands; high loan amounts relative to income should trigger automatic escalation to manual review or require additional collateral.

**3. Prioritize total household income, not just applicant income (Importance: 0.13 + 0.10)**
ApplicantIncome and CoapplicantIncome together account for 23% of model importance. Actively promote joint applications during onboarding; co-applicant income meaningfully tips the affordability calculation for borderline cases.

**4. Factor in number of dependents during risk scoring (Importance: 0.08)**
Dependents ranked 5th in the model; higher financial obligations reduce repayment capacity in ways that income alone doesn't capture. Include a dependents-adjusted income metric in the scoring logic.

**5. Use loan term as a lever for borderline approvals (Importance: 0.06)**
Longer loan terms lower the monthly EMI and improve Balance Income; the model reflects this. For applicants who are borderline on affordability, offering an extended term can move them into the approved tier while managing risk.

**6. Design for fairness from the start (Importance: ≤0.02 each)**
Gender, education, and self-employment each contribute less than 2% of predictive importance but carry real compliance and legal risk. Exclude them from production scoring; the model doesn't need them, and using them creates liability.

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
| `Predicting_Loan_Approval.pptx` | Presentation deck with full business recommendations |


