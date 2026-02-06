# Loan Eligibility Prediction (Classification) 🏦✅

## Overview
This project builds a **machine learning model** that predicts whether a customer's **home loan will be approved or not** based on their application details (income, credit history, loan amount, property area, etc.).  

The goal is to help a finance company (Easy House) **automate the loan approval decision** using past customer data, so the process becomes faster, more consistent, and less risky.


## Why this project is needed (Business Problem)
In real lending, approving the wrong customer can lead to **defaults and financial loss**, and rejecting good customers can lead to **lost revenue and poor customer experience**.

Today, many companies still do a lot of manual checks. This causes:
- Slow processing time
- Human bias / inconsistent decisions
- Higher operational cost
- Risk of approving high-risk customers

So the company wants a model that can:
✅ reduce risk  
✅ speed up approvals  
✅ Focus effort on the most eligible customers  
✅ support data-driven lending policy  


## Objective
Build a classification model that predicts:

**Loan_Status**
- `1` = Approved  
- `0` = Not Approved  

Using applicant + loan features from past data.

## Dataset (What data is used?)
This project uses a credit risk/loan dataset with features like:

- `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
- `ApplicantIncome`, `CoapplicantIncome`
- `LoanAmount`, `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- Target: `Loan_Status`


## What this project does (Workflow)
### 1) Data Cleaning & Preprocessing
- Fixed datatypes (example: `Dependents` converted from "3+" to numeric)
- Handled missing values:
  - Categorical columns → filled using **mode**
  - Numerical columns → filled using **median**
- Dropped `Loan_ID` because it's only a unique identifier

### 2) Exploratory Data Analysis (EDA)
- Checked distributions (histograms + bar charts)
- Compared features vs Loan_Status
- Correlation check to avoid multicollinearity

### 3) Feature Engineering
Created two helpful financial features:
- **Total_Income = ApplicantIncome + CoapplicantIncome**
- **Balance_Income = Total_Income - (LoanAmount*1000 / Loan_Amount_Term)**

### 4) Encoding + Scaling
- Converted categorical variables into numeric (dummy variables)
- Standardized features using **StandardScaler**

### 5) Handling Class Imbalance (SMOTE)
Loan approvals were more common than rejections (imbalanced target).
So we applied **SMOTE** on training data to balance both classes (50/50).

### 6) Model Building (Compared Multiple Models)
Models trained and compared:
- Logistic Regression (baseline, interpretable)
- Decision Tree (simple non-linear model)
- Random Forest (best performing)

### 7) Model Evaluation
Used:
- Confusion Matrix
- Precision / Recall / F1-score
- ROC-AUC score
- Cross-validation ROC-AUC

## Best Model Result 
**Random Forest performed best overall** and gave the strongest predictive results and stability.

It also helped identify the **most important drivers** behind loan approval decisions.

## Key Insights (What affects loan approval the most?)
From the models and feature importance, the strongest drivers were:

### Top Approval Drivers
1. **Credit_History**
   - Most powerful factor for approval (strongest predictor)
2. **LoanAmount**
   - Larger loan amounts reduce approval chances (higher risk)
3. **Income strength**
   - ApplicantIncome + CoapplicantIncome matter for affordability
4. **Property_Area**
   - Semi-urban applicants had a higher approval probability than rural applicants

### Low-impact features (should not drive decisions)
- Gender, education, and self-employment had minimal predictive value in this dataset  
(so they should not be used heavily in decisions for fairness + compliance)

## Business Recommendations 
Based on the best model outputs:

### 1) Prioritize Credit History Screening
- Make credit history checks strict and reliable
- Flag weak credit cases early
- Offer credit-building plans to improve future eligibility

### 2) Strengthen Income-to-Loan Criteria
- Use Total_Income + LoanAmount to set clear affordability rules
- Encourage joint applications when the co-applicant's income improves approval chances

### 3) Add Extra Review for High Loan Amount Requests
- High loan amount applications should go through stricter review
- Consider alternative loan products or higher collateral for high-risk segments

### 4) Use Property Area in Risk Scoring (carefully)
- Semi-urban zones showed higher approvals
- Review regional risk policy and adjust risk scoring per location if needed

### 5) Keep Process Compliance-Safe
- Since gender and some demographic features had little value,
  Reduce reliance on them to avoid concerns about bias.

## Tools & Libraries Used
- Python (Pandas, NumPy)
- Matplotlib / Seaborn (EDA + visualization)
- Scikit-learn (modeling, scaling, metrics)
- Imbalanced-learn (SMOTE)
