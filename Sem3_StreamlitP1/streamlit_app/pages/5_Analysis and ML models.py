import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Train-test split and machine learning models")

st.markdown("""
### Introduction
Before applying machine learning models, it is essential to understand the dataset, explore its structure, and identify potential issues that could affect model performance. Train-test splitting ensures that the model learns from one part of the data while being evaluated on another, helping to prevent overfitting.

This section covers:
- **Exploring the dataset** to understand feature distributions and class imbalance.
- **Performing a train-test split** and discussing its importance.
- **Training and comparing two machine learning models**: logistic regression and random forest.
""")

df = pd.read_excel("data/credit_card.xlsx")

st.subheader("Raw data sample")
st.write(df.head())

st.markdown("""
## Dataset exploration
Understanding the dataset is critical before applying machine learning models. Examining its structure allows us to detect missing values, check feature distributions, and analyze class balance.
""")

df.rename(columns={"default_payment_next_month": "DEFAULT"}, inplace=True)

st.subheader("Class distribution")
class_counts = df["DEFAULT"].value_counts(normalize=True) * 100

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="Blues", ax=ax)
ax.set_xticklabels(["No Default (0)", "Default (1)"])
ax.set_ylabel("Percentage")
ax.set_title("Class Distribution (Default vs. No Default)")
st.pyplot(fig)

st.markdown("""
The dataset is **imbalanced**, with approximately:
- **77.9% of clients who did not default (0)**
- **22.1% of clients who defaulted (1)**

This imbalance may lead to biased models that favor the majority class. Special techniques, such as class weighting or resampling, might be needed to improve model performance.
""")

st.subheader("Feature types")
st.markdown("""
The dataset consists of **30,000 records and 25 features**. The features can be grouped into three main types:

- **Demographic information**: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`.
- **Financial and credit history**:
  - `LIMIT_BAL` (credit limit).
  - `BILL_AMT1` to `BILL_AMT6` (bill statement amounts for the past six months).
  - `PAY_AMT1` to `PAY_AMT6` (payment amounts in the last six months).
- **Repayment status**: `PAY_0` to `PAY_6` (whether payments were made on time, delayed, or defaulted).

The target variable is **DEFAULT**, indicating whether a customer defaulted on their credit card payment (1) or not (0).
""")

st.subheader("Possible preprocessing steps")
st.markdown("""
Before training a machine learning model, some preprocessing steps might be necessary:

1. **Encoding categorical variables**:
   - `SEX`, `EDUCATION`, and `MARRIAGE` are categorical and should be converted into numerical form.
   - `PAY_0` to `PAY_6` represent repayment status and might need categorical encoding.

2. **Feature scaling**:
   - `LIMIT_BAL`, `BILL_AMT1` to `BILL_AMT6`, and `PAY_AMT1` to `PAY_AMT6` have **large variations** in values.
   - Normalizing these features may improve model performance.

3. **Handling class imbalance**:
   - The dataset is imbalanced, which can lead to models favoring the majority class.
   - Possible solutions include **oversampling the minority class**, **undersampling the majority class**, or **using class weights** in model training.
""")

st.markdown("""
## Train-test split
Splitting the dataset ensures that the model is trained on one part of the data and evaluated on unseen data. This prevents overfitting and provides a better estimate of how the model will perform in real-world scenarios.
""")

st.subheader("Performing the train-test split")
test_size = st.slider("Select test set size", 0.1, 0.5, 0.2)

X = df.drop(columns=["ID", "DEFAULT"])
y = df["DEFAULT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

st.write(f"Training set size: {X_train.shape[0]} samples")
st.write(f"Testing set size: {X_test.shape[0]} samples")

st.markdown("""
A **stratified split** is used to ensure that the class distribution remains the same in both the training and testing sets.
""")

st.markdown("""
## Training machine learning models
Two classification models will be trained and evaluated:

1. **Logistic regression**:
   - A linear model that estimates the probability of default based on input features.
   - It predicts the probability of default using the sigmoid function:
""")
st.latex(r" P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n)}} ")
st.markdown("""
   - Simple, interpretable, and computationally efficient.
   - Works best when features have a linear relationship with the target variable.

2. **Random forest**:
   - An ensemble model that builds multiple decision trees and averages their predictions.
   - Captures **non-linear relationships** and **feature interactions**.
   - Less interpretable but usually performs well on structured data.
""")

st.subheader("Train a machine learning model")
model_choice = st.selectbox("Choose a model", ["Logistic regression", "Random forest"])

if model_choice == "Logistic regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model accuracy: {accuracy:.2f}")

st.markdown("""
## Conclusion
Exploring the dataset before training helps identify key patterns, such as class imbalance and feature distributions. 

Train-test splitting ensures the model generalizes well by evaluating it on unseen data. A **stratified split** is particularly important for imbalanced datasets to maintain a representative distribution of the target variable.

Logistic regression is useful for interpretable models, while random forests capture complex relationships but may overfit small datasets. Evaluating models on a separate test set provides a realistic measure of performance before deployment.
""")
