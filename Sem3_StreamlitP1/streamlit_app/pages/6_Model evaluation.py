import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

st.title("Model evaluation")

st.markdown("""
### Introduction
Evaluating a machine learning model is essential to understand its reliability and limitations. Accuracy alone is not always a good indicator, especially when dealing with imbalanced datasets. To properly assess a model, multiple evaluation metrics must be considered.

This section covers:
- **How decision trees and random forests work.**
- **Key classification metrics and their interpretation.**
- **Confusion matrix analysis.**
- **Performance visualization using ROC and precision-recall curves.**
""")

# Load dataset
df = pd.read_excel("data/credit_card.xlsx")
df.rename(columns={"default_payment_next_month": "DEFAULT"}, inplace=True)

st.subheader("Train-test split and model training")

test_size = st.slider("Select test set size", 0.1, 0.5, 0.2)

X = df.drop(columns=["ID", "DEFAULT"])
y = df["DEFAULT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

st.markdown("""
## Decision trees and random forests
Decision trees and random forests are widely used for classification problems.

A **decision tree** is a flowchart-like model that splits data based on feature values. At each split, the algorithm chooses the feature that best separates the classes. The splitting process continues until a stopping criterion is met, such as reaching a minimum number of samples per leaf.

A **random forest** is an ensemble of multiple decision trees. Each tree is trained on a random subset of the data, and the final prediction is made by averaging their outputs (for regression) or using majority voting (for classification). This reduces overfitting and improves generalization.
""")

st.subheader("Choosing a model")
model_choice = st.selectbox("Choose a model", ["Decision tree", "Random forest"])

if model_choice == "Decision tree":
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

st.markdown("""
## Classification metrics
Machine learning models are evaluated using several metrics.

""")

st.latex(r"""
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
""")
st.markdown("""
Accuracy measures the proportion of correct predictions. However, in imbalanced datasets, it can be misleading.
""")

st.latex(r"""
\text{Precision} = \frac{TP}{TP + FP}
""")
st.markdown("""
Precision represents the fraction of positive predictions that were actually correct. A high precision means fewer false positives.
""")

st.latex(r"""
\text{Recall} = \frac{TP}{TP + FN}
""")
st.markdown("""
Recall measures the proportion of actual positives correctly identified. A high recall means fewer false negatives.
""")

st.latex(r"""
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
""")
st.markdown("""
The F1-score balances precision and recall. It is particularly useful for imbalanced datasets.
""")

st.subheader("Confusion matrix")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fig_cm = ff.create_annotated_heatmap(
    z=[[tn, fp], [fn, tp]],
    x=["Predicted No Default", "Predicted Default"],
    y=["Actual No Default", "Actual Default"],
    colorscale="Blues",
    annotation_text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]]
)

st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("""
A confusion matrix provides insight into misclassifications:

- **True negatives (TN):** correctly predicted non-defaults.
- **False positives (FP):** non-defaulters incorrectly classified as defaulters.
- **False negatives (FN):** defaulters incorrectly classified as non-defaulters.
- **True positives (TP):** correctly predicted defaulters.

Understanding this matrix helps improve model performance.
""")

st.subheader("Classification report")

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

st.write(df_report)

st.subheader("ROC curve and AUC score")

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.2f})"))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random Classifier"))
fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("""
The ROC curve compares how well the model distinguishes between classes. The **AUC (Area Under the Curve)** score provides a single measure of performance, where a higher value indicates a better model.
""")

st.subheader("Precision-recall curve")

precision, recall, _ = precision_recall_curve(y_test, y_probs)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall Curve"))
fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")

st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("""
The precision-recall curve illustrates the trade-off between precision and recall, which is useful when false positives and false negatives have different costs.
""")

st.markdown("""
## Conclusion
Decision trees provide a simple, interpretable model that splits data based on feature values. However, they tend to overfit, making them sensitive to noise in the dataset.

Random forests, on the other hand, mitigate overfitting by aggregating multiple decision trees trained on different subsets of data. This leads to better generalization and improved performance.

Model evaluation involves looking beyond accuracy. Metrics such as precision, recall, and F1-score help assess how well a model classifies positive and negative instances. Confusion matrices, ROC curves, and precision-recall curves provide further insights into performance.

A well-evaluated model ensures reliability before real-world deployment.
""")
