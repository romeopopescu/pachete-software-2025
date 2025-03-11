import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.title("Data visualization and correlation analysis")

st.markdown("""
### Introduction
Data visualization helps us explore datasets, identify patterns, and detect potential issues before applying machine learning models. Understanding how variables relate to each other and how data is distributed provides valuable insights that can improve feature selection, model performance, and data preprocessing.

This section focuses on two key areas:
- **Correlation analysis:** understanding relationships between numerical features.
- **Distribution analysis:** examining the spread of numerical data, detecting skewness, and identifying outliers.
""")

df = pd.read_csv("data/housing.csv")

st.subheader("Raw data sample")
st.write(df.head())

numeric_df = df.select_dtypes(include=["number"])

st.markdown("""
## Correlation analysis
Correlation measures how two numerical variables relate to each other. Detecting strong correlations is useful in feature selection, as highly correlated variables may be redundant. It also helps uncover meaningful relationships that could impact predictions. 

There are two commonly used correlation measures:
""")

st.markdown("""
- **Pearson correlation** measures the **linear** relationship between two numerical variables. It assumes that the data is normally distributed and calculates how strongly two variables move together. The Pearson correlation coefficient (\(r\)) is computed as:
""")
st.latex(r" r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2} \cdot \sqrt{\sum (Y_i - \bar{Y})^2}} ")
st.markdown("""
  - A value close to **1** indicates a strong **positive correlation** (e.g., house price and number of rooms).
  - A value close to **-1** indicates a strong **negative correlation** (e.g., house age and price).
  - A value near **0** suggests little or no linear relationship.

- **Spearman correlation** is a **rank-based** measure that assesses monotonic relationships. Unlike Pearson, it does not assume normally distributed data and is useful for non-linear relationships. The Spearman correlation formula is:
""")
st.latex(r" r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} ")
st.markdown("""
where \( d_i \) represents the difference between the ranks of corresponding values, and \( n \) is the number of observations. Spearman correlation is robust when working with skewed or ordinal data.
""")

st.subheader("Correlation heatmap (Pearson)")
correlation_matrix = numeric_df.corr().round(2)

fig_heatmap = px.imshow(
    correlation_matrix,
    labels=dict(color="correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    color_continuous_scale=[(0, "darkblue"), (0.5, "white"), (1, "darkred")],
    text_auto=True
)

fig_heatmap.update_layout(width=800, height=600)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("""
The heatmap provides a visual representation of variable relationships. 

- **Red cells** indicate strong positive correlations, meaning that as one variable increases, the other tends to increase as well.
- **Blue cells** indicate strong negative correlations, meaning that as one variable increases, the other tends to decrease.
- **Values near zero** indicate weak or no correlation.

A strong correlation does not always imply causation. High correlation between two features might be due to an underlying factor, so additional analysis is needed before drawing conclusions.
""")

st.subheader("Compare Pearson and Spearman correlation")
correlation_method = st.radio("Select correlation method", ["Pearson", "Spearman"])
correlation_matrix = numeric_df.corr(method=correlation_method.lower()).round(2)

fig_corr_table = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    colorscale=[(0, "darkblue"), (0.5, "white"), (1, "darkred")],
    showscale=True,
    annotation_text=correlation_matrix.values,
    hoverinfo="z"
)
st.plotly_chart(fig_corr_table, use_container_width=True)

st.markdown("""
## Distribution analysis
Examining the distribution of numerical features is essential for understanding their properties, detecting outliers, and ensuring appropriate transformations before modeling.

A **histogram** shows how frequently different values occur within a dataset. It helps identify patterns such as:
- **Normal (bell-shaped) distributions**, which are suitable for models that assume Gaussian data.
- **Right-skewed distributions**, where most values are concentrated at lower values, often requiring a log transformation.
- **Left-skewed distributions**, where higher values dominate.

A **boxplot** is another useful tool that summarizes data distribution. It highlights:
- **The median** (middle value of the dataset).
- **The interquartile range (IQR)**, representing the middle 50% of values.
- **Outliers**, which appear as points beyond the whiskers.

Detecting skewness or outliers is crucial, as extreme values can distort model training and impact prediction accuracy.
""")

selected_feature = st.selectbox("Choose a feature to analyze", numeric_df.columns)

st.subheader(f"Distribution of {selected_feature}")

fig_hist = px.histogram(numeric_df, x=selected_feature, nbins=30, marginal="box", title=f"Distribution of {selected_feature}")
fig_hist.update_layout(width=800, height=500)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""
A histogram provides an overview of value frequencies. If most values are concentrated in one area with a long tail extending to the right, the data may be **right-skewed** and could benefit from a log transformation to stabilize variance.
""")

st.subheader(f"Boxplot of {selected_feature}")

fig_box = px.box(numeric_df, y=selected_feature, title=f"Boxplot of {selected_feature}")
fig_box.update_layout(width=600, height=400)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("""
A boxplot presents a compact view of the data spread. 

- The **box** represents the interquartile range (middle 50% of the data).
- The **line inside the box** marks the median.
- The **whiskers** extend to 1.5 times the IQR.
- Any **points beyond the whiskers** are considered potential outliers.

Boxplots are particularly useful for detecting extreme values that might need further investigation.
""")

st.markdown("""
## Conclusion
Correlation analysis helps identify relationships between variables and can guide feature selection. A high correlation between two features may indicate redundancy, requiring one of them to be removed. Heatmaps provide an intuitive visualization of these relationships, allowing us to focus on the most relevant variables.

Distribution analysis is equally important for understanding data characteristics. Histograms and boxplots reveal patterns such as skewness, normality, and the presence of outliers, all of which influence data preprocessing decisions. Skewed data may require transformations, and outliers might need special handling to ensure robust model performance.
""")
