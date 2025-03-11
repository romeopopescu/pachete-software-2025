import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

st.title("Feature engineering and variable selection")

# introduction
st.markdown("""
### Introduction
Feature engineering is the process of transforming raw data into meaningful features that improve the performance of machine learning models. Well-engineered features help models recognize patterns more effectively, while poorly selected features can introduce noise and reduce accuracy.

This section covers:
- **Derived features**: Creating new variables from existing ones.
- **Aggregation features**: Computing statistical measures based on categorical groupings.
- **Temporal feature extraction**: Extracting information from date/time variables.
- **Feature selection techniques**: Identifying the most important features for model performance.
""")

# load dataset
df = pd.read_csv("data/housing.csv")

st.subheader("Raw data sample")
st.write(df.head())

# feature engineering theory
st.markdown("## Feature engineering")

st.markdown("### 1. Derived features")
st.markdown("Derived features are new variables created using existing ones to provide deeper insights.")

st.latex(r" \text{Price per Room} = \frac{\text{Median House Value}}{\text{Total Rooms}} ")
st.latex(r" \text{House Age} = \text{Housing Median Age} ")
st.latex(r" \text{Rooms per Household} = \frac{\text{Total Rooms}}{\text{Households}} ")
st.latex(r" \text{Bedrooms per Household} = \frac{\text{Total Bedrooms}}{\text{Households}} ")

# derived features
st.subheader("Derived features")

# price per room
df["price_per_room"] = df["median_house_value"] / df["total_rooms"]
st.write("Price per room example:")
st.write(df[["median_house_value", "total_rooms", "price_per_room"]].head())

# total rooms per household
df["rooms_per_household"] = df["total_rooms"] / df["households"]
st.write("Total rooms per household example:")
st.write(df[["total_rooms", "households", "rooms_per_household"]].head())

# total bedrooms per household
df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
st.write("Total bedrooms per household example:")
st.write(df[["total_bedrooms", "households", "bedrooms_per_household"]].head())

# artificially create a low-variance column
df["constant_feature"] = 1  # This feature has zero variance
df["almost_constant_feature"] = np.random.normal(0, 0.0001, df.shape[0])  # Almost zero variance

# aggregation features
st.markdown("### 2. Aggregation features")
st.markdown("Aggregation features summarize data based on categorical variables, which helps provide useful insights.")

st.latex(r" \text{Average House Value} = \frac{\sum \text{Median House Value}}{\text{Count of Houses in Category}} ")
st.latex(r" \text{Average Income} = \frac{\sum \text{Median Income}}{\text{Count of Houses in Location}} ")

st.subheader("Aggregation features")

# average house value per ocean proximity
avg_price_per_location = df.groupby("ocean_proximity")["median_house_value"].mean().reset_index()
st.write("Average house value per ocean proximity:")
st.write(avg_price_per_location)

# average income per location
avg_income_per_location = df.groupby("ocean_proximity")["median_income"].mean().reset_index()
st.write("Average median income per ocean proximity:")
st.write(avg_income_per_location)

# feature selection theory
st.markdown("## Feature selection")
st.markdown("Feature selection helps identify and retain the most relevant features while removing redundant or unimportant ones.")

st.markdown("### 1. Variance threshold selection")
st.markdown("This method removes features with very low variance. If a feature has almost the same value for all observations, it provides little to no information to the model.")

st.latex(r" \text{Variance}(X) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2 ")

st.markdown("### 2. Correlation-based selection")
st.markdown("If two features are highly correlated, they provide redundant information. We use a correlation matrix to remove one of the correlated features.")

st.latex(r" r_{X,Y} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2} \cdot \sqrt{\sum (Y_i - \bar{Y})^2}} ")

st.markdown("If \( r \) is close to **1** or **-1**, the features are highly correlated.")

# feature selection example
st.subheader("Feature selection using variance threshold and correlation")

# define features and target variable
X = df.drop(columns=["median_house_value", "ocean_proximity"])  
X = X.select_dtypes(include=np.number).dropna(axis=1) 

# variance threshold method
st.markdown("### Variance threshold selection")
threshold = st.slider("Select variance threshold", 0.0, 0.0005, 0.0001, 0.00001)

var_thresh = VarianceThreshold(threshold=threshold)
X_var_selected = X.loc[:, var_thresh.fit(X).get_support()]

st.write(f"Features selected using variance threshold ({threshold}):")
st.write(X_var_selected.head())

# correlation-based selection
st.markdown("### Correlation-based selection")

correlation_threshold = st.slider("Select correlation threshold", 0.0, 1.0, 0.8, 0.01)
corr_matrix = X.corr()

# Identify highly correlated features
correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            correlated_features.add(corr_matrix.columns[i])

X_corr_selected = X.drop(columns=correlated_features)

st.write(f"Features selected after removing correlations above {correlation_threshold}:")
st.write(X_corr_selected.head())

st.markdown("""
### Key takeaways
1. **Feature engineering** improves model performance by creating new meaningful variables.
2. **Aggregation features** summarize data and uncover trends.
3. **Variance threshold selection** removes features that contribute little information.
4. **Correlation-based selection** removes redundant features, ensuring efficiency.
""")
