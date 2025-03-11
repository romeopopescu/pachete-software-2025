import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title("Handling missing values, standardization, and normalization")

st.markdown("""
### Introduction
Real-world datasets often contain missing values and inconsistent numerical scales. Handling missing data properly is important to avoid bias, and standardization or normalization ensures that machine learning models perform optimally.

This section covers:
- methods for handling missing values;
- standardization and normalization techniques.
""")

df = pd.read_csv("../data/titanic.csv")

numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

st.subheader("Raw data sample")
st.write(df.head())

st.subheader("Missing data overview before imputation")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({"Missing values": missing_values, "percentage": missing_percentage})
missing_data = missing_data[missing_data["Missing values"] > 0]
st.write(missing_data)

impute_method = st.selectbox("Select an Imputation Method for Numerical Data", ["Mean", "Median", "Most frequent", "KNN imputer"])

if impute_method in ["Mean", "Median", "Most frequent"]:
    strategy = impute_method.lower().replace("most frequent", "most_frequent")
    num_imputer = SimpleImputer(strategy=strategy)
else:
    num_imputer = KNNImputer(n_neighbors=3)

df_imputed = df.copy() 
df_imputed[numeric_cols] = num_imputer.fit_transform(df_imputed[numeric_cols])


cat_imputer = SimpleImputer(strategy="most_frequent")
df_imputed[categorical_cols] = cat_imputer.fit_transform(df_imputed[categorical_cols])

st.subheader("Missing data overview after imputation")
missing_values_after = df_imputed.isnull().sum()
missing_percentage_after = (missing_values_after / len(df_imputed)) * 100
missing_data_after = pd.DataFrame({"Missing values": missing_values_after, "percentage": missing_percentage_after})
missing_data_after = missing_data_after[missing_data_after["Missing values"] > 0]

if missing_data_after.empty:
    st.success("All missing values have been successfully imputed")
else:
    st.warning("Some missing values still remain:")
    st.write(missing_data_after)

st.subheader("Before and after imputation comparison")
comparison_df = pd.concat([df[numeric_cols].head(10), df_imputed[numeric_cols].head(10)], axis=1)
comparison_df.columns = [f"{col} (Original)" for col in numeric_cols] + [f"{col} (Imputed)" for col in numeric_cols]
st.write(comparison_df)


st.subheader("Download the imputed dataset")
csv = df_imputed.to_csv(index=False).encode("utf-8")
st.download_button(label="Download imputed dataset", data=csv, file_name="imputed_data.csv", mime="text/csv")

st.markdown("""
## Standardization and normalization
Many machine learning algorithms require numerical features to be on a similar scale. Standardization and normalization are two common techniques used to achieve this.

- **Standardization (Z-score scaling)**: Transforms data to have a mean of 0 and a standard deviation of 1. It is useful when data follows a normal distribution.
- **Normalization (Min-Max scaling)**: Scales data between 0 and 1, which is useful for algorithms that are sensitive to magnitude differences, such as neural networks.

Choosing the right method depends on the distribution of the dataset and the machine learning model being used.
""")


scaling_method = st.radio("Choose a scaling method", ["standardization (Z-score)", "normalization (Min-Max)"])

# Apply scaling only to numeric columns
scaler = StandardScaler() if scaling_method == "standardization (Z-score)" else MinMaxScaler()
df_scaled = df_imputed.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

st.subheader("Scaled data sample")
st.write(df_scaled.head())

st.markdown("""
### Key insights
1. missing values should be handled carefully to prevent bias in the dataset;
2. standardization ensures data follows a standard normal distribution;
3. normalization rescales values between 0 and 1, which is useful for certain models
""")
