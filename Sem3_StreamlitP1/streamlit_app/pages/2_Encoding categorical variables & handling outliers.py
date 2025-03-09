import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

st.title("Encoding categorical variables & handling outliers")

df = pd.read_csv("data/titanic.csv")

st.markdown("""
### Understanding categorical variables
Categorical variables contain non-numeric values, such as 'Male' / 'Female' or 'Red' / 'Blue'. 
Since machine learning models work with numbers, we must convert categorical variables into numerical format.

#### Encoding techniques:
- **One-Hot encoding**: creates a binary column (0s and 1s) for each category  
- **Label encoding**: assigns a unique number to each category (used only for ordered categories)

#### **Why are 'Name' and 'Ticket' excluded from One-Hot Encoding?**
1. **'Name'** is a high-cardinality feature, meaning every individual has a unique value. Encoding it would create an excessive number of new columns, making the dataset too large and inefficient.
2. **'Ticket'** numbers are mostly unique and do not hold meaningful categorical information for machine learning models. Encoding them would introduce unnecessary noise rather than useful patterns.

For this reason, we **exclude 'Name' and 'Ticket' from encoding**.
""")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in ["Name", "Ticket"]:
    if col in categorical_cols:
        categorical_cols.remove(col) 

st.write("#### Categorical columns in dataset (excluding 'Name' and 'Ticket'):", categorical_cols)

encoding_method = st.radio("Choose an encoding method", ["One-Hot Encoding", "Label Encoding"])

df_encoded = df.copy()

if encoding_method == "One-Hot Encoding":
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
else:
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = label_encoder.fit_transform(df[col])

st.write("#### 🔢 Encoded data sample")
st.write(df_encoded.head())

st.markdown("""
### Understanding outliers
Outliers are extreme values that differ significantly from other observations. They can result from errors or reveal unusual patterns.

#### Methods for detecting outliers:
1. **Interquartile Range (IQR) method**  
   - Any value below **Q1 - 1.5 × IQR** or above **Q3 + 1.5 × IQR** is considered an outlier.

2. **Z-score method**  
   - Any value with a **Z-score > 3** (or **Z-score < -3**) is considered an outlier.
""")

st.subheader("Detecting outliers")
numeric_cols = df.select_dtypes(include=np.number).columns 
selected_column = st.selectbox("Choose a numeric column to analyze", numeric_cols)


fig_box = px.box(df, y=selected_column, title=f"Boxplot of {selected_column}")
fig_box.update_layout(width=600, height=400)
st.plotly_chart(fig_box, use_container_width=True)

outlier_method = st.radio("Choose a method to detect outliers", ["IQR Method", "Z-score Method"])

df_cleaned = df.copy()

if outlier_method == "IQR Method":
    Q1 = df[selected_column].quantile(0.25)
    Q3 = df[selected_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_cleaned = df[(df[selected_column] >= lower_bound) & (df[selected_column] <= upper_bound)]
elif outlier_method == "Z-score Method":
    df_cleaned = df[np.abs(zscore(df[selected_column])) < 3]

st.subheader("Distribution before and after outlier removal")

fig_hist_before = px.histogram(df, x=selected_column, title="Before outlier removal", nbins=30)
fig_hist_after = px.histogram(df_cleaned, x=selected_column, title="After outlier removal", nbins=30)

fig_hist_before.update_layout(width=500, height=400)
fig_hist_after.update_layout(width=500, height=400)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_hist_before, use_container_width=True)
with col2:
    st.plotly_chart(fig_hist_after, use_container_width=True)

st.write("#### ✅ Data after outlier temoval")
st.write(df_cleaned.head())

st.markdown("""
### Conclusions
1. Categorical variables must be converted into numbers using **One-Hot Encoding** (for unordered categories) or **Label Encoding** (for ordered categories).
2. **'Name' and 'Ticket' were excluded from encoding** due to their high cardinality and lack of predictive power.
3. **Outliers** can distort analysis and need to be **detected and handled properly**.
4. **IQR Method** and **Z-score Method** are commonly used to detect outliers.
""")
