import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Visualization")

st.title("Visualization in Streamlit")

st.markdown("""
Streamlit supports interactive visualizations using Plotly.  
You can create line charts, bar charts, scatter plots, and boxplots with full interactivity.
""")

st.header("Sample dataset")

df = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "Sales": [100, 120, 90, 140, 160],
    "Expenses": [80, 100, 70, 130, 150]
})

st.dataframe(df)

st.header("Choose a plot type for sample data")

plot_type = st.radio("Select a plot", ["Line chart", "Bar chart", "Scatter plot", "Boxplot"], key="sample_plot")

if plot_type == "Line chart":
    fig = px.line(df, x="Month", y=["Sales", "Expenses"], title="Sales and expenses over months")
    st.plotly_chart(fig)

elif plot_type == "Bar chart":
    fig = px.bar(df, x="Month", y=["Sales", "Expenses"], barmode="group", title="Sales and expenses by month")
    st.plotly_chart(fig)

elif plot_type == "Scatter plot":
    fig = px.scatter(df, x="Sales", y="Expenses", text="Month", title="Sales vs expenses")
    st.plotly_chart(fig)

elif plot_type == "Boxplot":
    fig = px.box(df, y=["Sales", "Expenses"], title="Distribution of sales and expenses")
    st.plotly_chart(fig)

st.markdown("---")
st.header("Upload your own dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.dataframe(user_df)

    numeric_columns = user_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    all_columns = user_df.columns.tolist()

    if len(numeric_columns) < 1:
        st.warning("No numeric columns found for plotting.")
    else:
        st.markdown("### Choose variables for your plot")
        x_axis = st.selectbox("X-axis", options=all_columns)
        y_axis = st.selectbox("Y-axis", options=numeric_columns)

        user_plot = st.radio("Select a plot", ["Line chart", "Bar chart", "Scatter plot", "Boxplot"], key="user_plot")

        if user_plot == "Line chart":
            fig = px.line(user_df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            st.plotly_chart(fig)

        elif user_plot == "Bar chart":
            fig = px.bar(user_df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            st.plotly_chart(fig)

        elif user_plot == "Scatter plot":
            fig = px.scatter(user_df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig)

        elif user_plot == "Boxplot":
            fig = px.box(user_df, x=x_axis, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}")
            st.plotly_chart(fig)
