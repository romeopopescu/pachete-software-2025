import streamlit as st
import pandas as pd

st.set_page_config(page_title="Displaying data")

st.title("Displaying data in Streamlit")

st.markdown("""
Streamlit provides several ways to display data, especially tabular data using pandas.  
You can use `st.write`, `st.dataframe`, and `st.table` to show data in different formats.
""")

st.header("Using st.write for quick display")

st.markdown("This is the simplest way to show a DataFrame or any object.")

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [88, 92, 85]
})

st.code("st.write(df)", language="python")
st.write(df)

st.header("Using st.dataframe for interactive tables")

st.markdown("This allows sorting, scrolling, and dynamic resizing.")

st.code("st.dataframe(df)", language="python")
st.dataframe(df)

st.header("Using st.table for static display")

st.markdown("This renders a static table without interactive features.")

st.code("st.table(df)", language="python")
st.table(df)

st.header("Displaying JSON data")

sample_json = {
    "name": "Alice",
    "skills": ["Python", "SQL", "Streamlit"],
    "active": True
}

st.code("st.json(sample_json)", language="python")
st.json(sample_json)

st.header("Displaying metrics")

st.markdown("You can use `st.metric` to highlight key numbers.")

st.code('st.metric(label="Accuracy", value="92%", delta="+2%")', language="python")
st.metric(label="Accuracy", value="92%", delta="+2%")

st.markdown("---")
st.success("Try using your own dataset below to experiment with different display styles")

st.header("Upload your own CSV")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    
    display_style = st.radio(
        "Choose how to display your data",
        ["Write", "Dataframe"]
    )
    
    st.markdown("### Output")

    if display_style == "Write":
        st.write(user_df)
    elif display_style == "Dataframe":
        st.dataframe(user_df)
