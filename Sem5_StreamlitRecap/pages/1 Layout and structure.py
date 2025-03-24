import streamlit as st

st.set_page_config(page_title="Layout and structure")

st.title("Layout and structure in streamlit")

st.markdown("""
Understanding layout is key to building clean and interactive streamlit apps.  
You can structure your content using containers, columns, expanders, and sidebars.
""")

st.header("Basic building blocks")

st.markdown("### `st.title`, `st.header`, `st.subheader`, `st.markdown`")
st.code("""
st.title("my streamlit app")
st.header("this is a header")
st.subheader("this is a subheader")
st.markdown("this is markdown")
""", language="python")

st.header("Sidebar layout")

st.markdown("Use `st.sidebar` to keep your ui clean and intuitive.")

with st.sidebar:
    st.markdown("### Sidebar example")
    st.selectbox("Select option", ["Option A", "Option B"])

st.header("Columns for responsive layout")

st.markdown("Use `st.columns()` to display widgets or content side by side.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### column 1")
    st.button("Click me in col 1")

with col2:
    st.markdown("#### column 2")
    st.button("Click me in col 2")

st.header("Expanders for hiding details")

with st.expander("Click to see more"):
    st.markdown("You can hide detailed explanations, long text, or advanced options inside expanders.")

st.markdown("---")
st.success("Try modifying the layout above or combining columns and expanders")
