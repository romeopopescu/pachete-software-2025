import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Caching and performance")

st.title("Caching and performance in Streamlit")

st.markdown("""
Caching helps speed up your app by avoiding unnecessary recomputations.  
Streamlit provides two main decorators:

- `@st.cache_data`: for caching data-related operations (e.g. loading, processing)
- `@st.cache_resource`: for caching resources like models, connections, or large objects

These are important when you’re working with heavy computation, slow I/O, or external APIs.
""")

st.header("Simulating a slow function (no caching)")

def slow_function_no_cache():
    time.sleep(3)
    return pd.DataFrame({"a": range(5)})

if st.button("Run without caching"):
    st.write("Running slow function...")
    start = time.time()
    df = slow_function_no_cache()
    end = time.time()
    st.write(df)
    st.info(f"Time taken: {round(end - start, 2)} seconds")

st.header("Using st.cache_data")

@st.cache_data
def slow_function_with_cache():
    time.sleep(3)
    return pd.DataFrame({"a": range(5)})

if st.button("Run with st.cache_data"):
    st.write("Running cached function...")
    start = time.time()
    df_cached = slow_function_with_cache()
    end = time.time()
    st.write(df_cached)
    st.success(f"Time taken: {round(end - start, 2)} seconds (faster on repeat runs)")

st.markdown("""
The first run is slow because the function is executed,  
but Streamlit caches the result so future runs are instant unless:
- The code inside the function changes
- The inputs change
""")

st.header("Using st.cache_resource")

st.markdown("Use this for models, clients, or heavy reusable objects.")

@st.cache_resource
def load_heavy_resource():
    time.sleep(3)
    return "Loaded a heavy resource"

if st.button("Load resource"):
    st.write("Loading resource...")
    start = time.time()
    resource = load_heavy_resource()
    end = time.time()
    st.success(f"{resource} (loaded in {round(end - start, 2)} seconds)")

st.header("When not to use caching")

st.markdown("""
Do not cache functions that:
- Depend on rapidly changing data (e.g. timestamps, real-time APIs)
- Have side effects (e.g. writing to files or databases)
- Use user input in uncontrolled ways

Use `st.cache_data` only when you're confident the result can be reused without issues.
""")

st.header("How caching works")

st.markdown("""
- Streamlit computes a **hash of the function code and input arguments**
- If nothing changed, the result is reused
- If anything changes, the function is re-executed

This makes Streamlit fast but also deterministic — results are tied to specific code and inputs.
""")

st.markdown("---")
st.success("Use caching to make your app more efficient, but test carefully to avoid stale or incorrect results")
