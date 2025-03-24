import streamlit as st

st.set_page_config(page_title="session state")

st.title("Session state in Streamlit")

st.markdown("""
Streamlit reruns your script top to bottom every time a user interacts with the app.  
Because of this, normal Python variables reset unless you explicitly store their state.

Streamlit solves this using `st.session_state`, a dictionary-like object that preserves values across interactions.
""")

st.header("Why does state matter?")

st.markdown("""
Try clicking a button or changing a slider — the whole script reruns.  
So if you want a counter or a toggle to remember its value,  
you **must** use session state.
""")

st.header("How to use st.session_state")

st.code("""
# Initialization (only if not already set)
if "counter" not in st.session_state:
    st.session_state.counter = 0

# Access or update
st.session_state.counter += 1
""", language="python")

st.header("Interactive counter example")

# Counter logic
if "counter" not in st.session_state:
    st.session_state.counter = 0

col1, col2 = st.columns(2)

with col1:
    if st.button("Increment"):
        st.session_state.counter += 1

with col2:
    if st.button("Reset"):
        st.session_state.counter = 0

st.write(f"Current count: {st.session_state.counter}")

st.header("Text input and memory")

name = st.text_input("Enter your name", key="user_name")

if st.button("Say hello"):
    st.write(f"Hello, {st.session_state.user_name}!")

st.markdown("""
When you give a widget a `key`, its value is automatically saved to `st.session_state`  
and updated when the user changes it.
""")

st.header("Checkbox example")

if "show_data" not in st.session_state:
    st.session_state.show_data = False

show = st.checkbox("Show hidden message", key="show_data")

if st.session_state.show_data:
    st.info("This is a hidden message using session state.")

st.header("Manual assignment")

st.markdown("""
You can also manually assign values to `st.session_state`, for example when working with forms or complex logic.
""")

if st.button("Store custom value"):
    st.session_state.my_variable = "custom data"
    st.write("Value stored manually.")

if "my_variable" in st.session_state:
    st.write(f"My variable: {st.session_state.my_variable}")

st.header("View everything in session_state")

st.json(st.session_state)
st.info("Note: Session state keeps values across interactions and page navigation as long as the widget key remains the same and the session is active. For inputs, use default `value` + `key` to sync reliably.")


st.markdown("---")
st.success("Use session state to remember values, share data between widgets, and build dynamic apps.")
