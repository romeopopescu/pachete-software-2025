import streamlit as st

st.set_page_config(page_title="widgets and interaction")

st.title("Widgets and interaction in Streamlit")

st.markdown("""
Widgets are the core of interactivity in Streamlit.  
They allow users to input data, make selections, and trigger actions.
""")

st.header("Basic input widgets")

st.markdown("### Text and number inputs")
st.code("""
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=120)
""", language="python")

name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=120)

if name:
    st.write(f"Hello, {name}!")

if age:
    st.write(f"You are {int(age)} years old.")

st.markdown("### Selectboxes and radio buttons")
st.code("""
option = st.selectbox("Choose an option", ["Option A", "Option B", "Option C"])
choice = st.radio("Pick one", ["Yes", "No"])
""", language="python")

option = st.selectbox("Choose an option", ["Option A", "Option B", "Option C"])
choice = st.radio("Pick one", ["Yes", "No", "Maybe"])

st.write(f"You chose: {option}, {choice}")

st.header("Checkboxes and buttons")

st.markdown("Use checkboxes and buttons to trigger custom behavior.")

st.code("""
if st.checkbox("Show message"):
    st.write("Checkbox is selected")

if st.button("Click me"):
    st.write("Button was clicked")
""", language="python")

if st.checkbox("Show message"):
    st.write("Checkbox is selected")

if st.button("Click me"):
    st.write("Button was clicked")

st.header("Forms for grouped input")

st.markdown("Forms help you submit multiple inputs at once.")

with st.form("my_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write(f"Logged in as: {username}")

st.header("Additional useful widgets")

st.subheader("Slider")
slider_value = st.slider("Select a value", 0, 100)
st.write(f"Slider value: {slider_value}")

range_slider = st.slider("Select a range", 0, 100, (10, 200))
st.write(f"Range selected: {range_slider}")

st.subheader("Date and time input")
date = st.date_input("Pick a date")
time = st.time_input("Pick a time")
st.write(f"You picked {date} at {time}")

st.subheader("Color picker")
color = st.color_picker("Pick a color", "#00f900")
st.write(f"Selected color: {color}")

st.subheader("Multiselect")
skills = st.multiselect("Select your skills", ["Python", "SQL", "Streamlit", "Pandas"])
st.write("You selected:", skills)

st.subheader("Select slider (categorical values)")
mood = st.select_slider("How do you feel today?", options=["😞", "😐", "😊"])
st.write("Mood:", mood)

st.subheader("File uploader")
file = st.file_uploader("Upload a file")
if file:
    st.write("File uploaded:", file.name)

st.subheader("Dependent widgets example")
fruit = st.selectbox("Choose a fruit", ["Apple", "Banana", "Cherry"])
quantity = st.slider(f"How many {fruit.lower()}s?", 1, 10)
st.write(f"You chose {quantity} {fruit.lower()}(s)")

st.markdown("---")
st.success("Try interacting with all the widgets above and observe how the app responds in real time")
