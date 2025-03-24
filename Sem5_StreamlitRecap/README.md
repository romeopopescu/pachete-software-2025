# Streamlit recap app: interactive guide for building Streamlit apps

## Overview

This repository contains an **interactive Streamlit application** designed as a **recap for students** who have previously learned Streamlit fundamentals.  
The app walks through key concepts used in real-world Streamlit apps, combining **explanations**, **code**, and **live examples**.

The app is organized into **six main sections**, each focused on a core building block of Streamlit:

1. **Layout and structure**  
   How to organize your app using titles, columns, sidebars, and expanders.

2. **Widgets and interactions**  
   How to use input elements like text boxes, sliders, forms, buttons, checkboxes, and more.

3. **Displaying data**  
   Methods for showing tables, JSON, and metrics using pandas and Streamlit’s built-in display tools.

4. **Visualization**  
   Creating interactive plots using Plotly, including line charts, bar charts, scatter plots, and boxplots.

5. **Caching and performance**  
   Improving app speed with `st.cache_data` and `st.cache_resource` to avoid unnecessary recomputation.

6. **Session state**  
   Preserving user inputs and state across interactions using `st.session_state`.

Each section is implemented as a separate page within the Streamlit app, accessible through the sidebar.

---

## Installation guide

### **Clone the repository**

First, clone this repository to your local machine:

```sh
git clone https://github.com/alingabriel743/software-packages-2025/tree/main
cd software-packages-2025/Sem5_StreamlitRecap

```

### **Create a virtual environment**

It is recommended to use a virtual environment to manage dependencies. Follow the steps below based on your operating system.

**Windows**

```sh
python -m venv venv
venv\Scripts\activate

```

**macOS/Linux**

```sh
python3 -m venv venv
source venv/bin/activate
```

### **Install dependencies**

Once the virtual environment is activated, install the required Python libraries:

```sh
pip install -r requirements.txt
```

### **Run the Streamlit application**

```sh
streamlit run Intro.py
```
