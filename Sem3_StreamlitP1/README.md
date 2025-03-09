# Data preprocessing and machine learning: a streamlit guide

## Introduction

This repository contains an **interactive Streamlit application** that guides students through essential **data preprocessing, feature engineering, visualization, and model evaluation techniques** in machine learning.

This project is divided into **six structured sections**, covering the following topics:

1. **Handling missing values, standardization, and normalization**
2. **Encoding categorical variables and handling outliers**
3. **Feature engineering and variable selection**
4. **Data visualization and correlation analysis**
5. **Train-test split and machine learning models**
6. **Model evaluation using decision trees and random forests**

Each section includes **theoretical explanations, Python code implementations, and interactive visualizations** to enhance learning.

---

## Datasets used

The datasets used in this project are **publicly available** and can be downloaded from the following sources:

- **Titanic dataset** _(for handling missing values and categorical encoding)_  
  [Download from Kaggle](https://www.kaggle.com/c/titanic/data)

- **Housing dataset** _(for feature engineering and visualization)_  
  [Download from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

- **Default of Credit Card Clients dataset** _(for model training and evaluation)_  
  [Download from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

**Before running the application, ensure that all datasets are placed in the appropriate directory.**

---

## Installation guide

### **Clone the repository**

First, clone this repository to your local machine:

```sh
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

```

### **Create a virtual environment**

It is recommended to use a virtual environment to manage dependencies. Follow the steps below based on your operating system.

**Windows**

```sh
python -m venv venv
venv\Scripts\activate

```

**macOS/Linux**
python3 -m venv venv
source venv/bin/activate

### **Install dependencies**

Once the virtual environment is activated, install the required Python libraries:

```sh
pip install -r requirements.txt
```

### **Run the Streamlit application**

```sh
streamlit run Intro.py
```
