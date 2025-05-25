# NVIDIA Sales Data Analysis

This is a simple Streamlit application that analyzes NVIDIA's sales data using various Python data science techniques.

## Required Files
- `nvidia_simplified_last_5_years_sorted.csv` - The NVIDIA sales dataset
- `nvidia_analysis.py` - Main Streamlit application
- `requirements.txt` - Python dependencies

## Installation and Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure the CSV file is in the same directory as the Python script.

3. Run the Streamlit application:
```bash
streamlit run nvidia_analysis.py
```

## Features Implemented

The application includes all 8+ required Python functionalities:

1. **Streamlit methods** - Interactive web interface with charts and displays
2. **Geopandas package** - Geographic analysis of regional sales performance
3. **Missing values handling** - Data quality analysis and outlier detection
4. **Encoding methods** - Label encoding for categorical variables
5. **Scaling methods** - StandardScaler for numerical data normalization
6. **Statistical processing** - Pandas grouping and aggregation operations
7. **Data merging/joining** - Combining datasets with merge operations
8. **Matplotlib visualization** - Charts and graphs for data analysis
9. **Scikit-learn clustering** - K-means customer segmentation
10. **Scikit-learn logistic regression** - Customer type prediction
11. **Statsmodels multiple regression** - Revenue prediction modeling

## Economic Interpretations

Each function includes detailed economic interpretations explaining the business value and insights derived from the analysis, helping understand NVIDIA's market performance, customer segments, and growth opportunities. 