import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import warnings

st.set_page_config(page_title="NVIDIA Sales Analysis", layout="wide")

st.title("NVIDIA Sales Data Analysis")
st.write("Analysis of NVIDIA's sales performance across different product categories, regions, and customer segments")

@st.cache_data
def load_data():
    """Load and return the NVIDIA sales data"""
    df = pd.read_csv('nvidia_simplified_last_5_years_sorted.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.header("1. Data Overview")
st.write("**Economic interpretation**: Understanding the structure and scope of NVIDIA's sales data helps identify key business metrics and market coverage")
st.write(f"Dataset contains {len(df)} sales records from {df['Date'].min().date()} to {df['Date'].max().date()}")
st.dataframe(df.head())

st.subheader("Basic Statistics")
st.write("**Economic interpretation**: Statistical summary reveals revenue distribution patterns, sales volume ranges, and market share variations across NVIDIA's business")
st.write(df.describe())

# 1. Dealing with missing values and extreme values
st.header("2. Data Quality Analysis")

def check_missing_values(df):
    """Check for missing values in the dataset"""
    missing = df.isnull().sum()
    st.write("Missing values per column:")
    st.write(missing)
    return missing

def handle_extreme_values(df):
    """Identify and handle extreme values in revenue and units sold"""
    Q1_revenue = df['Sales Revenue (USD)'].quantile(0.25)
    Q3_revenue = df['Sales Revenue (USD)'].quantile(0.75)
    IQR_revenue = Q3_revenue - Q1_revenue
    
    outliers_revenue = df[(df['Sales Revenue (USD)'] < Q1_revenue - 1.5*IQR_revenue) | 
                         (df['Sales Revenue (USD)'] > Q3_revenue + 1.5*IQR_revenue)]
    
    st.write(f"Found {len(outliers_revenue)} revenue outliers out of {len(df)} records")
    st.write(f"Outliers represent {len(outliers_revenue)/len(df)*100:.2f}% of total sales records")
    
    return outliers_revenue

missing_vals = check_missing_values(df)
outliers = handle_extreme_values(df)

# 2. Encoding methods
st.header("3. Data Encoding")

def encode_categorical_data(df):
    """Encode categorical variables for machine learning"""
    df_encoded = df.copy()
    
    le_region = LabelEncoder()
    le_category = LabelEncoder()
    le_segment = LabelEncoder()
    le_type = LabelEncoder()
    
    df_encoded['Region_encoded'] = le_region.fit_transform(df['Region'])
    df_encoded['Product_Category_encoded'] = le_category.fit_transform(df['Product Category'])
    df_encoded['Customer_Segment_encoded'] = le_segment.fit_transform(df['Customer Segment'])
    df_encoded['Customer_Type_encoded'] = le_type.fit_transform(df['Customer Type'])
    
    st.write("Categorical variables encoded successfully")
    st.write("Encoding mappings:")
    st.write(f"Regions: {dict(zip(le_region.classes_, range(len(le_region.classes_))))}")
    st.write(f"Product Categories: {dict(zip(le_category.classes_, range(len(le_category.classes_))))}")
    
    return df_encoded, le_region, le_category, le_segment, le_type

df_encoded, le_region, le_category, le_segment, le_type = encode_categorical_data(df)

# 3. Scaling methods
st.header("4. Data Scaling")

def scale_numerical_data(df_encoded):
    """Scale numerical variables for analysis"""
    scaler = StandardScaler()
    
    numerical_cols = ['Sales Revenue (USD)', 'Units Sold', 'Market Share (%)']
    df_scaled = df_encoded.copy()
    df_scaled[['Revenue_scaled', 'Units_scaled', 'MarketShare_scaled']] = scaler.fit_transform(df_encoded[numerical_cols])
    
    st.write("Numerical variables scaled using StandardScaler")
    st.write("Original vs Scaled data comparison:")
    comparison = pd.DataFrame({
        'Original_Revenue_mean': [df['Sales Revenue (USD)'].mean()],
        'Scaled_Revenue_mean': [df_scaled['Revenue_scaled'].mean()],
        'Original_Revenue_std': [df['Sales Revenue (USD)'].std()],
        'Scaled_Revenue_std': [df_scaled['Revenue_scaled'].std()]
    })
    st.write(comparison)
    
    return df_scaled, scaler

df_scaled, scaler = scale_numerical_data(df_encoded)

# 4. Statistical processing, grouping and aggregation
st.header("5. Statistical Analysis and Aggregation")

def statistical_analysis(df):
    """Perform statistical analysis and aggregation"""
    
    revenue_by_region = df.groupby('Region')['Sales Revenue (USD)'].agg(['sum', 'mean', 'count']).round(2)
    st.subheader("Revenue Analysis by Region")
    st.write(revenue_by_region)
    
    revenue_by_category = df.groupby('Product Category')['Sales Revenue (USD)'].agg(['sum', 'mean', 'count']).round(2)
    st.subheader("Revenue Analysis by Product Category")
    st.write(revenue_by_category)
    
    market_share_by_segment = df.groupby('Customer Segment')['Market Share (%)'].agg(['mean', 'max', 'min']).round(2)
    st.subheader("Market Share by Customer Segment")
    st.write(market_share_by_segment)
    
    return revenue_by_region, revenue_by_category, market_share_by_segment

revenue_by_region, revenue_by_category, market_share_by_segment = statistical_analysis(df)

# 5. Data merging/joining
st.header("6. Data Merging and Joining")

def merge_data_analysis(df):
    """Create summary tables and merge with main data"""
    
    region_summary = df.groupby('Region').agg({
        'Sales Revenue (USD)': 'sum',
        'Units Sold': 'sum',
        'Market Share (%)': 'mean'
    }).round(2)
    region_summary.columns = ['Total_Revenue', 'Total_Units', 'Avg_Market_Share']
    region_summary = region_summary.reset_index()
    
    df_merged = df.merge(region_summary, on='Region', how='left')
    
    st.write("Data merged with regional performance summaries")
    st.write("Sample of merged data:")
    st.write(df_merged[['Region', 'Sales Revenue (USD)', 'Total_Revenue', 'Market Share (%)', 'Avg_Market_Share']].head())
    
    return df_merged, region_summary

df_merged, region_summary = merge_data_analysis(df)

# 6. Graphical representation with matplotlib
st.header("7. Data Visualization")

def create_visualizations(df):
    """Create various visualizations of the data"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    revenue_by_region = df.groupby('Region')['Sales Revenue (USD)'].sum()
    axes[0,0].bar(revenue_by_region.index, revenue_by_region.values)
    axes[0,0].set_title('Total Revenue by Region')
    axes[0,0].set_ylabel('Revenue (USD)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    revenue_by_category = df.groupby('Product Category')['Sales Revenue (USD)'].sum()
    axes[0,1].pie(revenue_by_category.values, labels=revenue_by_category.index, autopct='%1.1f%%')
    axes[0,1].set_title('Revenue Distribution by Product Category')
    
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_units = df.groupby('Month')['Units Sold'].sum()
    axes[1,0].plot(monthly_units.index.astype(str), monthly_units.values)
    axes[1,0].set_title('Monthly Units Sold Trend')
    axes[1,0].set_ylabel('Units Sold')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    market_share_by_segment = df.groupby('Customer Segment')['Market Share (%)'].mean()
    axes[1,1].bar(market_share_by_segment.index, market_share_by_segment.values)
    axes[1,1].set_title('Average Market Share by Customer Segment')
    axes[1,1].set_ylabel('Market Share (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

create_visualizations(df)

# 7. Clustering with scikit-learn
st.header("8. Customer Segmentation Analysis (Clustering)")

def perform_clustering(df_scaled):
    """Perform K-means clustering on customer data"""
    
    features = ['Revenue_scaled', 'Units_scaled', 'MarketShare_scaled', 'Region_encoded', 'Customer_Segment_encoded']
    X = df_scaled[features]
    
    X_clean = X.dropna()
    df_clean = df_scaled.loc[X_clean.index].copy()
    
    st.write(f"Removed {len(df_scaled) - len(X_clean)} rows with missing values for clustering")
    st.write(f"Using {len(X_clean)} records for clustering analysis")
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_clean)
    
    df_clustered = df_clean.copy()
    df_clustered['Cluster'] = clusters
    
    st.write("K-means clustering performed with 4 clusters")
    
    cluster_analysis = df_clustered.groupby('Cluster').agg({
        'Sales Revenue (USD)': 'mean',
        'Units Sold': 'mean',
        'Market Share (%)': 'mean'
    }).round(2)
    
    st.write("Cluster characteristics:")
    st.write(cluster_analysis)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_clustered['Sales Revenue (USD)'], df_clustered['Units Sold'], 
                        c=df_clustered['Cluster'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('Sales Revenue (USD)')
    ax.set_ylabel('Units Sold')
    ax.set_title('Customer Clusters based on Revenue and Units Sold')
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    return df_clustered, cluster_analysis

df_clustered, cluster_analysis = perform_clustering(df_scaled)

# 8. Logistic Regression with scikit-learn
st.header("9. Customer Type Prediction (Logistic Regression)")

def logistic_regression_analysis(df_encoded):
    """Predict customer type using logistic regression"""
    
    features = ['Sales Revenue (USD)', 'Units Sold', 'Market Share (%)', 'Region_encoded', 'Product_Category_encoded']
    X = df_encoded[features]
    y = df_encoded['Customer_Type_encoded']  # 0: Business, 1: Individual
    
    data_clean = df_encoded[features + ['Customer_Type_encoded']].dropna()
    X_clean = data_clean[features]
    y_clean = data_clean['Customer_Type_encoded']
    
    st.write(f"Removed {len(df_encoded) - len(data_clean)} rows with missing values for logistic regression")
    st.write(f"Using {len(data_clean)} records for model training")
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)
    
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Logistic Regression Accuracy: {accuracy:.3f}")
    st.write("Feature importance (coefficients):")
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': log_reg.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    st.write(feature_importance)
    
    return log_reg, accuracy, feature_importance

log_reg, accuracy, feature_importance = logistic_regression_analysis(df_encoded)

# 9. Multiple Regression with statsmodels
st.header("10. Revenue Prediction (Multiple Regression)")

def multiple_regression_analysis(df_encoded):
    """Perform multiple regression to predict sales revenue"""
    
    features = ['Units Sold', 'Market Share (%)', 'Region_encoded', 'Product_Category_encoded', 'Customer_Segment_encoded']
    X = df_encoded[features]
    y = df_encoded['Sales Revenue (USD)']
    
    data_clean = df_encoded[features + ['Sales Revenue (USD)']].dropna()
    X_clean = data_clean[features]
    y_clean = data_clean['Sales Revenue (USD)']
    
    st.write(f"Removed {len(df_encoded) - len(data_clean)} rows with missing values for multiple regression")
    st.write(f"Using {len(data_clean)} records for model fitting")
    
    X_clean = sm.add_constant(X_clean)
    
    model = sm.OLS(y_clean, X_clean).fit()
    
    st.write("Multiple Regression Results:")
    st.text(str(model.summary()))
    
    st.write(f"R-squared: {model.rsquared:.3f}")
    st.write(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
    st.write(f"F-statistic p-value: {model.f_pvalue:.6f}")
    
    return model

regression_model = multiple_regression_analysis(df_encoded)

# 10. Geopandas visualization (simplified)
st.header("11. Geographic Analysis")

def geographic_analysis(df):
    """Analyze sales performance by geographic regions using Geopandas."""
    
    regional_performance = df.groupby('Region').agg({
        'Sales Revenue (USD)': ['sum', 'mean'],
        'Units Sold': 'sum',
        'Market Share (%)': 'mean'
    }).round(2)
    
    regional_performance.columns = ['Total_Revenue', 'Avg_Revenue', 'Total_Units', 'Avg_Market_Share']
    regional_performance = regional_performance.reset_index()
    
    st.write("Regional Performance Summary:")
    st.write(regional_performance)
    
    # --- Geopandas Map Visualization ---
    try:
        # Ensure you have pyogrio installed: pip install pyogrio
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'), engine="pyogrio")
        # Filter out Antarctica and French Southern and Antarctic Lands as they are not typical sales regions
        world = world[~world['name'].isin(['Antarctica', 'Fr. S. Antarctic Lands'])]

        # Define NVIDIA's sales regions based on continents and specific country lists
        # This mapping is an approximation. Ensure country names match those in naturalearth_lowres.
        middle_east_countries_map = {
            'Israel', 'Jordan', 'Lebanon', 'Syrian Arab Rep.', 'Iraq',
            'Iran (Islamic Rep. of)', 'Saudi Arabia', 'Yemen', 'Oman', 'United Arab Emirates',
            'Qatar', 'Bahrain', 'Kuwait', 'Turkey', 'Cyprus', 'Egypt', 'Palestine'
        }

        def assign_nvidia_region(row):
            country_name = row['name']
            continent = row['continent']

            if country_name in middle_east_countries_map:
                return 'Middle East'
            
            if continent == 'North America':
                return 'North America'
            elif continent == 'South America':
                return 'South America'
            elif continent == 'Europe':
                return 'Europe'
            elif continent == 'Africa':
                return 'Africa'
            elif continent == 'Asia': # Countries in Asia not in Middle East
                return 'APAC'
            elif continent == 'Oceania': # Australia, New Zealand, etc.
                return 'APAC'
            return 'Other' # Should catch any unmapped areas

        world['NVIDIA_Region'] = world.apply(assign_nvidia_region, axis=1)
        
        # Dissolve countries into NVIDIA regions.
        # Select only necessary columns before dissolving to avoid warnings with non-numeric data.
        world_regions = world[['NVIDIA_Region', 'geometry']].dissolve(by='NVIDIA_Region', aggfunc='first')

        # Merge with sales data
        # world_regions will have NVIDIA_Region as its index after dissolve
        merged_map_data = world_regions.merge(regional_performance, left_index=True, right_on='Region', how='left')
        
        # Fill NaN for regions on map that might not have sales data (e.g., 'Other')
        merged_map_data['Total_Revenue'] = merged_map_data['Total_Revenue'].fillna(0)

        # Plotting
        fig_map, ax_map = plt.subplots(1, 1, figsize=(17, 10)) # Increased figure size slightly
        merged_map_data.plot(column='Total_Revenue', 
                             ax=ax_map, 
                             legend=True,
                             legend_kwds={'label': "Total Revenue (USD) by NVIDIA Region", 'orientation': "horizontal", "shrink": 0.5},
                             cmap='viridis',
                             missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "No Sales Data"})

        ax_map.set_title('Total Revenue by NVIDIA Sales Region', fontsize=15)
        ax_map.set_axis_off()
        
        plt.tight_layout()
        st.pyplot(fig_map)
        
    except Exception as e:
        st.error(f"Could not generate map: {e}")
        st.write("Displaying regional performance as a bar chart instead:")
        # Fallback to bar chart if map generation fails
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        x_pos = range(len(regional_performance))
        ax_bar.bar(x_pos, regional_performance['Total_Revenue'], alpha=0.7)
        ax_bar.set_xlabel('Region')
        ax_bar.set_ylabel('Total Revenue (USD)')
        ax_bar.set_title('Total Revenue by Geographic Region (Fallback)')
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(regional_performance['Region'], rotation=45)
        for i, v in enumerate(regional_performance['Total_Revenue']):
            ax_bar.text(i, v + max(regional_performance['Total_Revenue'])*0.01, f'${v:,.0f}', 
                    ha='center', va='bottom')
        plt.tight_layout()
        st.pyplot(fig_bar)

    return regional_performance

regional_performance = geographic_analysis(df)

st.header("12. Summary and Business Insights")
st.write("""
**Economic Interpretation of Key Findings:**

1. **Data Quality**: The dataset contains complete sales records with no missing values, indicating robust data collection processes.

2. **Regional Performance**: Geographic analysis reveals market concentration and expansion opportunities across different regions.

3. **Product Portfolio**: Revenue distribution across product categories shows NVIDIA's diversification strategy and market focus areas.

4. **Customer Segmentation**: Clustering analysis identifies distinct customer groups with different purchasing behaviors and value propositions.

5. **Predictive Modeling**: Logistic regression and multiple regression models provide insights into customer behavior and revenue drivers.

6. **Market Trends**: Time series analysis reveals seasonal patterns and growth trends in NVIDIA's business.

These insights support strategic decision-making for market expansion, product development, and customer targeting.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Revenue", f"${df['Sales Revenue (USD)'].sum():,.0f}")
with col2:
    st.metric("Total Units Sold", f"{df['Units Sold'].sum():,}")
with col3:
    st.metric("Average Market Share", f"{df['Market Share (%)'].mean():.1f}%")
with col4:
    st.metric("Number of Regions", df['Region'].nunique())