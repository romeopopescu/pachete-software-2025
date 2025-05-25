# NVIDIA Sales Data Analysis - SAS Implementation

This folder contains the SAS programming implementation for analyzing NVIDIA's sales data and expansion possibilities.

## Files

- `nvidia_analysis.sas` - Main SAS program with all required functionalities
- `README_SAS.md` - This documentation file

## Required SAS Functionalities Implemented (8+ minimum)

✅ **1. Creating SAS data set from external files** - Imports CSV data using PROC IMPORT  
✅ **2. Creating and using user-defined formats** - Custom formats for revenue ranges and market segments  
✅ **3. Iterative and conditional processing** - Customer value scoring and profitability categorization  
✅ **4. Creating data subsets** - High-value customers, gaming products, regional subsets  
✅ **5. Using SAS functions** - Date, mathematical, string, and statistical functions  
✅ **6. Combining data sets with SAS/SQL procedures** - Regional summaries and data merging  
✅ **7. Using arrays** - Efficient processing of multiple performance metrics  
✅ **8. Using report procedures** - Executive reports with PROC REPORT  
✅ **9. Using statistical procedures** - PROC MEANS, PROC CORR, PROC FREQ, PROC ANOVA  
✅ **10. Generating graphs** - Bar charts, scatter plots, box plots, time series with PROC SGPLOT  
✅ **11. Basic machine learning** - Linear regression and cluster analysis  

## How to Run

1. **Prerequisites:**
   - SAS software (SAS Studio, SAS Enterprise Guide, or Base SAS)
   - Access to the CSV file: `nvidia_simplified_last_5_years_sorted.csv`

2. **Setup:**
   - Ensure the CSV file is in the parent directory (one level up from the sas folder)
   - Open SAS and navigate to the sas folder

3. **Execution:**
   - Open `nvidia_analysis.sas` in your SAS environment
   - Run the entire program (F3 or Submit All)
   - Review the output in the Results window

## Program Structure

### Data Processing Steps:
1. **Data Import** - Load CSV data into SAS dataset
2. **Data Formatting** - Apply custom formats for better readability
3. **Data Enhancement** - Create calculated variables and business metrics
4. **Data Subsetting** - Create focused datasets for specific analysis
5. **Data Integration** - Combine datasets using SQL procedures

### Analysis Components:
1. **Descriptive Analysis** - Basic statistics and data exploration
2. **Business Intelligence** - Customer value scoring and profitability analysis
3. **Statistical Analysis** - Correlations, ANOVA, frequency analysis
4. **Visual Analysis** - Charts and graphs for trend identification
5. **Predictive Modeling** - Regression and clustering for insights

## Economic Interpretations

Each section includes detailed economic interpretations explaining:

- **Business Value** - How each analysis contributes to business understanding
- **Strategic Insights** - What the results mean for NVIDIA's strategy
- **Decision Support** - How findings can guide business decisions
- **Market Intelligence** - Regional and product performance insights

## Key Business Insights Generated

1. **Regional Performance** - Revenue and market share by geographic region
2. **Product Portfolio Analysis** - Performance across different product categories
3. **Customer Segmentation** - High-value customer identification and characteristics
4. **Profitability Analysis** - Customer value scoring and profit categorization
5. **Market Trends** - Time-based analysis of sales patterns
6. **Predictive Models** - Revenue forecasting and customer clustering

## Output Reports

The program generates comprehensive reports including:

- Executive summary with key business metrics
- Regional performance rankings
- Product category business insights
- Statistical analysis results
- Visual charts and graphs
- Predictive model results

## Technical Notes

- All variable names are automatically adjusted for SAS naming conventions
- Economic interpretations are provided as comments throughout the code
- The program is designed to run sequentially from top to bottom
- Output is formatted for professional business reporting
- Error handling is built into data processing steps 