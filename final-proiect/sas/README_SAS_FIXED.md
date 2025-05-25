# NVIDIA Sales Data Analysis - SAS Implementation (FIXED VERSION)

## Overview
This is the corrected version of the SAS program for analyzing NVIDIA's sales data. All technical issues from the original implementation have been resolved.

## Files
- `nvidia_analysis_fixed.sas` - Main corrected SAS program
- `nvidia_simplified_last_5_years_sorted.csv` - Dataset (copy in sas/ directory)
- `README_SAS_FIXED.md` - This documentation file

## Issues Fixed

### 1. Variable Name Problems
**Problem**: SAS automatically converts column names with spaces and special characters to underscores.
- `Sales Revenue (USD)` → `Sales_Revenue__USD_`
- `Market Share (%)` → `Market_Share____`
- `Product Category` → `Product_Category`

**Solution**: Updated all variable references to use the correct SAS-converted names.

### 2. Format Definition Errors
**Problem**: Incorrect format type for character variables.
```sas
/* WRONG */
value region_fmt
    "North America" = "NA"

/* CORRECT */
value $region_fmt
    "North America" = "NA"
```

**Solution**: Added `$` prefix for character formats.

### 3. Date Conversion Issues
**Problem**: Invalid date conversion causing errors.
```sas
/* WRONG */
date_formatted = input(Date, yymmdd10.);

/* CORRECT */
if not missing(Date) then do;
    date_formatted = Date;
    format date_formatted date9.;
end;
```

**Solution**: Proper date handling with missing value checks.

### 4. Missing Data Handling
**Problem**: Operations on missing values causing errors and incorrect results.

**Solution**: Added comprehensive missing value checks:
```sas
if not missing(Sales_Revenue__USD_) then do;
    /* processing logic */
end;
```

### 5. Array Standardization Errors
**Problem**: Incorrect STD function usage in arrays.
```sas
/* WRONG */
std_nums{i} = (nums{i} - mean(nums{i})) / std(nums{i});

/* CORRECT */
/* Pre-calculate statistics */
proc means data=nvidia_merged noprint;
    var Sales_Revenue__USD_ Units_Sold Market_Share____;
    output out=stats_summary
           mean=mean_revenue mean_units mean_market_share
           std=std_revenue std_units std_market_share;
run;

/* Then use in data step */
std_nums{i} = (nums{i} - means{i}) / stds{i};
```

**Solution**: Pre-calculated statistics using PROC MEANS, then used arrays properly.

### 6. Variable Initialization
**Problem**: Uninitialized variables causing missing values.

**Solution**: Added proper variable initialization and conditional processing.

## Required Functionalities Implemented

### ✅ 1. Creating SAS Dataset from External Files
- `PROC IMPORT` to load CSV data
- Proper handling of variable names and formats

### ✅ 2. User-Defined Formats
- Revenue categorization formats
- Market share classification formats
- Region abbreviation formats (character format with `$`)

### ✅ 3. Iterative and Conditional Processing
- Customer value scoring algorithm
- Revenue categorization logic
- Profitability indicators

### ✅ 4. Creating Data Subsets
- High-value customers subset
- Gaming products subset
- Regional market subsets

### ✅ 5. Using SAS Functions
- **Date functions**: `YEAR()`, `MONTH()`, `QTR()`
- **Mathematical functions**: `ROUND()`, `LOG()`, `SQRT()`
- **String functions**: `LENGTH()`, `UPCASE()`, `CATX()`
- **Statistical functions**: `RANK()`

### ✅ 6. Combining Datasets with SAS/SQL Procedures
- Regional summary creation with `PROC SQL`
- Dataset merging with `LEFT JOIN`
- Aggregation and grouping operations

### ✅ 7. Using Arrays
- Standardization of multiple numeric variables
- Performance indicator creation
- Efficient processing of variable groups

### ✅ 8. Using Report Procedures
- `PROC REPORT` for executive summaries
- Revenue analysis by region
- Product category performance reports

### ✅ 9. Using Statistical Procedures
- `PROC MEANS` - Descriptive statistics
- `PROC CORR` - Correlation analysis
- `PROC FREQ` - Frequency analysis with chi-square tests
- `PROC ANOVA` - Analysis of variance with Tukey's test

### ✅ 10. Generating Graphs
- `PROC SGPLOT` for various visualizations:
  - Bar charts for regional revenue
  - Scatter plots for revenue vs units relationship
  - Box plots for market share distribution
  - Time series plots for trend analysis

### ✅ 11. Basic Machine Learning
- `PROC REG` - Linear regression modeling
- `PROC CLUSTER` - Ward's clustering method
- `PROC TREE` - Cluster tree creation
- Customer segmentation analysis

## Setup Instructions

### 1. File Preparation
```sas
/* Ensure the CSV file is in the correct location */
/* Update the path in the program if needed */
proc import datafile="/home/u64228087/proiect/nvidia_simplified_last_5_years_sorted.csv"
    out=nvidia_sales
    dbms=csv
    replace;
    getnames=yes;
run;
```

### 2. Running the Program
1. Open SAS Studio or SAS Enterprise Guide
2. Load the `nvidia_analysis_fixed.sas` file
3. Update the file path in the `PROC IMPORT` statement if necessary
4. Run the entire program

### 3. Expected Output
The program will generate:
- Dataset structure information
- Statistical summaries and reports
- Correlation matrices
- ANOVA results
- Multiple visualizations (bar charts, scatter plots, box plots, time series)
- Regression analysis results
- Cluster analysis results
- Executive summary reports

## Economic Interpretations

Each section includes detailed economic interpretations:

1. **Regional Analysis**: Identifies top-performing markets for expansion
2. **Customer Segmentation**: Reveals high-value customer characteristics
3. **Product Performance**: Shows which product categories drive revenue
4. **Market Share Analysis**: Indicates competitive positioning
5. **Predictive Modeling**: Forecasts revenue based on key factors
6. **Cluster Analysis**: Groups customers for targeted marketing

## Technical Notes

### Variable Naming Convention
- SAS converts spaces to underscores
- Special characters become underscores
- Multiple spaces/characters become multiple underscores

### Missing Data Strategy
- All operations include missing value checks
- Conditional processing prevents errors
- Statistical procedures handle missing values appropriately

### Performance Optimization
- Pre-calculated statistics for array operations
- Efficient SQL joins for data combination
- Proper indexing for large dataset operations

## Troubleshooting

### Common Issues
1. **File Path Error**: Update the path in `PROC IMPORT`
2. **Variable Not Found**: Check variable names match SAS conventions
3. **Format Errors**: Ensure character formats use `$` prefix
4. **Missing Values**: All operations include missing value checks

### Verification Steps
1. Check `PROC CONTENTS` output for correct variable names
2. Review `PROC PRINT` sample to verify data loading
3. Monitor SAS log for any remaining warnings or errors

## Output Files Generated
- `nvidia_sales` - Original imported dataset
- `nvidia_processed` - Dataset with calculated variables
- `nvidia_calculated` - Dataset with SAS functions applied
- `nvidia_merged` - Dataset merged with regional summaries
- `nvidia_arrays` - Final dataset with standardized variables
- `regional_summary` - Regional performance summary
- `regression_results` - Regression model results
- `clusters` - Customer cluster assignments

This fixed version addresses all technical issues while maintaining the comprehensive analysis required for the project. 