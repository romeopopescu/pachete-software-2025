/* ========================================================================= */
/* NVIDIA Sales Data Analysis - SAS Programming                             */
/* Project: Analysis of NVIDIA's activity and expansion possibilities       */
/* ========================================================================= */

/* Clear previous work */
proc datasets library=work kill nolist;
quit;

/* Set options for better output */
options nodate nonumber;
title "NVIDIA Sales Data Analysis - SAS Implementation";

/* ========================================================================= */
/* 1. CREATING SAS DATA SET FROM EXTERNAL FILES                             */
/* Economic interpretation: Loading external CSV data enables analysis of   */
/* NVIDIA's sales performance across different markets and time periods     */
/* ========================================================================= */

/* Import CSV data */
proc import datafile="nvidia_simplified_last_5_years_sorted.csv"
    out=nvidia_sales
    dbms=csv
    replace;
    getnames=yes;
run;

/* Display basic information about the dataset */
proc contents data=nvidia_sales;
    title2 "Dataset Structure and Variables";
run;

/* ========================================================================= */
/* 2. CREATING AND USING USER-DEFINED FORMATS                               */
/* Economic interpretation: Custom formats help categorize revenue ranges   */
/* and customer segments for better business analysis and reporting         */
/* ========================================================================= */

/* Create user-defined formats */
proc format;
    value revenue_fmt
        low-<500000 = "Low Revenue"
        500000-<1000000 = "Medium Revenue"
        1000000-<1500000 = "High Revenue"
        1500000-high = "Very High Revenue";
    
    value market_share_fmt
        low-<30 = "Low Market Share"
        30-<50 = "Medium Market Share"
        50-<70 = "High Market Share"
        70-high = "Dominant Market Share";
    
    value region_fmt
        "North America" = "NA"
        "Europe" = "EU"
        "APAC" = "AP"
        "South America" = "SA"
        "Middle East" = "ME"
        "Africa" = "AF";
run;

/* ========================================================================= */
/* 3. ITERATIVE AND CONDITIONAL PROCESSING OF DATA                          */
/* Economic interpretation: Conditional processing identifies high-value    */
/* customers and profitable product categories for strategic focus          */
/* ========================================================================= */

/* Create new variables using conditional processing */
data nvidia_processed;
    set nvidia_sales;
    
    /* Convert date variable */
    date_formatted = input(Date, yymmdd10.);
    format date_formatted date9.;
    
    /* Create revenue categories using conditional logic */
    if Sales_Revenue__USD_ < 500000 then revenue_category = "Low";
    else if Sales_Revenue__USD_ < 1000000 then revenue_category = "Medium";
    else if Sales_Revenue__USD_ < 1500000 then revenue_category = "High";
    else revenue_category = "Very High";
    
    /* Create customer value score using iterative logic */
    customer_value_score = 0;
    if Sales_Revenue__USD_ > 1000000 then customer_value_score + 3;
    if Units_Sold > 3000 then customer_value_score + 2;
    if Market_Share____ > 50 then customer_value_score + 2;
    if Customer_Type = "Business" then customer_value_score + 1;
    
    /* Create profitability indicator */
    if customer_value_score >= 5 then profitability = "High Profit";
    else if customer_value_score >= 3 then profitability = "Medium Profit";
    else profitability = "Low Profit";
    
    /* Apply formats */
    format Sales_Revenue__USD_ revenue_fmt.
           Market_Share____ market_share_fmt.
           Region region_fmt.;
run;

/* ========================================================================= */
/* 4. CREATING DATA SUBSETS                                                 */
/* Economic interpretation: Data subsets help analyze specific market       */
/* segments and high-performing regions for targeted business strategies    */
/* ========================================================================= */

/* Create subset for high-value customers */
data high_value_customers;
    set nvidia_processed;
    where customer_value_score >= 5;
run;

/* Create subset for gaming products */
data gaming_products;
    set nvidia_processed;
    where Product_Category = "Gaming";
run;

/* Create subset for North American market */
data north_america;
    set nvidia_processed;
    where Region = "North America";
run;

/* Display subset information */
proc print data=high_value_customers (obs=10);
    title2 "High Value Customers Sample";
    var Region Product_Category Sales_Revenue__USD_ customer_value_score profitability;
run;

/* ========================================================================= */
/* 5. USING SAS FUNCTIONS                                                   */
/* Economic interpretation: SAS functions enable calculation of key         */
/* business metrics like revenue growth rates and market penetration        */
/* ========================================================================= */

/* Create calculated variables using SAS functions */
data nvidia_calculated;
    set nvidia_processed;
    
    /* Date functions */
    year = year(date_formatted);
    month = month(date_formatted);
    quarter = qtr(date_formatted);
    
    /* Mathematical functions */
    revenue_per_unit = round(Sales_Revenue__USD_ / Units_Sold, 0.01);
    log_revenue = log(Sales_Revenue__USD_);
    sqrt_units = sqrt(Units_Sold);
    
    /* String functions */
    product_length = length(Product_Name);
    region_upper = upcase(Region);
    customer_concat = catx(" - ", Customer_Segment, Customer_Type);
    
    /* Statistical functions */
    revenue_rank = rank(Sales_Revenue__USD_);
run;

/* ========================================================================= */
/* 6. COMBINING DATA SETS WITH SAS PROCEDURES                               */
/* Economic interpretation: Combining datasets provides comprehensive view  */
/* of regional performance and customer behavior patterns                   */
/* ========================================================================= */

/* Create regional summary */
proc sql;
    create table regional_summary as
    select Region,
           count(*) as total_transactions,
           sum(Sales_Revenue__USD_) as total_revenue,
           mean(Sales_Revenue__USD_) as avg_revenue,
           sum(Units_Sold) as total_units,
           mean(Market_Share____) as avg_market_share
    from nvidia_calculated
    group by Region;
quit;

/* Merge main data with regional summary */
proc sql;
    create table nvidia_merged as
    select a.*,
           b.total_transactions,
           b.total_revenue as region_total_revenue,
           b.avg_revenue as region_avg_revenue
    from nvidia_calculated a
    left join regional_summary b
    on a.Region = b.Region;
quit;

/* ========================================================================= */
/* 7. USING ARRAYS                                                          */
/* Economic interpretation: Arrays enable efficient processing of multiple  */
/* performance metrics to identify trends and outliers in sales data        */
/* ========================================================================= */

/* Use arrays to process multiple variables */
data nvidia_arrays;
    set nvidia_merged;
    
    /* Array for standardizing numeric variables */
    array nums{3} Sales_Revenue__USD_ Units_Sold Market_Share____;
    array std_nums{3} std_revenue std_units std_market_share;
    
    /* Standardize variables using arrays */
    do i = 1 to 3;
        std_nums{i} = (nums{i} - mean(nums{i})) / std(nums{i});
    end;
    
    /* Array for creating performance indicators */
    array indicators{3} revenue_indicator units_indicator market_indicator;
    
    do j = 1 to 3;
        if std_nums{j} > 1 then indicators{j} = "Above Average";
        else if std_nums{j} > 0 then indicators{j} = "Average";
        else indicators{j} = "Below Average";
    end;
    
    drop i j;
run;

/* ========================================================================= */
/* 8. USING REPORT PROCEDURES                                               */
/* Economic interpretation: Reports provide executive summaries of key      */
/* business metrics and performance indicators for decision making          */
/* ========================================================================= */

/* Revenue report by region */
proc report data=nvidia_arrays nowd;
    title2 "Revenue Analysis by Region";
    column Region Sales_Revenue__USD_=revenue_sum Sales_Revenue__USD_=revenue_mean Units_Sold=units_sum;
    define Region / group "Region" width=15;
    define revenue_sum / analysis sum "Total Revenue" format=dollar15.2;
    define revenue_mean / analysis mean "Average Revenue" format=dollar12.2;
    define units_sum / analysis sum "Total Units" format=comma12.;
    rbreak after / summarize;
run;

/* Product category performance report */
proc report data=nvidia_arrays nowd;
    title2 "Product Category Performance";
    column Product_Category customer_value_score=score_mean Market_Share____=share_mean;
    define Product_Category / group "Product Category" width=15;
    define score_mean / analysis mean "Avg Customer Value" format=6.2;
    define share_mean / analysis mean "Avg Market Share %" format=6.2;
run;

/* ========================================================================= */
/* 9. USING STATISTICAL PROCEDURES                                          */
/* Economic interpretation: Statistical analysis reveals relationships      */
/* between variables and helps predict future sales performance             */
/* ========================================================================= */

/* Descriptive statistics */
proc means data=nvidia_arrays n mean std min max;
    title2 "Descriptive Statistics for Key Variables";
    var Sales_Revenue__USD_ Units_Sold Market_Share____ customer_value_score;
run;

/* Correlation analysis */
proc corr data=nvidia_arrays;
    title2 "Correlation Analysis";
    var Sales_Revenue__USD_ Units_Sold Market_Share____ customer_value_score;
run;

/* Frequency analysis */
proc freq data=nvidia_arrays;
    title2 "Frequency Analysis";
    tables Region*Product_Category / chisq;
    tables revenue_category*profitability;
run;

/* ANOVA - Revenue by Product Category */
proc anova data=nvidia_arrays;
    title2 "ANOVA: Revenue by Product Category";
    class Product_Category;
    model Sales_Revenue__USD_ = Product_Category;
    means Product_Category / tukey;
run;

/* ========================================================================= */
/* 10. GENERATING GRAPHS                                                    */
/* Economic interpretation: Visual analysis reveals trends, patterns, and   */
/* outliers in sales data for better business understanding                 */
/* ========================================================================= */

/* Revenue by region bar chart */
proc sgplot data=regional_summary;
    title2 "Total Revenue by Region";
    vbar Region / response=total_revenue;
    xaxis label="Region";
    yaxis label="Total Revenue (USD)" grid;
run;

/* Scatter plot: Revenue vs Units Sold */
proc sgplot data=nvidia_arrays;
    title2 "Revenue vs Units Sold Relationship";
    scatter x=Units_Sold y=Sales_Revenue__USD_ / group=Product_Category;
    xaxis label="Units Sold" grid;
    yaxis label="Sales Revenue (USD)" grid;
run;

/* Box plot: Market Share by Customer Type */
proc sgplot data=nvidia_arrays;
    title2 "Market Share Distribution by Customer Type";
    vbox Market_Share____ / category=Customer_Type;
    xaxis label="Customer Type";
    yaxis label="Market Share %" grid;
run;

/* Time series plot */
proc sgplot data=nvidia_arrays;
    title2 "Monthly Revenue Trend";
    series x=date_formatted y=Sales_Revenue__USD_ / group=Product_Category;
    xaxis label="Date" grid;
    yaxis label="Sales Revenue (USD)" grid;
run;

/* ========================================================================= */
/* 11. BASIC MACHINE LEARNING WITH SAS                                      */
/* Economic interpretation: Predictive models help forecast sales and       */
/* identify factors that drive revenue growth for strategic planning        */
/* ========================================================================= */

/* Linear regression model */
proc reg data=nvidia_arrays;
    title2 "Linear Regression: Predicting Sales Revenue";
    model Sales_Revenue__USD_ = Units_Sold Market_Share____ customer_value_score;
    output out=regression_results predicted=predicted_revenue residual=residual;
run;

/* Cluster analysis */
proc cluster data=nvidia_arrays method=ward outtree=tree;
    title2 "Cluster Analysis of Customer Segments";
    var std_revenue std_units std_market_share;
run;

/* Create clusters */
proc tree data=tree out=clusters nclusters=4;
run;

/* Analyze clusters */
proc means data=clusters;
    title2 "Cluster Characteristics";
    class cluster;
    var Sales_Revenue__USD_ Units_Sold Market_Share____ customer_value_score;
run;

/* ========================================================================= */
/* 12. SUMMARY AND BUSINESS INSIGHTS                                        */
/* Economic interpretation: Final summary provides actionable insights for  */
/* NVIDIA's business strategy and market expansion decisions                 */
/* ========================================================================= */

/* Final summary report */
proc sql;
    title2 "Executive Summary - Key Business Metrics";
    select "Total Records" as Metric, count(*) as Value from nvidia_arrays
    union
    select "Total Revenue (USD)", sum(Sales_Revenue__USD_) from nvidia_arrays
    union
    select "Average Revenue per Transaction", mean(Sales_Revenue__USD_) from nvidia_arrays
    union
    select "Total Units Sold", sum(Units_Sold) from nvidia_arrays
    union
    select "Average Market Share %", mean(Market_Share____) from nvidia_arrays;
quit;

/* Regional performance ranking */
proc sql;
    title2 "Regional Performance Ranking";
    select Region,
           sum(Sales_Revenue__USD_) as Total_Revenue format=dollar15.2,
           count(*) as Transactions,
           mean(Market_Share____) as Avg_Market_Share format=6.2
    from nvidia_arrays
    group by Region
    order by Total_Revenue desc;
quit;

/* Product category insights */
proc sql;
    title2 "Product Category Business Insights";
    select Product_Category,
           count(*) as Transactions,
           sum(Sales_Revenue__USD_) as Total_Revenue format=dollar15.2,
           mean(customer_value_score) as Avg_Customer_Value format=6.2,
           mean(Market_Share____) as Avg_Market_Share format=6.2
    from nvidia_arrays
    group by Product_Category
    order by Total_Revenue desc;
quit;

/* End of program */
title;
footnote "Analysis completed: NVIDIA Sales Data - SAS Implementation";
footnote2 "Economic insights provided for strategic business decision making"; 