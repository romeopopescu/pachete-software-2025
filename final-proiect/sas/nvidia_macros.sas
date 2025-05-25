/* ========================================================================= */
/* NVIDIA Sales Data Analysis - SAS Macros                                  */
/* Reusable macros for enhanced analysis capabilities                       */
/* ========================================================================= */

/* Macro to create summary statistics for any numeric variable */
%macro summary_stats(dataset, variable, title_text);
    proc means data=&dataset n mean std min max median;
        title "&title_text - Summary Statistics for &variable";
        var &variable;
    run;
%mend summary_stats;

/* Macro to create frequency tables for categorical variables */
%macro freq_analysis(dataset, variable, title_text);
    proc freq data=&dataset;
        title "&title_text - Frequency Analysis for &variable";
        tables &variable / nocum;
    run;
%mend freq_analysis;

/* Macro to create cross-tabulation analysis */
%macro crosstab_analysis(dataset, var1, var2, title_text);
    proc freq data=&dataset;
        title "&title_text - Cross-tabulation: &var1 by &var2";
        tables &var1 * &var2 / chisq;
    run;
%mend crosstab_analysis;

/* Macro to create regional performance report */
%macro regional_report(dataset, metric, title_text);
    proc sql;
        title "&title_text - Regional Performance: &metric";
        select Region,
               count(*) as Transactions,
               sum(&metric) as Total format=comma15.2,
               mean(&metric) as Average format=comma12.2,
               std(&metric) as StdDev format=comma10.2
        from &dataset
        group by Region
        order by Total desc;
    quit;
%mend regional_report;

/* Macro to create product category analysis */
%macro product_analysis(dataset, metric, title_text);
    proc sql;
        title "&title_text - Product Category Analysis: &metric";
        select Product_Category,
               count(*) as Transactions,
               sum(&metric) as Total format=comma15.2,
               mean(&metric) as Average format=comma12.2
        from &dataset
        group by Product_Category
        order by Total desc;
    quit;
%mend product_analysis;

/* Example usage of macros (commented out) */
/*
%summary_stats(nvidia_arrays, Sales_Revenue__USD_, NVIDIA Revenue Analysis);
%freq_analysis(nvidia_arrays, Product_Category, Product Distribution);
%crosstab_analysis(nvidia_arrays, Region, Customer_Type, Regional Customer Analysis);
%regional_report(nvidia_arrays, Sales_Revenue__USD_, Revenue Performance);
%product_analysis(nvidia_arrays, Units_Sold, Units Sold Performance);
*/ 