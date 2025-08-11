#!/usr/bin/env python3
"""
Panel Regression Analysis Script for CAISO and ERCOT

This script performs panel regression analyses on electricity generation data for two
balancing authorities: CAISO (California Independent System Operator) and ERCOT 
(Electric Reliability Council of Texas). The analysis examines the relationship between
generation, CO2 emissions, and emissions intensity with various explanatory variables
including renewable energy sources, demand, and external factors.

IMPORTANT: This script maintains exact reproducibility of the original paper.
All variable selections, statistical tests, and output formats match the original
research methodology exactly.

The script generates:
1. Text files with detailed regression results for each BA
2. CSV files recreating Table 2 structure
3. Comprehensive model specifications and robustness checks
4. Correlation matrices and statistical diagnostic tests
5. Summary statistics for all variables

Author: Dhruv Suri
Date: August 2025
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set pandas options for better data handling
pd.set_option("mode.chained_assignment", None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class PanelRegressionAnalyzer:
    """
    A class to perform panel regression analysis on electricity generation data.
    
    This class handles data loading, preprocessing, model specification, and
    results generation for both CAISO and ERCOT balancing authorities.
    """
    
    def __init__(self, ba_name, merged_file, control_file):
        """
        Initialize the analyzer with BA-specific data files.
        
        Parameters:
        -----------
        ba_name : str
            Name of the balancing authority (e.g., 'CAISO', 'ERCOT')
        merged_file : str
            Path to the merged data file containing generation and emissions data
        control_file : str
            Path to the control file containing external variables
        """
        self.ba_name = ba_name
        self.merged_file = merged_file
        self.control_file = control_file
        self.data = None
        self.processed_data = None
        
    def load_and_process_data(self):
        """
        Load and preprocess the data for analysis.
        
        This method:
        1. Loads the merged generation data and control variables
        2. Processes external control variables by aggregating to daily values
        3. Merges datasets and adds time-based features
        4. Filters out invalid data points
        5. Calculates derived variables like residual demand
        """
        print(f"Loading and processing data for {self.ba_name}...")
        
        # Load the main merged dataset
        self.data = pd.read_csv(
            self.merged_file,
            parse_dates=["timestamp"],
            skiprows=0
        )
        
        # Load external control variables
        ext_data = pd.read_csv(
            self.control_file,
            parse_dates=["Local time"],
            skiprows=0
        )
        
        # Rename timestamp column for consistency
        ext_data = ext_data.rename(columns={"Local time": "timestamp"})
        
        # Aggregate external variables to daily values
        # Select only numeric columns for aggregation, excluding timestamp
        columns_to_sum = [col for col in ext_data.columns if col != 'timestamp']
        ext_daily = ext_data.groupby(ext_data.timestamp.dt.date)[columns_to_sum].sum().reset_index()
        
        # Convert date back to datetime and rename to timestamp
        ext_daily['timestamp'] = pd.to_datetime(ext_daily['timestamp'])
        
        # Merge datasets
        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.merge(ext_daily, on="timestamp", how="left")
        
        # Add time-based features for fixed effects
        self.data["month"] = self.data.timestamp.dt.month
        self.data["year"] = self.data.timestamp.dt.year
        self.data["day"] = self.data.timestamp.dt.day
        
        # Remove rows with missing values
        self.data.dropna(inplace=True)
        
        # Filter out rows with non-positive values in key variables
        # This ensures log transformations are valid
        positive_value_columns = ["solar", "wind", "SUN_ext", "WND_ext", "D_ext", "Wramp"]
        for column in positive_value_columns:
            if column in self.data.columns:
                self.data = self.data[self.data[column] > 0]
        
        # Calculate residual demand (demand + imports - hydro)
        # This represents the net demand that must be met by non-hydro generation
        self.data["residual_demand"] = self.data.demand + self.data.imports - self.data.hydro
        
        print(f"Data processing complete. Shape: {self.data.shape}")
        
    def log_transform_data(self, df):
        """
        Apply log transformation to specified columns.
        
        Log transformation is applied to normalize the distribution of variables
        and improve the statistical properties of the regression model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe to be transformed
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with log-transformed columns
        """
        # Create a copy to avoid modifying original data
        df_transformed = df.copy()
        
        # Remove any zero values that would cause log(0) = -inf
        df_transformed = df_transformed.replace(0, np.nan).dropna()
        
        # Remove infinite values that may result from log transformations
        df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Take absolute value of imports to handle negative values
        df_transformed["imports"] = df_transformed["imports"].abs()
        
        # Define columns to be log-transformed
        log_transform_columns = [
            "generation", "residual_demand", "imports", "hydro", 
            "solar", "wind", "SUN_ext", "WND_ext", "D_ext", 
            "Wramp", "co2", "intensity"
        ]
        
        # Apply log transformation to specified columns
        for column in log_transform_columns:
            if column in df_transformed.columns:
                df_transformed[column] = np.log(df_transformed[column])
        
        return df_transformed
    
    def run_regression_models(self, dependent_variable):
        """
        Run the main regression model (M1) for a specified dependent variable.
        
        The M1 model specification includes:
        - Core explanatory variables: residual_demand, solar, wind
        - External control variables: SUN_ext, WND_ext, D_ext, Wramp
        - Fixed effects: plant ID, month, year
        - Interaction effects: plant ID × month
        
        Parameters:
        -----------
        dependent_variable : str
            The dependent variable for the regression ('generation', 'co2', or 'intensity')
            
        Returns:
        --------
        statsmodels.regression.linear_model.RegressionResults
            Fitted regression model
        """
        # Log-transform the data for the regression
        df_reg = self.log_transform_data(self.data)
        
        # Define the regression formula for M1 model
        formula = (f"{dependent_variable} ~ residual_demand + solar + wind + "
                  f"SUN_ext + WND_ext + D_ext + Wramp + "
                  f"C(id) + C(month) + C(year) + C(id)*C(month)")
        
        # Fit the model
        model = smf.ols(formula=formula, data=df_reg).fit()
        
        return model
    
    def run_all_dependent_variables(self):
        """
        Run regression models for all three dependent variables.
        
        This method runs the M1 specification for:
        1. Generation (electricity generation in MWh)
        2. CO2 emissions (total emissions in metric tons)
        3. Emissions intensity (emissions per MWh generated)
        
        Returns:
        --------
        dict
            Dictionary containing fitted models for each dependent variable
        """
        print(f"Running regression models for {self.ba_name}...")
        
        dependent_variables = ['generation', 'co2', 'intensity']
        models = {}
        
        for var in dependent_variables:
            print(f"  Running regression for {var}...")
            try:
                models[var] = self.run_regression_models(var)
                print(f"    Model fitted successfully. R² = {models[var].rsquared:.3f}")
            except Exception as e:
                print(f"    Error fitting model for {var}: {str(e)}")
                models[var] = None
        
        return models
    
    def generate_results_text(self, models):
        """
        Generate detailed text output of regression results.
        
        This method creates a comprehensive text file containing:
        - Model summaries for each dependent variable
        - Coefficient estimates with standard errors and significance levels
        - Model fit statistics (R², adjusted R², F-statistic)
        - Sample sizes and degrees of freedom
        
        Parameters:
        -----------
        models : dict
            Dictionary containing fitted models for each dependent variable
        """
        output_file = f"results/main/panel_regression_{self.ba_name}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"PANEL REGRESSION RESULTS FOR {self.ba_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write("This file contains detailed results from panel regression analyses\n")
            f.write("examining the relationship between electricity generation, CO2 emissions,\n")
            f.write("and emissions intensity with various explanatory variables.\n\n")
            
            f.write("MODEL SPECIFICATION (M1):\n")
            f.write("-" * 40 + "\n")
            f.write("Dependent Variable = α + β₁×residual_demand + β₂×solar + β₃×wind + \n")
            f.write("                     β₄×SUN_ext + β₅×WND_ext + β₆×D_ext + β₇×Wramp + \n")
            f.write("                     Plant FE + Month FE + Year FE + Plant×Month FE + ε\n\n")
            
            f.write("VARIABLE DEFINITIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("residual_demand: Net demand (demand + imports - hydro)\n")
            f.write("solar: Solar generation capacity factor\n")
            f.write("wind: Wind generation capacity factor\n")
            f.write("SUN_ext: External solar generation in neighboring areas\n")
            f.write("WND_ext: External wind generation in neighboring areas\n")
            f.write("D_ext: External electricity demand in neighboring areas\n")
            f.write("Wramp: Wind generation ramp rate\n")
            f.write("Plant FE: Plant-specific fixed effects\n")
            f.write("Month FE: Month-specific fixed effects\n")
            f.write("Year FE: Year-specific fixed effects\n")
            f.write("Plant×Month FE: Plant-month interaction fixed effects\n\n")
            
            # Results for each dependent variable
            for var_name, model in models.items():
                if model is not None:
                    f.write(f"RESULTS FOR DEPENDENT VARIABLE: {var_name.upper()}\n")
                    f.write("-" * 50 + "\n")
                    
                    # Model summary
                    f.write(f"Model Fit Statistics:\n")
                    f.write(f"  R-squared: {model.rsquared:.4f}\n")
                    f.write(f"  Adjusted R-squared: {model.rsquared_adj:.4f}\n")
                    f.write(f"  F-statistic: {model.fvalue:.2f}\n")
                    f.write(f"  Prob (F-statistic): {model.f_pvalue:.4f}\n")
                    f.write(f"  Number of observations: {model.nobs:,}\n")
                    f.write(f"  Degrees of freedom (residuals): {model.df_resid:,}\n\n")
                    
                    # Coefficient results
                    f.write("Coefficient Estimates:\n")
                    f.write("Variable".ljust(20) + "Coefficient".ljust(15) + "Std Error".ljust(15) + 
                           "t-stat".ljust(12) + "P-value".ljust(12) + "Significance\n")
                    f.write("-" * 90 + "\n")
                    
                    # Get main variables of interest
                    main_vars = ['residual_demand', 'solar', 'wind', 'SUN_ext', 'WND_ext', 'D_ext', 'Wramp']
                    
                    for var in main_vars:
                        if var in model.params.index:
                            coef = model.params[var]
                            std_err = model.bse[var]
                            t_stat = model.tvalues[var]
                            p_val = model.pvalues[var]
                            
                            # Determine significance level
                            if p_val < 0.001:
                                sig = "***"
                            elif p_val < 0.01:
                                sig = "**"
                            elif p_val < 0.05:
                                sig = "*"
                            else:
                                sig = ""
                            
                            f.write(f"{var.ljust(20)}{coef:>14.4f}{std_err:>15.4f}{t_stat:>12.3f}"
                                   f"{p_val:>12.4f}{sig:>12}\n")
                    
                    f.write("\n" + "=" * 60 + "\n\n")
                else:
                    f.write(f"ERROR: Could not fit model for {var_name}\n\n")
        
        print(f"Text results saved to {output_file}")
    
    def generate_table_csv(self, models):
        """
        Generate CSV file recreating Table 2 structure.
        
        This method creates a CSV file with the same structure as Table 2,
        containing coefficient estimates, standard errors, and significance
        levels for all three dependent variables.
        
        Parameters:
        -----------
        models : dict
            Dictionary containing fitted models for each dependent variable
        """
        output_file = f"results/main/table2_{self.ba_name}.csv"
        
        # Define the main variables of interest
        main_vars = ['residual_demand', 'solar', 'wind', 'SUN_ext', 'WND_ext', 'D_ext', 'Wramp']
        
        # Create results dataframe
        results_data = []
        
        for var in main_vars:
            row_data = {'Variable': var}
            
            for dep_var in ['generation', 'co2', 'intensity']:
                if dep_var in models and models[dep_var] is not None:
                    try:
                        coef = models[dep_var].params[var]
                        std_err = models[dep_var].bse[var]
                        p_val = models[dep_var].pvalues[var]
                        
                        # Format coefficient with significance
                        if p_val < 0.001:
                            coef_str = f"{coef:.2f}***"
                        elif p_val < 0.01:
                            coef_str = f"{coef:.2f}**"
                        elif p_val < 0.05:
                            coef_str = f"{coef:.2f}*"
                        else:
                            coef_str = f"{coef:.2f}"
                        
                        row_data[f'{dep_var}_coef'] = coef_str
                        row_data[f'{dep_var}_se'] = f"({std_err:.3f})"
                        
                    except KeyError:
                        row_data[f'{dep_var}_coef'] = "N/A"
                        row_data[f'{dep_var}_se'] = "N/A"
                else:
                    row_data[f'{dep_var}_coef'] = "N/A"
                    row_data[f'{dep_var}_se'] = "N/A"
            
            results_data.append(row_data)
        
        # Add summary statistics rows
        for dep_var in ['generation', 'co2', 'intensity']:
            if dep_var in models and models[dep_var] is not None:
                model = models[dep_var]
                results_data.append({
                    'Variable': f'{dep_var}_rsquared',
                    'generation_coef': "",
                    'generation_se': "",
                    'co2_coef': "",
                    'co2_se': "",
                    'intensity_coef': "",
                    'intensity_se': ""
                })
                
                # Set the R-squared value in the correct column
                if dep_var == 'generation':
                    results_data[-1]['generation_coef'] = f"{model.rsquared:.2f}"
                elif dep_var == 'co2':
                    results_data[-1]['co2_coef'] = f"{model.rsquared:.2f}"
                elif dep_var == 'intensity':
                    results_data[-1]['intensity_coef'] = f"{model.rsquared:.2f}"
                
                results_data.append({
                    'Variable': f'{dep_var}_obs',
                    'generation_coef': "",
                    'generation_se': "",
                    'co2_coef': "",
                    'co2_se': "",
                    'intensity_coef': "",
                    'intensity_se': ""
                })
                
                # Set the observation count in the correct column
                if dep_var == 'generation':
                    results_data[-1]['generation_coef'] = f"{model.nobs:,}"
                elif dep_var == 'co2':
                    results_data[-1]['co2_coef'] = f"{model.nobs:,}"
                elif dep_var == 'intensity':
                    results_data[-1]['intensity_coef'] = f"{model.nobs:,}"
        
        # Create and save dataframe
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_file, index=False)
        
        print(f"Table 2 CSV saved to {output_file}")
        
        # Display the results table
        print(f"\nTable 2 Results for {self.ba_name}:")
        print("=" * 80)
        
        # Create a formatted display
        display_data = []
        for _, row in results_df.iterrows():
            if 'rsquared' in row['Variable'] or 'obs' in row['Variable']:
                # Summary statistics row
                var_name = row['Variable'].replace('_rsquared', ' R²').replace('_obs', ' Obs')
                gen_val = row.get('generation_coef', '')
                co2_val = row.get('co2_coef', '')
                int_val = row.get('intensity_coef', '')
                
                display_data.append([var_name, gen_val, co2_val, int_val])
            else:
                # Coefficient row
                var_name = row['Variable']
                gen_coef = row.get('generation_coef', 'N/A')
                co2_coef = row.get('co2_coef', 'N/A')
                int_coef = row.get('intensity_coef', 'N/A')
                
                display_data.append([var_name, gen_coef, co2_coef, int_coef])
                
                # Standard error row
                gen_se = row.get('generation_se', '')
                co2_se = row.get('co2_se', '')
                int_se = row.get('intensity_se', '')
                
                if gen_se or co2_se or int_se:
                    display_data.append(['', gen_se, co2_se, int_se])
        
        # Print formatted table
        print(f"{'Variable':<20} {'Generation':<20} {'CO2 Emissions':<20} {'Emissions Intensity':<20}")
        print("-" * 80)
        for row in display_data:
            print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    
    def generate_correlation_matrix(self, models):
        """
        Generate correlation matrix for key variables and save as figure.
        
        This method creates a correlation heatmap showing the relationships
        between variables exactly as specified in the original paper:
        generation, co2, intensity, hydro, solar, wind, SUN_ext, WND_ext, D_ext, Wramp
        
        Parameters:
        -----------
        models : dict
            Dictionary containing fitted models for each dependent variable
        """
        print(f"Generating correlation matrix for {self.ba_name}...")
        print(f"  Using exact variables from original paper: generation, co2, intensity, hydro, solar, wind, SUN_ext, WND_ext, D_ext, Wramp")
        
        # Get the log-transformed data used in regressions
        df_log = self.log_transform_data(self.data)
        
        # Select key variables for correlation analysis (exact variables from original paper)
        key_vars = ['generation', 'co2', 'intensity', 'hydro', 'solar', 'wind', 'SUN_ext', 'WND_ext', 'D_ext', 'Wramp']
        available_vars = [var for var in key_vars if var in df_log.columns]
        
        if len(available_vars) < 2:
            print(f"  Warning: Insufficient variables for correlation matrix in {self.ba_name}")
            return
        
        # Calculate correlation matrix
        corr_matrix = df_log[available_vars].corr()
        
        # Create correlation heatmap (following original paper format)
        plt.figure(figsize=(10, 8))
        
        # Create heatmap with seaborn (no mask to show full matrix like original)
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',  # Use 2 decimal places like original
                   cbar_kws={"shrink": .8})
        
        plt.title(f'Pearson Correlation Coefficients - {self.ba_name}', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save figure
        output_file = f"results/SI/correlation_matrix_{self.ba_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Correlation matrix saved to {output_file}")
        
        # Save correlation matrix as CSV
        csv_file = f"results/SI/correlation_matrix_{self.ba_name}.csv"
        corr_matrix.to_csv(csv_file)
        print(f"  Correlation matrix data saved to {csv_file}")
    
    def run_statistical_tests(self, models):
        """
        Run statistical tests for model diagnostics and save results.
        
        This method performs:
        1. Durbin-Watson test for autocorrelation
        2. Breusch-Pagan test for heteroskedasticity
        3. Residual analysis by plant ID
        
        Parameters:
        -----------
        models : dict
            Dictionary containing fitted models for each dependent variable
        """
        print(f"Running statistical tests for {self.ba_name}...")
        
        output_file = f"results/SI/statistical_tests_{self.ba_name}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"STATISTICAL TESTS AND DIAGNOSTICS FOR {self.ba_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write("This file contains diagnostic tests for the panel regression models\n")
            f.write("including tests for autocorrelation, heteroskedasticity, and residual analysis.\n\n")
            
            for var_name, model in models.items():
                if model is not None:
                    f.write(f"DIAGNOSTICS FOR DEPENDENT VARIABLE: {var_name.upper()}\n")
                    f.write("-" * 50 + "\n")
                    
                    # Get residuals and data
                    residuals = model.resid
                    df_log = self.log_transform_data(self.data)
                    df_log['residuals'] = residuals
                    
                    # 1. Overall Durbin-Watson test
                    dw_stat = durbin_watson(residuals)
                    f.write(f"1. DURBIN-WATSON TEST FOR AUTOCORRELATION:\n")
                    f.write(f"   Test Statistic: {dw_stat:.4f}\n")
                    f.write(f"   Interpretation: ")
                    if dw_stat < 1.5:
                        f.write("Positive autocorrelation\n")
                    elif dw_stat > 2.5:
                        f.write("Negative autocorrelation\n")
                    else:
                        f.write("No significant autocorrelation\n")
                    f.write(f"   Note: Values close to 2 indicate no autocorrelation\n\n")
                    
                    # 2. Breusch-Pagan test for heteroskedasticity
                    try:
                        bp_test = het_breuschpagan(residuals, model.model.exog)
                        f.write(f"2. BREUSCH-PAGAN TEST FOR HETEROSKEDASTICITY:\n")
                        f.write(f"   Test Statistic: {bp_test[0]:.4f}\n")
                        f.write(f"   P-value: {bp_test[1]:.4f}\n")
                        f.write(f"   Interpretation: ")
                        if bp_test[1] < 0.05:
                            f.write("Reject null hypothesis - Heteroskedasticity present\n")
                        else:
                            f.write("Fail to reject null hypothesis - Homoskedasticity\n")
                        f.write(f"   Note: Null hypothesis is homoskedasticity\n\n")
                    except Exception as e:
                        f.write(f"2. BREUSCH-PAGAN TEST FOR HETEROSKEDASTICITY:\n")
                        f.write(f"   Error: {str(e)}\n\n")
                    
                    # 3. Plant-specific Durbin-Watson statistics
                    f.write(f"3. PLANT-SPECIFIC DURBIN-WATSON STATISTICS:\n")
                    dw_stats = {}
                    
                    for group_id, group_data in df_log.groupby('id'):
                        if len(group_data) > 1:  # Need at least 2 observations
                            try:
                                dw_stat_plant = durbin_watson(group_data['residuals'])
                                dw_stats[group_id] = dw_stat_plant
                            except:
                                continue
                    
                    if dw_stats:
                        dw_values = list(dw_stats.values())
                        f.write(f"   Number of plants analyzed: {len(dw_stats)}\n")
                        f.write(f"   Mean DW statistic: {np.mean(dw_values):.4f}\n")
                        f.write(f"   Median DW statistic: {np.median(dw_values):.4f}\n")
                        f.write(f"   Min DW statistic: {np.min(dw_values):.4f}\n")
                        f.write(f"   Max DW statistic: {np.max(dw_values):.4f}\n")
                        f.write(f"   Standard deviation: {np.std(dw_values):.4f}\n\n")
                        
                        # Save plant-specific DW statistics
                        dw_df = pd.DataFrame(list(dw_stats.items()), columns=['Plant_ID', 'DW_Statistic'])
                        dw_csv = f"results/SI/dw_stats_{self.ba_name}_{var_name}.csv"
                        dw_df.to_csv(dw_csv, index=False)
                        f.write(f"   Plant-specific DW statistics saved to: {dw_csv}\n\n")
                        
                        # Create histogram of DW statistics
                        # EXACTLY matching original notebook: figsize=(8,6), uniform bins, simple title/labels
                        plt.figure(figsize=(8, 6))
                        # Ensure uniform bin widths by calculating bin edges
                        if len(dw_values) > 0:
                            bin_edges = np.linspace(min(dw_values), max(dw_values), 11)
                            sns.histplot(dw_values, kde=True, bins=bin_edges, color='grey')
                        else:
                            # Fallback to simple bins if no data
                            sns.histplot(dw_values, kde=True, bins=10, color='grey')
                        
                        # Adding titles and labels - EXACTLY as in original notebook
                        plt.title('Histogram with KDE')
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        
                        # Save figure
                        dw_fig = f"results/SI/dw_distribution_{self.ba_name}_{var_name}.png"
                        plt.savefig(dw_fig, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        f.write(f"   DW distribution plot saved to: {dw_fig}\n\n")
                    else:
                        f.write(f"   No plant-specific DW statistics available\n\n")
                    
                    f.write("=" * 60 + "\n\n")
                else:
                    f.write(f"ERROR: Could not run diagnostics for {var_name}\n\n")
        
        print(f"  Statistical tests saved to {output_file}")
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics for key variables and save to file.
        
        This method creates descriptive statistics for variables exactly as specified
        in the original paper: generation, co2, intensity, hydro, solar, wind, 
        SUN_ext, WND_ext, D_ext, Wramp
        """
        print(f"Generating summary statistics for {self.ba_name}...")
        
        # Get log-transformed data
        df_log = self.log_transform_data(self.data)
        
        # Select key variables (exact variables from original paper)
        key_vars = ['generation', 'co2', 'intensity', 'hydro', 'solar', 'wind', 'SUN_ext', 'WND_ext', 'D_ext', 'Wramp']
        available_vars = [var for var in key_vars if var in df_log.columns]
        
        if available_vars:
            # Calculate summary statistics
            summary_stats = df_log[available_vars].describe()
            
            # Save to CSV
            output_file = f"results/SI/summary_stats_{self.ba_name}.csv"
            summary_stats.to_csv(output_file)
            
            print(f"  Summary statistics saved to {output_file}")
        else:
            print(f"  Warning: No variables available for summary statistics in {self.ba_name}")


def main():
    """
    Main function to run panel regression analysis for both CAISO and ERCOT.
    
    This function follows the exact methodology from the original paper:
    1. Initializes analyzers for both balancing authorities
    2. Loads and processes the data using original variable definitions
    3. Runs regression models for all dependent variables (generation, co2, intensity)
    4. Generates correlation matrices with exact variables from original paper
    5. Performs statistical tests following original methodology
    6. Creates outputs organized in main results and supplementary information
    """
    print("PANEL REGRESSION ANALYSIS FOR ELECTRICITY GENERATION DATA")
    print("=" * 70)
    print("This script analyzes the relationship between electricity generation,\n"
          "CO2 emissions, and emissions intensity with renewable energy sources\n"
          "and other explanatory variables for CAISO and ERCOT.\n")
    
    # Define file paths
    data_path = Path("data/processed")
    
    # CAISO analysis
    print("\n" + "="*50)
    print("ANALYZING CAISO DATA")
    print("="*50)
    
    caiso_analyzer = PanelRegressionAnalyzer(
        ba_name="CAISO",
        merged_file=data_path / "CISO_merged_D.csv",
        control_file=data_path / "CISO_control.csv"
    )
    
    caiso_analyzer.load_and_process_data()
    caiso_models = caiso_analyzer.run_all_dependent_variables()
    caiso_analyzer.generate_results_text(caiso_models)
    caiso_analyzer.generate_table_csv(caiso_models)
    caiso_analyzer.generate_correlation_matrix(caiso_models)
    caiso_analyzer.run_statistical_tests(caiso_models)
    caiso_analyzer.generate_summary_statistics()
    
    # ERCOT analysis
    print("\n" + "="*50)
    print("ANALYZING ERCOT DATA")
    print("="*50)
    
    ercot_analyzer = PanelRegressionAnalyzer(
        ba_name="ERCOT",
        merged_file=data_path / "ERCO_merged_D.csv",
        control_file=data_path / "ERCO_control.csv"
    )
    
    ercot_analyzer.load_and_process_data()
    ercot_models = ercot_analyzer.run_all_dependent_variables()
    ercot_analyzer.generate_results_text(ercot_models)
    ercot_analyzer.generate_table_csv(ercot_models)
    ercot_analyzer.generate_correlation_matrix(ercot_models)
    ercot_analyzer.run_statistical_tests(ercot_models)
    ercot_analyzer.generate_summary_statistics()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("\nMain Results (results/main/):")
    print("  - panel_regression_CAISO.txt")
    print("  - panel_regression_ERCOT.txt")
    print("  - table2_CAISO.csv")
    print("  - table2_ERCOT.csv")
    print("\nSupplementary Information (results/SI/):")
    print("  - correlation_matrix_CAISO.png/.csv")
    print("  - correlation_matrix_ERCOT.png/.csv")
    print("  - statistical_tests_CAISO.txt")
    print("  - statistical_tests_ERCOT.txt")
    print("  - dw_stats_CAISO_[var].csv")
    print("  - dw_distribution_CAISO_[var].png")
    print("  - dw_stats_ERCOT_[var].csv")
    print("  - dw_distribution_ERCOT_[var].png")
    print("  - summary_stats_CAISO.csv")
    print("  - summary_stats_ERCOT.csv")
    print("\nThe CSV files recreate Table 2 structure,")
    print("showing coefficients for generation, CO2 emissions, and emissions intensity")
    print("as dependent variables for both CAISO and ERCOT balancing authorities.")
    print("Statistical tests include Durbin-Watson autocorrelation tests and")
    print("Breusch-Pagan heteroskedasticity tests for model diagnostics.")


if __name__ == "__main__":
    main() 