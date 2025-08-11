#!/usr/bin/env python3
"""
Generate Figure 3: Plant-Level Displacement Coefficients

This script recreates Figure 3, performing plant-level
regression analysis to calculate displacement coefficients for solar and wind energy
across different unit categories, then generating Figure 3.

The analysis includes:
1. Plant-level regressions for generation, emissions, and intensity
2. Categorization of units by type (combined cycle, combustion turbine, coal)
3. Calculation of solar and wind displacement coefficients
4. Generation of Figure 3 with box plots showing coefficient distributions

Author: Dhruv Suri
Date: August 2025
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === Global Styling for Scientific (Nature) Plots ===
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,  # Ensure black spines
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'grid.alpha': 0.4  # Keep a subtle grid for readability
})

def analyze_energy_data(primary_path, control_path, external_plant_info_path, bacode, fuel_types, reg_var):
    """
    Analyze energy data and perform plant-level regressions.
    
    Parameters:
    -----------
    primary_path : str
        Path to primary dataset
    control_path : str
        Path to control dataset
    external_plant_info_path : str
        Path to eGRID plant information
    bacode : str
        Balancing authority code
    fuel_types : list
        List of fuel types to include
    reg_var : str
        Regression variable (generation, co2, intensity)
        
    Returns:
    --------
    pandas.DataFrame
        Results with regression coefficients and plant information
    """
    print(f"Analyzing {bacode} data for {reg_var}...")
    
    # Load the primary dataset
    primary = pd.read_csv(
        primary_path,
        parse_dates=["timestamp"],
    )

    # Load the control dataset
    control = pd.read_csv(
        control_path,
        parse_dates=["Local time"],
    ).rename(columns={"Local time": "timestamp"})

    # Process and merge primary and control datasets
    control['timestamp'] = pd.to_datetime(control['timestamp'])
    primary['timestamp'] = pd.to_datetime(primary['timestamp'].astype(str).str[:-6])
    primary.set_index('timestamp', inplace=True)
    merged_data = primary.merge(control, on="timestamp", how="left")

    # Add date-related columns and drop NaNs
    merged_data["month"] = merged_data.timestamp.dt.month
    merged_data["year"] = merged_data.timestamp.dt.year
    merged_data["day"] = merged_data.timestamp.dt.day
    merged_data.dropna(inplace=True)

    # Filter out rows with 0 or negative values in specified columns
    positive_value_columns = ["solar", "wind", "SUN_ext", "WND_ext", "D_ext", "Wramp"]
    for column in positive_value_columns:
        merged_data = merged_data[merged_data[column] > 0]

    # Calculate residual demand
    merged_data["residual_demand"] = merged_data.demand + merged_data.imports - merged_data.hydro

    # Log-transform specified columns
    def log_df(df):
        df = df.replace(0, np.nan).dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        log_transform_columns = ["generation", "demand", "solar", "wind", "SUN_ext", "WND_ext", "D_ext", "Wramp", "co2", "intensity", "residual_demand"]
        for column in log_transform_columns:
            df[column] = np.log(df[column])
        return df

    # OLS regression function
    def ols_regression(df, variable, log=False):
        if log:
            df = log_df(df)
        model = smf.ols(
            formula=f"{variable} ~ residual_demand + solar + wind + SUN_ext + WND_ext + D_ext + Wramp + C(month) + C(year) + solar*C(year) + C(type)",
            data=df,
        ).fit()
        return model

    # Perform OLS regression for each plant
    plant_ols_results = []
    cols = ["id", "rsquared", "residual_demand_coeff", "residual_demand_std", "solar_coeff", "solar_std", "solar_sig", "wind_coeff", "wind_std", "wind_sig", "wramp_coeff", "wramp_std", "wramp_sig"]
    for plant_id in merged_data.id.unique():
        temp_df = merged_data[merged_data.id == plant_id]
        try:
            model_results = ols_regression(temp_df, reg_var, log=True)
            results = {
                "id": plant_id,
                "rsquared": round(model_results.rsquared, 3),
                "residual_demand_coeff": round(model_results.params['residual_demand'], 3),
                "residual_demand_std": round(model_results.bse['residual_demand'], 3),
                "solar_coeff": round(model_results.params['solar'], 3),
                "solar_std": round(model_results.bse['solar'], 3),
                "solar_sig": round(model_results.pvalues['solar'], 3),
                "wind_coeff": round(model_results.params['wind'], 3),
                "wind_std": round(model_results.bse['wind'], 3),
                "wind_sig": round(model_results.pvalues['wind'], 3),
                "wramp_coeff": round(model_results.params['Wramp'], 3),
                "wramp_std": round(model_results.bse['Wramp'], 3),
                "wramp_sig": round(model_results.pvalues['Wramp'], 3),
            }
            plant_ols_results.append(results)
        except Exception as e:
            continue

    plant_ols_df = pd.DataFrame(plant_ols_results, columns=cols)

    # Load external plant information
    df_plants_info = pd.read_excel(
        external_plant_info_path,
        skiprows=1,
        usecols=["ORISPL", "NAMEPCAP", "BACODE", "LAT", "LON", "PLPRMFL", "PLFUELCT", "CAPFAC", "CHPFLAG", "PLNGENAN", "PLNOXAN", "PLSO2AN", "PLCO2AN", "PLHTRT", "PSTATABB", "ISORTO"],
        sheet_name="PLNT20",
    )

    # Filter based on BACODE and fuel types
    df_plants_filtered = df_plants_info[(df_plants_info.BACODE == bacode) & (df_plants_info.PLFUELCT.isin(fuel_types))].rename(columns={"ORISPL": "id"})

    # Merge with OLS results
    final_df = plant_ols_df.merge(df_plants_filtered, on="id", how="inner")

    # Calculate emission intensities
    final_df["NOXEI"] = final_df["PLNOXAN"] / final_df["PLNGENAN"]
    final_df["SO2EI"] = final_df["PLSO2AN"] / final_df["PLNGENAN"]
    final_df["CO2EI"] = final_df["PLCO2AN"] / final_df["PLNGENAN"]

    print(f"  Completed {reg_var} analysis: {len(final_df)} plants")
    return final_df

def create_figure3(df_ciso_combined, df_ercot_combined):
    """
    Create Figure 3 showing displacement coefficients for solar and wind.
    
    Parameters:
    -----------
    df_ciso_combined : pandas.DataFrame
        CAISO regression results data
    df_ercot_combined : pandas.DataFrame
        ERCOT regression results data
    """
    print("Creating Figure 3...")
    
    # Define colors for different unit categories
    viridis_colors = ['#440154', '#30678D', '#35B778', '#FDE724']

    # Create figure with subplots (DO NOT SHARE Y-AXIS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # === Add panel labels ('a', 'b') ===
    ax1.text(-0.18, 1.05, 'a', transform=ax1.transAxes, fontsize=18, fontweight='bold')
    ax2.text(-0.18, 1.05, 'b', transform=ax2.transAxes, fontsize=18, fontweight='bold')

    # === First subplot: CAISO (Gas categories) ===
    sns.boxplot(x='dependent_var', y='solar_coeff', hue='Unit Category', data=df_ciso_combined, 
                ax=ax1, showfliers=False, width=0.5, linewidth=1.5, 
                palette=viridis_colors[:3], 
                hue_order=['Gas - combined cycle', 'Gas - combustion turbine', 'Gas - Other'])

    ax1.set_xticklabels(['Generation', 'Emissions', 'Intensity'], fontsize=16)
    ax1.set_title('CAISO, Solar coefficient', fontsize=18, pad=10)
    ax1.set_xlabel('')
    ax1.set_ylabel('Displacement coefficient', fontsize=18)
    ax1.axhline(0, color='black', alpha=0.8, linestyle='--', linewidth=1.5)
    ax1.tick_params(axis='y', labelsize=16)

    # Set black axis spines for a professional look
    for spine in ax1.spines.values():
        spine.set_color('black')

    # === Second subplot: ERCOT (Gas + Coal) ===
    sns.boxplot(x='dependent_var', y='wind_coeff', hue='Unit Category', data=df_ercot_combined, 
                ax=ax2, showfliers=False, width=0.5, linewidth=1.5, 
                palette=viridis_colors, 
                hue_order=['Gas - combined cycle', 'Gas - combustion turbine', 'Gas - Other', 'Coal'])

    ax2.set_xticklabels(['Generation', 'Emissions', 'Intensity'], fontsize=16)
    ax2.set_title('ERCOT, Wind coefficient', fontsize=18, pad=10)
    ax2.set_xlabel('')
    ax2.set_ylabel('Displacement coefficient', fontsize=18)
    ax2.axhline(0, color='black', alpha=0.8, linestyle='--', linewidth=1.5)
    ax2.tick_params(axis='y', labelsize=16)

    # Set black axis spines for a professional look
    for spine in ax2.spines.values():
        spine.set_color('black')

    # === Set consistent y-axis limits across both subplots ===
    y_min = -0.7
    y_max = 0.35

    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # === Remove individual legends ===
    if ax1.get_legend():
        ax1.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()

    # === Create a common legend at the bottom ===
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05),
               frameon=False, fontsize=16)

    # === Adjust figure spacing ===
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(wspace=0.3)

    # === Save the figure ===
    results_dir = Path("results/main")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        results_dir / "Figure3_displacement_coefficients.pdf",
        results_dir / "Figure3_displacement_coefficients.png"
    ]

    for output_file in output_files:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"  Saved: {output_file}")

    # Show plot
    plt.show()

    print("Figure 3 generation complete!")

def main():
    """Main function to run the complete analysis."""
    print("=" * 70)
    print("PLANT-LEVEL REGRESSION ANALYSIS AND FIGURE 3 GENERATION")
    print("=" * 70)
    print("This script performs plant-level regressions to calculate displacement")
    print("coefficients for solar and wind energy, then generates Figure 3.")
    print()

    # Define file paths
    data_path = Path("data/processed")
    
    try:
        # === Load and process unit information ===
        print("=" * 50)
        print("LOADING UNIT INFORMATION")
        print("=" * 50)
        
        # Load CAISO units
        units_ciso = pd.read_csv(data_path / 'largest_units_ciso.csv')
        units_ciso['Unit Type'] = units_ciso['Unit Type'].replace(to_replace=r'^Combustion turbine.*', 
                                                                  value='Combustion turbine', regex=True)
        units_ciso.rename(columns={'Facility ID': 'id'}, inplace=True)
        
        # Load ERCOT units
        units_erco = pd.read_csv(data_path / 'largest_units_ercot.csv')
        units_erco['Unit Type'] = units_erco['Unit Type'].replace(to_replace=r'^Combustion turbine.*', 
                                                                  value='Combustion turbine', regex=True)
        units_erco.rename(columns={'Facility ID': 'id'}, inplace=True)
        
        print(f"Loaded {len(units_ciso)} CAISO units and {len(units_erco)} ERCOT units")

        # === CAISO Analysis ===
        print("\n" + "=" * 50)
        print("ANALYZING CAISO DATA")
        print("=" * 50)
        
        df_ciso_analysis_G = analyze_energy_data(
            data_path / 'CISO_merged_H.csv',
            data_path / 'CISO_control.csv',
            data_path / 'eGRID2020_Data_v2.xlsx',
            'CISO',
            ['GAS', 'OIL', 'COAL'],
            "generation"
        )

        df_ciso_analysis_C = analyze_energy_data(
            data_path / 'CISO_merged_H.csv',
            data_path / 'CISO_control.csv',
            data_path / 'eGRID2020_Data_v2.xlsx',
            'CISO',
            ['GAS', 'OIL', 'COAL'],
            "co2"
        )

        df_ciso_analysis_EI = analyze_energy_data(
            data_path / 'CISO_merged_H.csv',
            data_path / 'CISO_control.csv',
            data_path / 'eGRID2020_Data_v2.xlsx',
            'CISO',
            ['GAS', 'OIL', 'COAL'],
            "intensity"
        )

        # Merge with unit information
        df_ciso_analysis_G = pd.merge(df_ciso_analysis_G, units_ciso, how='left', on='id')
        df_ciso_analysis_C = pd.merge(df_ciso_analysis_C, units_ciso, how='left', on='id')
        df_ciso_analysis_EI = pd.merge(df_ciso_analysis_EI, units_ciso, how='left', on='id')

        # === ERCOT Analysis ===
        print("\n" + "=" * 50)
        print("ANALYZING ERCOT DATA")
        print("=" * 50)
        
        df_erco_analysis_G = analyze_energy_data(
            data_path / 'ERCO_merged_H.csv',
            data_path / 'ERCO_control.csv',
            data_path / 'eGRID2020_Data_v2.xlsx',
            'ERCO',
            ['GAS', 'OIL', 'COAL'],
            "generation"
        )

        df_erco_analysis_C = analyze_energy_data(
            data_path / 'ERCO_merged_H.csv',
            data_path / 'ERCO_control.csv',
            data_path / 'eGRID2020_Data_v2.xlsx',
            'ERCO',
            ['GAS', 'OIL', 'COAL'],
            "co2"
        )

        df_erco_analysis_EI = analyze_energy_data(
            data_path / 'ERCO_merged_H.csv',
            data_path / 'ERCO_control.csv',
            data_path / 'eGRID2020_Data_v2.xlsx',
            'ERCO',
            ['GAS', 'OIL', 'COAL'],
            "intensity"
        )

        # Merge with unit information
        df_erco_analysis_G = pd.merge(df_erco_analysis_G, units_erco, how='left', on='id')
        df_erco_analysis_C = pd.merge(df_erco_analysis_C, units_erco, how='left', on='id')
        df_erco_analysis_EI = pd.merge(df_erco_analysis_EI, units_erco, how='left', on='id')

        # === Prepare data for Figure 3 ===
        print("\n" + "=" * 50)
        print("PREPARING FIGURE DATA")
        print("=" * 50)
        
        # Step 1: Create a combined dataframe for CISO
        df_ciso_analysis_G['dependent_var'] = 'Generation'
        df_ciso_analysis_C['dependent_var'] = 'Emissions'
        df_ciso_analysis_EI['dependent_var'] = 'Intensity'

        # Concatenate the CISO dataframes
        df_ciso_combined = pd.concat([df_ciso_analysis_G, df_ciso_analysis_C, df_ciso_analysis_EI])

        # Modify 'Unit Type' column for Gas categories
        df_ciso_combined['Unit Category'] = df_ciso_combined.apply(
            lambda x: x['Unit Type'] if x['Unit Type'] in ['Combined cycle', 'Combustion turbine']
            else 'Gas - Other', axis=1)
        
        # Rename for consistency with original
        df_ciso_combined['Unit Category'] = df_ciso_combined['Unit Category'].replace({
            'Combined cycle': 'Gas - combined cycle',
            'Combustion turbine': 'Gas - combustion turbine'
        })

        # Step 2: Create a combined dataframe for ERCO
        df_erco_analysis_G['dependent_var'] = 'Generation'
        df_erco_analysis_C['dependent_var'] = 'Emissions'
        df_erco_analysis_EI['dependent_var'] = 'Intensity'

        # Concatenate the ERCO dataframes
        df_ercot_combined = pd.concat([df_erco_analysis_G, df_erco_analysis_C, df_erco_analysis_EI])

        # Modify 'Unit Category' for ERCO
        df_ercot_combined['Unit Category'] = df_ercot_combined.apply(
            lambda x: 'Gas - combined cycle' if x['PLFUELCT'] == 'GAS' and x['Unit Type'] == 'Combined cycle'
            else 'Gas - combustion turbine' if x['PLFUELCT'] == 'GAS' and x['Unit Type'] == 'Combustion turbine'
            else 'Gas - Other' if x['PLFUELCT'] == 'GAS'
            else 'Coal', axis=1)

        print(f"CAISO combined data: {len(df_ciso_combined)} observations")
        print(f"ERCOT combined data: {len(df_ercot_combined)} observations")

        # === Generate Figure 3 ===
        print("\n" + "=" * 50)
        print("GENERATING FIGURE 3")
        print("=" * 50)
        
        create_figure3(df_ciso_combined, df_ercot_combined)

        # === Save Results ===
        print("\n" + "=" * 50)
        print("SAVING RESULTS")
        print("=" * 50)
        
        results_dir = Path("results/main")
        
        # Save combined data for reference
        df_ciso_combined.to_csv(results_dir / "caiso_combined_data.csv", index=False)
        df_ercot_combined.to_csv(results_dir / "ercot_combined_data.csv", index=False)
        
        # Save individual plant coefficients for each dependent variable
        print("\nSaving individual plant coefficient data...")
        
        # CAISO individual coefficients
        for dep_var, df in [('Generation', df_ciso_analysis_G), ('Emissions', df_ciso_analysis_C), ('Intensity', df_ciso_analysis_EI)]:
            filename = f"caiso_plant_coefficients_{dep_var.lower()}.csv"
            df.to_csv(results_dir / filename, index=False)
            print(f"  CAISO {dep_var} coefficients: {results_dir / filename}")
        
        # ERCOT individual coefficients
        for dep_var, df in [('Generation', df_erco_analysis_G), ('Emissions', df_erco_analysis_C), ('Intensity', df_erco_analysis_EI)]:
            filename = f"ercot_plant_coefficients_{dep_var.lower()}.csv"
            df.to_csv(results_dir / filename, index=False)
            print(f"  ERCOT {dep_var} coefficients: {results_dir / filename}")
        
        print(f"\n  CAISO combined data saved to: {results_dir / 'caiso_combined_data.csv'}")
        print(f"  ERCOT combined data saved to: {results_dir / 'ercot_combined_data.csv'}")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("Generated files:")
        print(f"  - {results_dir / 'Figure3_displacement_coefficients.pdf'}")
        print(f"  - {results_dir / 'Figure3_displacement_coefficients.png'}")
        print(f"  - {results_dir / 'caiso_combined_data.csv'}")
        print(f"  - {results_dir / 'ercot_combined_data.csv'}")
        print("\nIndividual plant coefficient files:")
        print("  - caiso_plant_coefficients_generation.csv")
        print("  - caiso_plant_coefficients_emissions.csv")
        print("  - caiso_plant_coefficients_intensity.csv")
        print("  - ercot_plant_coefficients_generation.csv")
        print("  - ercot_plant_coefficients_emissions.csv")
        print("  - ercot_plant_coefficients_intensity.csv")

    except Exception as e:
        print(f"\nERROR: Analysis failed: {str(e)}")
        print("Please check that all required data files exist and are properly formatted.")
        raise

if __name__ == "__main__":
    main() 