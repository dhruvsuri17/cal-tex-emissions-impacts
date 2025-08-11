#!/usr/bin/env python3
"""
Generate Supplementary Information Figures

This script creates SI figures showing capacity factor vs emissions intensity
for individual power plants in CAISO and ERCOT regions.

Features:
- CAISO Natural Gas: 3 figures (82 plants)
- ERCOT Natural Gas: 3 figures (86 plants)  
- ERCOT Coal: 1 figure (11 plants)
- 8x4 subplot grids with individual plant scatter plots

Author: Dhruv Suri
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot parameters for consistent styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 17.5})
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.titlepad'] = 15

def process_files(year, fuel_type, region):
    """
    Process emissions and generator data for a specific year, fuel type, and region.
    
    This function loads and filters CEMS and 860 generator data for specific fuel types
    and regions, then merges them for analysis.
    
    Parameters:
    -----------
    year : int
        Year to process
    fuel_type : str
        Fuel type to filter ('Pipeline Natural Gas', 'Natural Gas', 'Coal')
    region : str
        Region identifier ('ca' for CAISO, 'tx' for ERCOT)
    
    Returns:
    --------
    pd.DataFrame
        Processed and merged data with capacity factor calculations
    """
    # Load CEMS emissions data
    cems_path = f"data/processed/emissions-hourly-{year}-combined-{region}"
    df = pd.read_csv(cems_path, low_memory=False)
    
    # Filter by fuel type
    if fuel_type == 'Coal':
        df = df[df['Primary Fuel Type'] == 'Coal']
    else:
        # For natural gas, include both pipeline and regular natural gas
        df = df[(df['Primary Fuel Type'] == 'Pipeline Natural Gas') | 
                (df['Primary Fuel Type'] == 'Natural Gas')]
    
    # Process datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_timedelta(df['Hour'], unit='h')
    df['Datetime'] = df['Date'] + df['Hour']
    
    # Select relevant columns
    columns = ['Facility ID', 'Datetime', 'CO2 Mass (short tons)', 'Gross Load (MW)', 'Operating Time']
    df_mod = df[columns].copy()
    
    # Convert short tons to metric tons and calculate total energy produced
    df_mod['CO2 Mass (metric tons)'] = df_mod['CO2 Mass (short tons)'] * 0.907185
    df_mod['Generation (MWh)'] = df_mod['Gross Load (MW)'] * df_mod['Operating Time']
    
    # Drop unnecessary columns
    df_mod.drop(columns=['CO2 Mass (short tons)', 'Gross Load (MW)', 'Operating Time'], inplace=True)
    
    # Calculate emissions intensity
    df_mod['Intensity (tons/MWh)'] = df_mod['CO2 Mass (metric tons)'] / df_mod['Generation (MWh)']
    
    # Replace inf with na and drop
    df_mod.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_mod.dropna(inplace=True)
    
    # Group by datetime and facility
    df_mod = df_mod.groupby(['Datetime', 'Facility ID']).sum().reset_index()
    
    # Load 860 generator data
    df860_path = f"data/processed/3_1_Generator_Y{year}.xlsx"
    df_860 = pd.read_excel(df860_path, sheet_name='Operable', skiprows=1)
    
    # Select relevant columns
    columns = ['Plant Code', 'Generator ID', 'Technology', 'Nameplate Capacity (MW)']
    df_860_mod = df_860[columns].copy()
    
    # Define technologies to include
    technologies = ['Petroleum Liquids', 'Natural Gas Steam Turbine',
                   'Conventional Steam Coal', 'Natural Gas Fired Combined Cycle',
                   'Natural Gas Fired Combustion Turbine',
                   'Natural Gas Internal Combustion Engine',
                   'Coal Integrated Gasification Combined Cycle', 'Other Gases',
                   'Petroleum Coke',
                   'Natural Gas with Compressed Air Storage',
                   'Other Natural Gas']
    
    df_860_mod = df_860_mod[df_860_mod['Technology'].isin(technologies)]
    
    # Sum nameplate capacity for each plant
    df_860_mod.drop(columns=['Technology'], inplace=True)
    df_860_grouped = df_860_mod.groupby(['Plant Code'], as_index=False).sum()
    
    # Rename Facility ID to Plant Code for merging
    df_mod.rename(columns={'Facility ID': 'Plant Code'}, inplace=True)
    
    # Merge emissions data with generator data
    df_merged = pd.merge(df_mod, df_860_grouped, on='Plant Code', how='left')
    
    # Calculate capacity factor
    df_merged['cf'] = df_merged['Generation (MWh)'] / (df_merged['Nameplate Capacity (MW)'])
    
    return df_merged

def generate_si_figure(region, fuel_type, years, output_dir):
    """
    Generate SI figure for a specific region and fuel type.
    
    Parameters:
    -----------
    region : str
        Region identifier ('ca' for CAISO, 'tx' for ERCOT)
    fuel_type : str
        Fuel type ('Natural Gas', 'Coal')
    years : list
        List of years to process
    output_dir : Path
        Directory to save output files
    """
    print(f"\nProcessing {region.upper()} {fuel_type} data...")
    
    # Process data for each year
    df_list = []
    for year in years:
        print(f"  Processing year {year}...")
        try:
            if region == 'ca':
                cems_file = f'emissions-hourly-{year}-combined-CAISO'
            else:  # tx
                cems_file = f'emissions-hourly-{year}-combined-ERCOT'
            
            df860_file = f'3_1_Generator_Y{year}'
            
            df_year = process_files(year, fuel_type, region)
            df_list.append(df_year)
            print(f"    Year {year}: {len(df_year)} records")
        except Exception as e:
            print(f"    Error processing year {year}: {e}")
            continue
    
    if not df_list:
        print(f"  No data processed for {region} {fuel_type}")
        return
    
    # Concatenate all years
    df_filtered = pd.concat(df_list, ignore_index=True)
    print(f"  Total records after concatenation: {len(df_filtered)}")
    
    # Drop rows with NaN capacity factor
    df_filtered.dropna(subset=['cf'], inplace=True)
    print(f"  Records after dropping NaN cf: {len(df_filtered)}")
    
    # Get unique plants
    unique_plants = df_filtered['Plant Code'].unique()
    num_plants = len(unique_plants)
    print(f"  Number of unique plants: {num_plants}")
    
    if num_plants == 0:
        print(f"  No plants found for {region} {fuel_type}")
        return
    
    # Calculate number of figures needed (8x4 subplots per figure)
    plants_per_figure = 8 * 4
    num_figures = -(-num_plants // plants_per_figure)  # Ceiling division
    
    print(f"  Generating {num_figures} figure(s) with {plants_per_figure} subplots each")
    
    # Generate figures
    for fig_idx in range(num_figures):
        start_idx = fig_idx * plants_per_figure
        end_idx = min((fig_idx + 1) * plants_per_figure, num_plants)
        
        # Create figure
        fig, axs = plt.subplots(8, 4, figsize=(20, 25), sharex=False, sharey=False, tight_layout=True)
        
        # Plot capacity factor vs intensity for each plant
        for i, ax in enumerate(axs.flat):
            plant_idx = start_idx + i
            if plant_idx >= num_plants:
                # Hide empty subplots
                ax.axis('off')
                continue
            
            plant = unique_plants[plant_idx]
            df_temp = df_filtered[df_filtered['Plant Code'] == plant]
            
            # Create scatter plot
            ax.scatter(df_temp['cf'], df_temp['Intensity (tons/MWh)'], s=30, color='black', alpha=0.3)
            ax.set_title(f'ORISPL {plant}')
            ax.grid(True)
        
        # Save figure
        fig_filename = f"{region}_{fuel_type.replace(' ', '_')}_figure_{fig_idx + 1}.png"
        fig_path = output_dir / fig_filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved figure {fig_idx + 1}: {fig_path}")
    
    # Save processed data
    data_filename = f"{region}_{fuel_type.replace(' ', '_')}_processed_data.csv"
    data_path = output_dir / data_filename
    df_filtered.to_csv(data_path, index=False)
    print(f"  Saved processed data: {data_path}")

def main():
    """
    Main function to generate all SI figures.
    """
    print("GENERATING SI FIGURES FOR CAISO NG, ERCOT GAS, AND ERCOT COAL")
    print("=" * 70)
    print("This script creates SI figures showing capacity factor vs emissions intensity\n"
          "for individual power plants in CAISO and ERCOT regions.\n")
    
    # Create output directory
    output_dir = Path("results/SI_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define years to process
    years = [2018, 2019, 2020, 2021, 2022]
    
    # Generate figures for each region and fuel type
    print("Processing data and generating figures...")
    
    # CAISO Natural Gas
    generate_si_figure('ca', 'Natural Gas', years, output_dir)
    
    # ERCOT Natural Gas
    generate_si_figure('tx', 'Natural Gas', years, output_dir)
    
    # ERCOT Coal
    generate_si_figure('tx', 'Coal', years, output_dir)
    
    print("\n" + "=" * 70)
    print("SI FIGURES GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    
    # List generated files
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"  - {file_path.name}")
    
    print("\nEach figure shows capacity factor vs emissions intensity scatter plots")
    print("for individual power plants, organized in 8x4 subplot grids.")
    print("Processed data CSV files are also saved for further analysis.")

if __name__ == "__main__":
    main() 