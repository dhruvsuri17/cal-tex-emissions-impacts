#!/usr/bin/env python3
"""
Generate Figure 2: Emissions Comparisons Across Scenarios

This script creates Figure 2 showing emissions comparisons across different regions and fuel types in a 3x2 grid.

The figure includes:
- Row 1: CAISO Gas (panels a, b)
- Row 2: ERCOT Gas (panels c, d)  
- Row 3: ERCOT Coal (panels e, f)

Each row contains:
- Left panel: Bar plot of P10, historical, and P90 emissions by year
- Right panel: Box plot of emissions intensity percentiles by year

Author: Dhruv Suri
Date: August 2025
"""

import pandas as pd
import numpy as np
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path

def process_files(file_paths, region, fuel_types):
    """
    Process files for a specific region and fuel type.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to process
    region : str
        Region identifier (e.g., 'CAISO', 'ERCOT')
    fuel_types : list
        List of fuel types to filter for
        
    Returns:
    --------
    pandas.DataFrame
        Processed summary data for all years
    """
    df_summary_all_years = pd.DataFrame()
    
    for file_path in file_paths:
        # Load the data for each year
        df = pd.read_csv(file_path)
        
        # Preprocess the data
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour'] = pd.to_timedelta(df['Hour'], unit='h')
        df['Datetime'] = df['Date'] + df['Hour']
        df['Year'] = df['Datetime'].dt.year
        
        # Filter for specified fuel type
        df = df[df['Primary Fuel Type'].isin(fuel_types)]
        
        if len(df) == 0:
            continue  # Skip if no data matches the fuel type
            
        columns = ['Facility ID', 'Unit ID', 'Datetime', 'CO2 Mass (short tons)', 'Gross Load (MW)', 'Operating Time', 'Year']
        df_mod = df[columns]
        
        # Drop 'Unit ID' and handle numeric conversions
        df_grouped = df_mod.drop(['Unit ID'], axis=1)
        df_grouped['Gross Load (MW)'] = pd.to_numeric(df_grouped['Gross Load (MW)'], errors='coerce')
        df_grouped['CO2 Mass (short tons)'] = pd.to_numeric(df_grouped['CO2 Mass (short tons)'], errors='coerce')
        
        # Group data by Facility ID and Datetime, summing emissions
        df_grouped = df_grouped.groupby(['Facility ID', 'Datetime']).sum().reset_index()
        
        # Convert CO2 Mass to metric tons and calculate EI
        df_grouped['CO2 Mass (short tons)'] = df_grouped['CO2 Mass (short tons)'] * 0.907185
        df_grouped['EI'] = df_grouped['CO2 Mass (short tons)'] / df_grouped['Gross Load (MW)']
        
        # Replace infinite and NaN values
        df_grouped.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_grouped.dropna(inplace=True)
        # drop when EI is 0
        df_grouped = df_grouped[df_grouped['EI'] != 0]
        
        # Group by Facility ID and calculate statistics
        df_summary = df_grouped.groupby('Facility ID').agg({
            'EI': ['mean', 'median', lambda x: scoreatpercentile(x, 90), lambda x: scoreatpercentile(x, 10)],
            'Gross Load (MW)': 'sum',
            'CO2 Mass (short tons)': 'sum'
        }).reset_index()

        # Rename columns
        df_summary.columns = ['Facility ID', 'Mean EI', 'Median EI', '90th Percentile EI', '10th Percentile EI', 
                             'Total Gross Load (MW)', 'Total CO2 Mass (short tons)']
        df_summary['Mean Annual EI'] = df_summary['Total CO2 Mass (short tons)'] / df_summary['Total Gross Load (MW)']
        df_summary['P90 emissions'] = df_summary['90th Percentile EI'] * df_summary['Total Gross Load (MW)']
        df_summary['P10 emissions'] = df_summary['10th Percentile EI'] * df_summary['Total Gross Load (MW)']
        
        # Add year column to summary
        df_summary['Year'] = df_grouped['Datetime'].dt.year.iloc[0]
        df_summary['Region'] = region
        df_summary['Fuel Type'] = '+'.join(fuel_types)
        
        # Append to the master DataFrame
        df_summary_all_years = pd.concat([df_summary_all_years, df_summary], ignore_index=True)
    
    return df_summary_all_years

def create_row_plots(gs_row, data, row_title, panel_label, gs):
    """
    Create a row of plots (bar + box) for a specific region and fuel type.
    
    Parameters:
    -----------
    gs_row : int
        GridSpec row index
    data : pandas.DataFrame
        Data for the specific region and fuel type
    row_title : str
        Title for the row
    panel_label : tuple
        Panel labels (e.g., ('a', 'b'))
    gs : matplotlib.gridspec.GridSpec
        GridSpec object for subplot arrangement
        
    Returns:
    --------
    tuple
        Tuple of (ax1, ax2) for the left and right subplots
    """
    ax1 = plt.subplot(gs[gs_row, 0])  # Left plot (bar)
    ax2 = plt.subplot(gs[gs_row, 1])  # Right plot (box)
    
    # Bar plot for P10 emissions, Total CO2 Mass, and P90 emissions by year
    for i, year in enumerate(years):
        df_year = data[data['Year'] == year]
        if len(df_year) == 0:
            continue
        ax1.bar(i - 0.25, df_year['P10 emissions'].sum() / 1e6, width=0.25, color=colors_viridis[0])
        ax1.bar(i, df_year['Total CO2 Mass (short tons)'].sum() / 1e6, width=0.25, color=colors_viridis[2])
        ax1.bar(i + 0.25, df_year['P90 emissions'].sum() / 1e6, width=0.25, color=colors_viridis[3])
    
    # Box plot preparation
    df_melted = pd.melt(data, 
                        id_vars=['Year', 'Region', 'Fuel Type'], 
                        value_vars=['10th Percentile EI', 'Mean Annual EI', '90th Percentile EI'],
                        var_name='EI Type', 
                        value_name='EI Value')
    
    # Add transparency to box plot colors
    lightened_colors = [
        list(colors_viridis[0][:3]) + [0.1],
        list(colors_viridis[2][:3]) + [0.1],
        list(colors_viridis[3][:3]) + [0.1],
    ]
    
    # Create box plot
    sns.boxplot(x='Year', y='EI Value', hue='EI Type', data=df_melted, 
                ax=ax2, palette=lightened_colors, showfliers=False)
    
    # Adjust transparency of patches
    for patch in ax2.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .8))
    
    for patch in ax1.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .9))
    
    # Style the plots
    ax1.set_xticks(range(len(years)))
    ax1.set_xticklabels(years)
    ax1.set_ylabel('Annual CO$_2$ emissions (Mtons)', fontsize=11)
    ax2.set_ylabel('CO$_2$ emissions intensity (tons per MWh)', fontsize=11)
    ax2.set_xlabel('')  # Remove x label from the right subplot
    
    # Set y-limits
    ax1.set_ylim(0, 110)
    ax2.set_ylim(0, 1.2)
    
    # Remove spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Adjust tick label size
    ax1.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
    # Hide legends
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    
    # Add panel labels to the left of the y axis labels
    ax1.text(-0.3, 1.05, panel_label[0], transform=ax1.transAxes, 
             fontsize=14, fontweight='bold')
    ax2.text(-0.3, 1.05, panel_label[1], transform=ax2.transAxes,
             fontsize=14, fontweight='bold')
    
    return ax1, ax2

def create_figure2():
    """
    Create Figure 2 showing emissions comparisons across regions and fuel types.
    
    This function creates the complete figure with three rows and two columns:
    - Row 1: CAISO Gas (panels a, b)
    - Row 2: ERCOT Gas (panels c, d)
    - Row 3: ERCOT Coal (panels e, f)
    """
    
    # === Data Loading and Processing ===
    print("Loading and processing emissions data...")
    
    # Define file paths - data files are in the processed directory
    data_base_path = Path("data/processed")
    
    # List of CSV file paths for each year
    # Assuming files follow naming pattern like 'emissions-hourly-YEAR-REGION.csv'
    ercot_files = list(data_base_path.glob('emissions-hourly-*-combined-ERCOT.csv'))
    caiso_files = list(data_base_path.glob('emissions-hourly-*-combined-CAISO.csv'))
    
    if not ercot_files or not caiso_files:
        print("Warning: Data files not found. Please check the data paths.")
        print(f"Looking for files in: {data_base_path}")
        print("Expected patterns:")
        print("  - emissions-hourly-*-combined-ERCOT.csv")
        print("  - emissions-hourly-*-combined-CAISO.csv")
        return
    
    print(f"Found {len(ercot_files)} ERCOT files and {len(caiso_files)} CAISO files")
    
    # Process data for each region and fuel type
    global years
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    coal_fuel = ['Coal']
    gas_fuel = ['Natural Gas', 'Pipeline Natural Gas']
    
    print("Processing CAISO gas data...")
    caiso_gas_data = process_files(caiso_files, 'CAISO', gas_fuel)
    
    print("Processing ERCOT gas data...")
    ercot_gas_data = process_files(ercot_files, 'ERCOT', gas_fuel)
    
    print("Processing ERCOT coal data...")
    ercot_coal_data = process_files(ercot_files, 'ERCOT', coal_fuel)
    
    # === Create Figure ===
    print("Creating Figure 2...")
    
    # Create a figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], 
                           left=0.12, right=0.95, top=0.95, bottom=0.1, hspace=0.5, wspace=0.3)
    
    # Viridis colors for consistency
    global colors_viridis
    colors_viridis = plt.cm.viridis(np.linspace(0, 1, 4))
    
    # Create each row
    print("  Creating Row 1: CAISO Gas...")
    ax1_caiso, ax2_caiso = create_row_plots(0, caiso_gas_data, 'CAISO Gas', ('a', 'b'), gs)
    
    print("  Creating Row 2: ERCOT Gas...")
    ax1_ercot_gas, ax2_ercot_gas = create_row_plots(1, ercot_gas_data, 'ERCOT Gas', ('c', 'd'), gs)
    
    print("  Creating Row 3: ERCOT Coal...")
    ax1_ercot_coal, ax2_ercot_coal = create_row_plots(2, ercot_coal_data, 'ERCOT Coal', ('e', 'f'), gs)
    
    # Add a common legend at the bottom
    legend_entries = [
        ('Low emissions (P10)', colors_viridis[0]),
        ('Historical (measured)', colors_viridis[2]),
        ('High emissions (P90)', colors_viridis[3])
    ]
    
    # Create square legend handles
    handles = [plt.Line2D([0], [0], marker='s', color=color, lw=0, markersize=14) 
               for label, color in legend_entries]
    labels = [label for label, color in legend_entries]
    
    # Position the legend outside the plot at the bottom, horizontal arrangement
    fig.legend(handles, labels, loc='lower center', fontsize=12, 
               frameon=False, ncol=3, bbox_to_anchor=(0.5, 0.01), labelspacing=0.5)
    
    # Tight arrangement with extra space at bottom for legend
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    # === Save Figure ===
    print("Saving Figure 2...")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/main")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    output_files = [
        results_dir / "Figure2_emissions_comparison.pdf",
        results_dir / "Figure2_emissions_comparison.png"
    ]
    
    for output_file in output_files:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"  Saved: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\nFigure 2 generation complete!")
    print("Files saved in results/main/ directory")

def main():
    """
    Main function to generate Figure 2.
    """
    print("=" * 70)
    print("GENERATING FIGURE 2: EMISSIONS COMPARISON ACROSS REGIONS AND FUEL TYPES")
    print("=" * 70)
    print("This script recreates Figure 2 from the research notebook showing:")
    print("  - Row 1: CAISO Gas (panels a, b)")
    print("  - Row 2: ERCOT Gas (panels c, d)")
    print("  - Row 3: ERCOT Coal (panels e, f)")
    print()
    print("Each row contains:")
    print("  - Left panel: Bar plot of P10, historical, and P90 emissions by year")
    print("  - Right panel: Box plot of emissions intensity percentiles by year")
    print()
    
    try:
        create_figure2()
        print("\n" + "=" * 70)
        print("SUCCESS: Figure 2 generated successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR: Failed to generate Figure 2: {str(e)}")
        print("Please check that the data files exist and are properly formatted.")
        print("Data paths in the script may need to be adjusted.")
        raise

if __name__ == "__main__":
    main() 