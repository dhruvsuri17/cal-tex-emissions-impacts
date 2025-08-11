#!/usr/bin/env python3
"""
Generate Figure 1: Emissions Intensity vs Capacity Factor

This script creates Figure 1 showing the relationship between CO2 emissions intensity
and capacity factor for power plants in CAISO and ERCOT regions.

Features:
- Correlation analysis between emissions intensity and capacity factor
- Summary statistics for key variables
- Output saved as PNG file

Author: Dhruv Suri
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.cm import viridis
from matplotlib.lines import Line2D
from pathlib import Path

def process_plant_data(df, nameplate_capacity):
    """
    Process plant data to calculate capacity factor and emissions intensity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with plant data
    nameplate_capacity : float
        Nameplate capacity in MW
        
    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with calculated fields
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Calculate capacity factor
    processed_df['Capacity Factor'] = processed_df['Gross Load (MW)'] / nameplate_capacity
    
    # Convert CO2 mass from short tons to metric tons (1 short ton = 0.907185 metric tons)
    processed_df['CO2 Mass (metric tons)'] = processed_df['CO2 Mass (short tons)'] * 0.907185
    
    # Calculate emissions intensity (tons CO2 per MWh)
    # Avoid division by zero
    mask = processed_df['Gross Load (MW)'] > 0
    processed_df.loc[mask, 'CO2 Emissions Intensity (ton per MWh)'] = (
        processed_df.loc[mask, 'CO2 Mass (metric tons)'] / processed_df.loc[mask, 'Gross Load (MW)']
    )
    
    # Filter out invalid values
    processed_df = processed_df[
        (processed_df['Capacity Factor'] > 0) & 
        (processed_df['CO2 Emissions Intensity (ton per MWh)'] > 0)
    ]
    
    return processed_df

def calculate_p10_capacity_factors(df, unit_ids):
    """
    Calculate P10 capacity factors for specified units.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe with capacity factor data
    unit_ids : list
        List of unit IDs to calculate P10 for
        
    Returns:
    --------
    dict
        Dictionary mapping unit IDs to P10 capacity factors
    """
    p10_factors = {}
    for unit in unit_ids:
        if unit in df['Unit ID'].unique():
            p10 = df[df['Unit ID'] == unit]['Capacity Factor'].quantile(0.1)
            p10_factors[unit] = p10
            print(f"Unit {unit} P10 capacity factor: {p10:.3f}")
    
    return p10_factors

def create_figure1():
    """
    Create Figure 1 showing CO2 emissions intensity vs capacity factor.
    
    This function creates the complete figure with two subplots:
    - Plot a: Moss Landing (CAISO, natural gas)
    - Plot b: Tolk Station (ERCOT, coal)
    """
    
    # === Data Loading and Processing ===
    print("Loading and processing plant data...")
    
    # Define file paths
    data_path = Path("data/processed")
    moss_landing_file = data_path / "260_2023_v2.csv"
    tolk_station_file = data_path / "6194_2023.csv"
    
    # Load data
    df_moss = pd.read_csv(moss_landing_file)
    df_tolk = pd.read_csv(tolk_station_file)
    
    # Define nameplate capacities
    nameplate_capacity_moss = 349.5  # MW for Moss Landing
    nameplate_capacity_tolk = 567.9  # MW for Tolk Station
    
    # Process data
    df_moss_processed = process_plant_data(df_moss, nameplate_capacity_moss)
    df_tolk_processed = process_plant_data(df_tolk, nameplate_capacity_tolk)
    
    # === Global Styling for Scientific (Nature) Plots ===
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'grid.alpha': 0.3
    })
    
    # === Color Definitions ===
    viridis_colors = ['#440154', '#30678D', '#35B778', '#FDE724']
    moss_colors = {
        '1A': viridis_colors[0],  # Dark purple
        '2A': viridis_colors[1],  # Blue
        '3A': viridis_colors[2],  # Green
        '4A': viridis_colors[3]   # Yellow
    }
    
    # === Create Figure ===
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # === Plot a: Moss Landing ===
    print("Creating Moss Landing plot...")
    ax1 = plt.subplot(gs[0, 0])
    units_moss = sorted(df_moss_processed['Unit ID'].unique())
    
    # Create scatter plots for each unit
    for unit in units_moss:
        unit_data = df_moss_processed[df_moss_processed['Unit ID'] == unit]
        ax1.scatter(unit_data['Capacity Factor'], 
                    unit_data['CO2 Emissions Intensity (ton per MWh)'],
                    s=40, c=moss_colors.get(unit, 'gray'), 
                    alpha=0.7, label=f'{unit}')
    
    # Set labels and title
    ax1.set_xlabel('Capacity factor')
    ax1.set_ylabel('CO$_2$ emissions intensity (ton per MWh)')
    ax1.set_title('Moss Landing, ORISPL 260 (CAISO, natural gas)', pad=15)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 5.0)
    
    # Remove grid and top/right spines for clean look
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Force one decimal place on y-axis
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Calculate and add P10 capacity factor lines
    p10_moss = calculate_p10_capacity_factors(df_moss_processed, ['1A', '2A', '3A', '4A'])
    for unit, p10 in p10_moss.items():
        if unit in moss_colors:
            ax1.axvline(x=p10, color=moss_colors[unit], linestyle='--', alpha=0.7)
    
    # Create custom legend
    legend_handles = [Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=moss_colors[unit], markersize=10, label=unit) 
                      for unit in sorted(units_moss) if unit in moss_colors]
    legend_handles.append(Line2D([0], [0], color='gray', linestyle='--', label='P10 capacity factor'))
    ax1.legend(handles=legend_handles, loc='upper right', framealpha=1, fontsize=12)
    
    # === Plot b: Tolk Station ===
    print("Creating Tolk Station plot...")
    ax2 = plt.subplot(gs[0, 1])
    units_tolk = sorted(df_tolk_processed['Unit ID'].unique())
    
    # Define colors for Tolk Station units
    tolk_colors = {}
    if len(units_tolk) >= 2:
        tolk_colors[units_tolk[0]] = viridis_colors[1]  # Blue for first unit
        tolk_colors[units_tolk[1]] = viridis_colors[3]  # Yellow for second unit
    
    # Create scatter plots for each unit
    for unit in units_tolk:
        unit_data = df_tolk_processed[df_tolk_processed['Unit ID'] == unit]
        ax2.scatter(unit_data['Capacity Factor'], 
                    unit_data['CO2 Emissions Intensity (ton per MWh)'],
                    s=40, c=tolk_colors.get(unit, 'gray'), 
                    alpha=0.7, label=f'{unit}')
    
    # Set labels and title
    ax2.set_xlabel('Capacity factor')
    ax2.set_ylabel('CO$_2$ emissions intensity (ton per MWh)')
    ax2.set_title('Tolk Station, ORISPL 6194 (ERCOT, coal)', pad=15)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 2.5)  # Lower y-limit for coal plant
    
    # Remove grid and top/right spines for clean look
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Force one decimal place on y-axis
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Calculate and add P10 capacity factor lines
    p10_tolk = calculate_p10_capacity_factors(df_tolk_processed, units_tolk)
    for unit, p10 in p10_tolk.items():
        if unit in tolk_colors:
            ax2.axvline(x=p10, color=tolk_colors[unit], linestyle='--', alpha=0.7)
    
    # Create custom legend for Tolk Station
    legend_handles_2 = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=tolk_colors[unit], markersize=10, label=unit) 
                        for unit in sorted(units_tolk) if unit in tolk_colors]
    legend_handles_2.append(Line2D([0], [0], color='gray', linestyle='--', label='P10 capacity factor'))
    ax2.legend(handles=legend_handles_2, loc='upper right', framealpha=1, fontsize=12)
    
    # === Panel Labels ===
    ax1.text(-0.15, 1.1, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax2.text(-0.15, 1.1, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold')
    
    # === Figure adjustments ===
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    
    # === Save Figure ===
    print("Saving Figure 1...")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/main")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    output_files = [
        results_dir / "Figure1_emissions_capacity.pdf",
        results_dir / "Figure1_emissions_capacity.png"
    ]
    
    for output_file in output_files:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"  Saved: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\nFigure 1 generation complete!")
    print("Files saved in results/main/ directory")

def main():
    """
    Main function to generate Figure 1.
    """
    print("=" * 60)
    print("GENERATING FIGURE 1: CO2 EMISSIONS INTENSITY VS CAPACITY FACTOR")
    print("=" * 60)
    print("This script creates Figure 1 showing:")
    print("  - Plot a: Moss Landing, ORISPL 260 (CAISO, natural gas)")
    print("  - Plot b: Tolk Station, ORISPL 6194 (ERCOT, coal)")
    print()
    
    try:
        create_figure1()
        print("\n" + "=" * 60)
        print("SUCCESS: Figure 1 generated successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: Failed to generate Figure 1: {str(e)}")
        print("Please check that the data files exist and are properly formatted.")
        raise

if __name__ == "__main__":
    main() 