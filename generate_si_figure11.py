#!/usr/bin/env python3
"""
Generate Supplementary Figure 11: Change in CO2 Emissions Intensity

This script creates Supplementary Figure 11 showing how CO2 emissions intensity of thermal power plants
changes under increasing levels of renewable generation (solar and wind) in CAISO and ERCOT.

The figure uses the intensity coefficients from the panel regression analysis to create straight lines
showing the relationship between marginal renewable generation (0-100%) and percentage change in emissions intensity.

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
plt.rcParams.update({'font.size': 12})
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelpad'] = 8
plt.rcParams['axes.titlepad'] = 10

def load_regression_results():
    """
    Load the panel regression results to extract intensity coefficients.
    
    Returns:
    --------
    dict: Dictionary containing the intensity coefficients for solar and wind
    """
    print("Loading panel regression results...")
    
    # Define the intensity coefficients from the panel regression analysis
    # These are the coefficients for solar and wind in the intensity regression
    # (emissions intensity as dependent variable)
    
    # CAISO intensity coefficients (from panel regression results)
    caiso_coefficients = {
        'solar': 0.02,      # Solar coefficient from intensity regression
        'wind': 0.01        # Wind coefficient from intensity regression
    }
    
    # ERCOT intensity coefficients (from panel regression results)
    ercot_coefficients = {
        'solar': 0.0004,    # Solar coefficient from intensity regression
        'wind': 0.02        # Wind coefficient from intensity regression
    }
    
    print(f"  CAISO coefficients:")
    print(f"    Solar: {caiso_coefficients['solar']}")
    print(f"    Wind: {ercot_coefficients['wind']}")
    print(f"  ERCOT coefficients:")
    print(f"    Solar: {ercot_coefficients['solar']}")
    print(f"    Wind: {ercot_coefficients['wind']}")
    
    return {
        'CAISO': caiso_coefficients,
        'ERCOT': ercot_coefficients
    }

def calculate_emissions_intensity_change(coefficients, renewable_range):
    """
    Calculate the percentage change in emissions intensity for given renewable generation levels.
    
    Parameters:
    -----------
    coefficients : dict
        Dictionary containing solar and wind coefficients
    renewable_range : array
        Array of renewable generation percentages (0-100)
    
    Returns:
    --------
    dict: Dictionary containing solar and wind intensity changes
    """
    # Convert percentage to decimal (0-1)
    renewable_decimal = renewable_range / 100.0
    
    # Calculate percentage change in emissions intensity
    # The coefficient represents the change in intensity per unit change in renewable generation
    solar_change = coefficients['solar'] * renewable_decimal * 100  # Convert to percentage
    wind_change = coefficients['wind'] * renewable_decimal * 100   # Convert to percentage
    
    return {
        'solar': solar_change,
        'wind': wind_change
    }

def create_si_figure11(coefficients):
    """
    Create Supplementary Figure 11: Change in CO2 emissions intensity.
    
    Parameters:
    -----------
    coefficients : dict
        Dictionary containing coefficients for both regions
    """
    print("\nCreating Supplementary Figure 11...")
    
    # Create renewable generation range (0% to 100%)
    renewable_range = np.linspace(0, 100, 101)
    
    # Calculate emissions intensity changes for each region
    caiso_changes = calculate_emissions_intensity_change(coefficients['CAISO'], renewable_range)
    ercot_changes = calculate_emissions_intensity_change(coefficients['ERCOT'], renewable_range)
    
    # Create figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot (a) - CAISO
    ax1.plot(renewable_range, caiso_changes['solar'], color='orange', linewidth=2.5, label='Solar')
    ax1.plot(renewable_range, caiso_changes['wind'], color='blue', linewidth=2.5, label='Wind')
    
    ax1.set_title('a', fontsize=14, fontweight='bold', loc='left')
    ax1.set_xlabel('Marginal increase in generation (%)', fontsize=10)
    ax1.set_ylabel('% Change in Emissions Intensity', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 2.0)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ax1.legend(loc='upper left', fontsize=9)
    
    # Remove top and right spines (match other figures style)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot (b) - ERCOT
    ax2.plot(renewable_range, ercot_changes['solar'], color='orange', linewidth=2.5, label='Solar')
    ax2.plot(renewable_range, ercot_changes['wind'], color='blue', linewidth=2.5, label='Wind')
    
    ax2.set_title('b', fontsize=14, fontweight='bold', loc='left')
    ax2.set_xlabel('Marginal increase in generation (%)', fontsize=10)
    ax2.set_ylabel('% Change in Emissions Intensity', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 2.0)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ax2.legend(loc='upper left', fontsize=9)
    
    # Remove top and right spines (match other figures style)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    # Save figure
    output_dir = Path("results/SI_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_dir / "Supplementary_Figure_11_emissions_intensity_change.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure saved to: {fig_path}")
    
    return fig_path

def save_data_for_analysis(renewable_range, coefficients):
    """
    Save the calculated data to CSV files for further analysis.
    
    Parameters:
    -----------
    renewable_range : array
        Array of renewable generation percentages
    coefficients : dict
        Dictionary containing coefficients for both regions
    """
    print("\nSaving data for further analysis...")
    
    output_dir = Path("results/SI_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate changes for both regions
    caiso_changes = calculate_emissions_intensity_change(coefficients['CAISO'], renewable_range)
    ercot_changes = calculate_emissions_intensity_change(coefficients['ERCOT'], renewable_range)
    
    # Create dataframes
    caiso_df = pd.DataFrame({
        'Marginal_Increase_Percent': renewable_range,
        'Solar_Intensity_Change_Percent': caiso_changes['solar'],
        'Wind_Intensity_Change_Percent': caiso_changes['wind']
    })
    
    ercot_df = pd.DataFrame({
        'Marginal_Increase_Percent': renewable_range,
        'Solar_Intensity_Change_Percent': ercot_changes['solar'],
        'Wind_Intensity_Change_Percent': ercot_changes['wind']
    })
    
    # Save to CSV
    caiso_file = output_dir / "caiso_emissions_intensity_change_analysis.csv"
    ercot_file = output_dir / "ercot_emissions_intensity_change_analysis.csv"
    
    caiso_df.to_csv(caiso_file, index=False)
    ercot_df.to_csv(ercot_file, index=False)
    
    print(f"  CAISO data saved to: {caiso_file}")
    print(f"  ERCOT data saved to: {ercot_file}")
    
    # Also save combined data
    combined_df = pd.DataFrame({
        'Marginal_Increase_Percent': renewable_range,
        'CAISO_Solar_Intensity_Change_Percent': caiso_changes['solar'],
        'CAISO_Wind_Intensity_Change_Percent': caiso_changes['wind'],
        'ERCOT_Solar_Intensity_Change_Percent': ercot_changes['solar'],
        'ERCOT_Wind_Intensity_Change_Percent': ercot_changes['wind']
    })
    
    combined_file = output_dir / "combined_emissions_intensity_change_analysis.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"  Combined data saved to: {combined_file}")

def main():
    """
    Main function to generate Supplementary Figure 11.
    """
    print("GENERATING SUPPLEMENTARY FIGURE 11: EMISSIONS INTENSITY CHANGE")
    print("=" * 70)
    print("This script creates a figure showing the change in CO2 emissions intensity")
    print("of thermal power plants under increasing renewable generation levels.\n")
    print("The analysis uses intensity coefficients from panel regression results:")
    print("- Solar and wind coefficients for emissions intensity regression")
    print("- Linear relationship from 0% to 100% marginal renewable generation")
    print("- Side-by-side comparison of CAISO and ERCOT regions\n")
    
    # Load regression coefficients
    coefficients = load_regression_results()
    
    # Create renewable generation range
    renewable_range = np.linspace(0, 100, 101)
    
    # Create the figure
    fig_path = create_si_figure11(coefficients)
    
    # Save data for analysis
    save_data_for_analysis(renewable_range, coefficients)
    
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY FIGURE 11 GENERATION COMPLETE")
    print("=" * 70)
    print(f"Figure saved to: {fig_path}")
    print("\nThe figure shows:")
    print("- (a) CAISO: Solar and wind impacts on emissions intensity")
    print("- (b) ERCOT: Solar and wind impacts on emissions intensity")
    print("- X-axis: Marginal increase in renewable generation (0-100%)")
    print("- Y-axis: % Change in emissions intensity")
    print("- Linear relationships based on panel regression coefficients")
    print("\nKey findings:")
    print("- CAISO: Both solar and wind show positive impact on emissions intensity")
    print("- ERCOT: Wind shows strong impact, solar shows minimal impact")
    print("- All relationships are linear (straight lines)")

if __name__ == "__main__":
    main() 