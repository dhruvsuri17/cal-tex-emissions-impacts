#!/usr/bin/env python3
"""
Generate Figure 4: Displacement Coefficients vs Capacity Factor

This script recreates Figure 4 from the research notebook, showing scatter plots
of displacement coefficients vs capacity factor with CO2 emissions intensity
as the color variable for CAISO and ERCOT plants.

The figure includes:
- Panel a: CAISO Solar coefficient vs Capacity factor
- Panel b: ERCOT Wind coefficient vs Capacity factor
- Color mapping: CO2 emissions intensity (ton per MWh)
- Size mapping: Plant capacity (MW)
- Output saved as PNG file

Author: Dhruv Suri
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === Global Styling for Scientific (Nature) Plots ===
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'grid.alpha': 0.4
})

def load_regression_data():
    """
    Load the regression results data needed for Figure 4.
    
    Returns:
    --------
    tuple
        (df_ciso_analysis_G, df_erco_analysis_G) - Generation analysis results
    """
    print("Loading regression data for Figure 4...")
    
    # Check if the combined data files exist from Figure 3
    results_dir = Path("results/main")
    
    if (results_dir / "caiso_combined_data.csv").exists() and (results_dir / "ercot_combined_data.csv").exists():
        print("  Loading existing combined data files...")
        
        # Load the combined data
        df_ciso_combined = pd.read_csv(results_dir / "caiso_combined_data.csv")
        df_ercot_combined = pd.read_csv(results_dir / "ercot_combined_data.csv")
        
        # Filter for generation analysis only
        df_ciso_analysis_G = df_ciso_combined[df_ciso_combined['dependent_var'] == 'Generation'].copy()
        df_erco_analysis_G = df_ercot_combined[df_ercot_combined['dependent_var'] == 'Generation'].copy()
        
        print(f"  Loaded {len(df_ciso_analysis_G)} CAISO plants and {len(df_erco_analysis_G)} ERCOT plants")
        
    else:
        print("  Combined data files not found. Please run generate_figure3.py first.")
        print("  This will create the required regression results.")
        return None, None
    
    return df_ciso_analysis_G, df_erco_analysis_G

def create_figure4(df_ciso_analysis_G, df_erco_analysis_G):
    """
    Create Figure 4 showing displacement coefficients vs capacity factor.
    
    Parameters:
    -----------
    df_ciso_analysis_G : pandas.DataFrame
        CAISO generation analysis results
    df_erco_analysis_G : pandas.DataFrame
        ERCOT generation analysis results
    """
    print("Creating Figure 4...")
    
    # Define global min/max CO2 emissions intensity for normalization
    global_min_CO2EI = 0
    global_max_CO2EI = 1.5

    # Create figure with gridspec arrangement
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 3, width_ratios=[5, 0.2, 0.2], height_ratios=[1, 1, 0.2])

    # Normalize color scale
    norm = Normalize(vmin=global_min_CO2EI, vmax=global_max_CO2EI)

    # === Scatter Plot: CAISO (Solar Coefficient) ===
    ax0 = fig.add_subplot(gs[0, 0])
    solar_scatter = ax0.scatter(
        df_ciso_analysis_G.CAPFAC,
        df_ciso_analysis_G.solar_coeff,
        s=df_ciso_analysis_G.NAMEPCAP / 5,
        c=df_ciso_analysis_G.CO2EI,
        cmap="viridis",
        alpha=0.7,  # Slight transparency for clarity
        edgecolor="black",
        linewidth=0.8,  # Thinner edge lines
        norm=norm
    )
    ax0.set_xlabel("Capacity factor", fontsize=18)
    ax0.set_ylabel("CAISO, Solar coefficient", fontsize=18)
    ax0.set_xlim(0, 1)
    ax0.grid(True, linestyle='--', alpha=0.6)

    # Add panel label
    ax0.text(-0.12, 1.05, 'a', transform=ax0.transAxes, fontsize=18, fontweight='bold')

    # === Scatter Plot: ERCOT (Wind Coefficient) ===
    ax1 = fig.add_subplot(gs[1, 0])
    wind_scatter = ax1.scatter(
        df_erco_analysis_G.CAPFAC,
        df_erco_analysis_G.wind_coeff,
        s=df_erco_analysis_G.NAMEPCAP / 5,
        c=df_erco_analysis_G.CO2EI,
        cmap="viridis",
        alpha=0.7,  # Consistent transparency
        edgecolor="black",
        linewidth=0.8,
        norm=norm
    )
    ax1.set_xlabel("Capacity factor", fontsize=18)
    ax1.set_ylabel("ERCOT, Wind coefficient", fontsize=18)
    ax1.set_xlim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Add panel label
    ax1.text(-0.12, 1.05, 'b', transform=ax1.transAxes, fontsize=18, fontweight='bold')

    # === Colorbar for CAISO (aligned properly) ===
    cbar_ax0 = fig.add_subplot(gs[0, 1])
    cbar0 = fig.colorbar(solar_scatter, cax=cbar_ax0, orientation='vertical')
    cbar0.set_label(r"CO$_2$ emissions intensity (ton per MWh)", fontsize=14)
    cbar0.ax.tick_params(labelsize=14)

    # === Colorbar for ERCOT (aligned properly) ===
    cbar_ax1 = fig.add_subplot(gs[1, 1])
    cbar1 = fig.colorbar(wind_scatter, cax=cbar_ax1, orientation='vertical')
    cbar1.set_label(r"CO$_2$ emissions intensity (ton per MWh)", fontsize=14)
    cbar1.ax.tick_params(labelsize=14)

    # === Plant Capacity Legend (Bottom) ===
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis('off')  # Hide axis

    # Define marker sizes and labels for legend
    sizes = [50, 100, 200, 400]
    labels = [f'{size * 5} MW' for size in sizes]

    # Create dummy scatter points for the legend
    for size, label in zip(sizes, labels):
        ax2.scatter([], [], s=size, edgecolor='black', facecolor='gray', label=label, alpha=0.6)

    # Create the legend, adjusted to avoid overlap
    ax2.legend(
        loc='center', title='Plant Capacity (MW)',
        scatterpoints=1, fontsize=14, title_fontsize=16,
        ncol=4, bbox_to_anchor=(0.5, 0.8)  # Moves legend above x-axis
    )

    # === Adjust font sizes for axis ticks ===
    ax0.tick_params(axis='both', labelsize=16)
    ax1.tick_params(axis='both', labelsize=16)

    # === Adjust spacing to prevent overlap ===
    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Adjust bottom space
    plt.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.15)  # Increased spacing for colorbars

    # === Save the figure ===
    results_dir = Path("results/main")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        results_dir / "Figure4_displacement_capacity_scatter.pdf",
        results_dir / "Figure4_displacement_capacity_scatter.png"
    ]

    for output_file in output_files:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"  Saved: {output_file}")

    # Show plot
    plt.show()

    print("Figure 4 generation complete!")

def main():
    """Main function to generate Figure 4."""
    print("=" * 70)
    print("GENERATING FIGURE 4: DISPLACEMENT COEFFICIENTS VS CAPACITY FACTOR")
    print("=" * 70)
    print("This script creates scatter plots showing the relationship between")
    print("displacement coefficients and capacity factor, with CO2 emissions")
    print("intensity as the color variable.")
    print()
    print("The figure includes:")
    print("  - Panel a: CAISO Solar coefficient vs Capacity factor")
    print("  - Panel b: ERCOT Wind coefficient vs Capacity factor")
    print("  - Color mapping: CO2 emissions intensity")
    print("  - Size mapping: Plant capacity")
    print()

    try:
        # Load regression data
        df_ciso_analysis_G, df_erco_analysis_G = load_regression_data()
        
        if df_ciso_analysis_G is None or df_erco_analysis_G is None:
            print("\nERROR: Could not load required regression data.")
            print("Please run generate_figure3.py first to create the regression results.")
            return

        # Check if required columns exist
        required_cols = ['CAPFAC', 'solar_coeff', 'wind_coeff', 'CO2EI', 'NAMEPCAP']
        missing_caiso = [col for col in required_cols if col not in df_ciso_analysis_G.columns]
        missing_ercot = [col for col in required_cols if col not in df_erco_analysis_G.columns]
        
        if missing_caiso:
            print(f"Warning: Missing columns in CAISO data: {missing_caiso}")
        if missing_ercot:
            print(f"Warning: Missing columns in ERCOT data: {missing_ercot}")
        
        if missing_caiso or missing_ercot:
            print("\nERROR: Required columns missing. Please ensure generate_figure3.py")
            print("was run successfully and includes all required plant information.")
            return

        # Generate Figure 4
        print("\n" + "=" * 50)
        print("GENERATING FIGURE 4")
        print("=" * 50)
        
        create_figure4(df_ciso_analysis_G, df_erco_analysis_G)
        
        # Save individual plant coefficient data for Figure 4
        print("\n" + "=" * 50)
        print("SAVING INDIVIDUAL PLANT DATA")
        print("=" * 50)
        
        results_dir = Path("results/main")
        
        # Save CAISO generation analysis data (used for Figure 4)
        df_ciso_analysis_G.to_csv(results_dir / "caiso_plant_coefficients_generation_figure4.csv", index=False)
        print(f"  CAISO generation coefficients saved to: {results_dir / 'caiso_plant_coefficients_generation_figure4.csv'}")
        
        # Save ERCOT generation analysis data (used for Figure 4)
        df_erco_analysis_G.to_csv(results_dir / "ercot_plant_coefficients_generation_figure4.csv", index=False)
        print(f"  ERCOT generation coefficients saved to: {results_dir / 'ercot_plant_coefficients_generation_figure4.csv'}")

        print("\n" + "=" * 70)
        print("SUCCESS: Figure 4 generated successfully!")
        print("=" * 70)
        print("Generated files:")
        print("  - results/main/Figure4_displacement_capacity_scatter.pdf")
        print("  - results/main/Figure4_displacement_capacity_scatter.png")
        print("\nIndividual plant coefficient files:")
        print("  - caiso_plant_coefficients_generation_figure4.csv")
        print("  - ercot_plant_coefficients_generation_figure4.csv")
        print("\nThe figure shows the relationship between displacement coefficients")
        print("and capacity factor, with CO2 emissions intensity as color and")
        print("plant capacity as marker size.")

    except Exception as e:
        print(f"\nERROR: Failed to generate Figure 4: {str(e)}")
        print("Please check that the regression data files exist and contain")
        print("all required columns (CAPFAC, solar_coeff, wind_coeff, CO2EI, NAMEPCAP).")
        raise

if __name__ == "__main__":
    main() 