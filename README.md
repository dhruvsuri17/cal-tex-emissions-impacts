# Panel Regression Analysis for Electricity Generation Data

## Quick Start (GitHub Users)

**After cloning this repository:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run main analysis
python run_panel_regressions.py

# 3. Generate all figures
python generate_figure1.py
python generate_figure2.py
python generate_figure3.py
python generate_figure4.py

# 4. Generate SI figures
python generate_si_figures.py
python generate_si_figure11.py

# 5. Check results in results/ folder
```

**Important**: The data files in `data/processed/` are required to run the analysis. See [SETUP.md](SETUP.md) for details.

---

## Citation

**Suri, D., de Chalendar, J. & Azevedo, I.M.L. Assessing the real implications for CO2 as generation from renewables increases. Nat Commun 16, 7124 (2025). https://doi.org/10.1038/s41467-025-59800-4**

## Overview

This repository contains scripts for panel regression analysis of electricity generation data, analyzing the relationship between electricity generation, CO2 emissions, and emissions intensity with renewable energy sources and other explanatory variables for CAISO and ERCOT balancing authorities.

## Purpose

The analysis examines how renewable energy integration affects thermal power plant operations and emissions, providing insights into the real-world implications of renewable energy deployment in major US electricity markets.

## Scripts

### Main Analysis
- **`run_panel_regressions.py`** - Main script for panel regression analysis
  - Runs regressions for generation, CO2 emissions, and emissions intensity
  - Generates correlation matrices with exact variables from original paper
  - Performs statistical tests (Durbin-Watson, Breusch-Pagan)
  - Creates comprehensive outputs organized in main results and supplementary information

### Figure Generation
- **`generate_figure1.py`** - Generates Figure 1: Emissions intensity vs capacity factor
  - Uses data files: `260_2023_v2.csv` and `6194_2023.csv`
  - Creates correlation plots and summary statistics
  - Output: `results/main/figure1_emissions_intensity_capacity_factor.png`

- **`generate_figure2.py`** - Generates Figure 2: Emissions comparisons across scenarios
  - Creates 3×2 grid showing emissions for different years and regions
  - Uses emissions-hourly data files
  - Output: `results/main/figure2_emissions_scenarios.png`

- **`generate_figure3.py`** - Generates Figure 3: Plant-level displacement coefficients
  - Runs plant-level regressions for generation, emissions, and intensity
  - Creates box plots of displacement coefficients
  - Output: `results/main/figure3_displacement_coefficients.png`

- **`generate_figure4.py`** - Generates Figure 4: Displacement coefficients vs capacity factor
  - Creates scatter plots using data from Figure 3
  - Shows relationship between displacement and plant efficiency
  - Output: `results/main/figure4_displacement_vs_capacity_factor.png`

### Supplementary Information (SI) Figures
- **`generate_si_figures.py`** - Generates SI figures for plant performance analysis
  - **CAISO Natural Gas**: 3 figures (82 plants) showing capacity factor vs emissions intensity
  - **ERCOT Natural Gas**: 3 figures (86 plants) showing capacity factor vs emissions intensity  
  - **ERCOT Coal**: 1 figure (11 plants) showing capacity factor vs emissions intensity
  - Each figure shows 8×4 subplot grids with individual plant scatter plots
  - Output: `results/SI_figures/` directory

- **`generate_si_figure11.py`** - Generates SI Figure 11: Emissions intensity change analysis
  - Shows change in CO2 emissions intensity under increasing renewable generation (0-100%)
  - Uses intensity coefficients from panel regression results
  - Side-by-side comparison of CAISO and ERCOT regions
  - Output: `results/SI_figures/Supplementary_Figure_11_emissions_intensity_change.png`

### Data Documentation
- **`generate_combined_emissions.py`** - Documents data preprocessing pipeline
  - Shows how combined emissions files were created from raw CEMS data
  - Documents facility ID filtering and state combination process
  - **Note**: This is for documentation only - combined files already exist

## Data Requirements

### Input Files
The scripts require the following data files in `data/processed/`:

#### Emissions Data (CEMS)
- `emissions-hourly-2018-combined-CAISO.csv` through `emissions-hourly-2023-combined-CAISO.csv`
- `emissions-hourly-2018-combined-ERCOT.csv` through `emissions-hourly-2023-combined-ERCOT.csv`

#### Generator Data (EIA-860)
- `3_1_Generator_Y2018.xlsx` through `3_1_Generator_Y2023.xlsx`

#### Control Variables
- `CISO_control.csv` - CAISO external control variables
- `ERCO_control.csv` - ERCOT external control variables

#### Additional Data
- `260_2023_v2.csv` - CAISO plant data for Figure 1
- `6194_2023.csv` - ERCOT plant data for Figure 1

## Methodology

### Panel Regression Models
The analysis uses the M1 model specification:

```
Dependent Variable = α + β₁×residual_demand + β₂×solar + β₃×wind + 
                     β₄×SUN_ext + β₅×WND_ext + β₆×D_ext + β₇×Wramp + 
                     Plant FE + Month FE + Year FE + Plant×Month FE + ε
```

Where:
- **residual_demand**: Net demand (demand + imports - hydro)
- **solar**: Solar generation capacity factor
- **wind**: Wind generation capacity factor  
- **SUN_ext**: External solar generation in neighboring areas
- **WND_ext**: External wind generation in neighboring areas
- **D_ext**: External electricity demand in neighboring areas
- **Wramp**: Wind generation ramp rate
- **Fixed Effects**: Plant, Month, Year, and Plant×Month interactions

### Statistical Approach
- **Dependent Variables**: Generation, CO2 emissions, emissions intensity
- **Transformation**: All variables log-transformed for improved statistical properties
- **Fixed Effects**: Control for unobserved heterogeneity across plants and time
- **Standard Errors**: Robust to heteroskedasticity
- **Significance Levels**: *** p<0.001, ** p<0.01, * p<0.05

### Variable Selection
The analysis uses **exact variables from the original paper**:
- `generation`, `co2`, `intensity`, `hydro`, `solar`, `wind`
- `SUN_ext`, `WND_ext`, `D_ext`, `Wramp`

## Outputs

### Main Results (`results/main/`)
- `panel_regression_CAISO.txt` - Complete regression results for CAISO
- `panel_regression_ERCOT.txt` - Complete regression results for ERCOT
- `table2_CAISO.csv` - Table 2 recreation for CAISO
- `table2_ERCOT.csv` - Table 2 recreation for ERCOT
- `figure1_emissions_intensity_capacity_factor.png` - Figure 1
- `figure2_emissions_scenarios.png` - Figure 2
- `figure3_displacement_coefficients.png` - Figure 3
- `figure4_displacement_vs_capacity_factor.png` - Figure 4

### Individual Plant Coefficient Data
- `caiso_plant_coefficients_*.csv` - Individual plant coefficients for Figure 3
- `ercot_plant_coefficients_*.csv` - Individual plant coefficients for Figure 3
- `caiso_plant_coefficients_generation_figure4.csv` - Plant coefficients for Figure 4
- `ercot_plant_coefficients_generation_figure4.csv` - Plant coefficients for Figure 4
- `caiso_combined_data.csv` - Combined data for Figure 3
- `ercot_combined_data.csv` - Combined data for Figure 3

### Supplementary Information (`results/SI/`)
- `correlation_matrix_CAISO.png/.csv` - Correlation heatmaps and data for CAISO
- `correlation_matrix_ERCOT.png/.csv` - Correlation heatmaps and data for ERCOT
- `statistical_tests_CAISO.txt` - Diagnostic tests for CAISO models
- `statistical_tests_ERCOT.txt` - Diagnostic tests for ERCOT models
- `dw_stats_CAISO_[var].csv` - Plant-specific Durbin-Watson statistics for CAISO
- `dw_distribution_CAISO_[var].png` - Distribution plots of DW statistics for CAISO
- `dw_stats_ERCOT_[var].csv` - Plant-specific Durbin-Watson statistics for ERCOT
- `dw_distribution_ERCOT_[var].png` - Distribution plots of DW statistics for ERCOT
- `summary_stats_CAISO.csv` - Descriptive statistics for CAISO variables
- `summary_stats_ERCOT.csv` - Descriptive statistics for ERCOT variables

### SI Figures (`results/SI_figures/`)
- `ca_Natural_Gas_figure_*.png` - CAISO Natural Gas plant performance (3 figures, 82 plants)
- `tx_Natural_Gas_figure_*.png` - ERCOT Natural Gas plant performance (3 figures, 86 plants)
- `tx_Coal_figure_*.png` - ERCOT Coal plant performance (1 figure, 11 plants)
- `Supplementary_Figure_11_emissions_intensity_change.png` - Side-by-side plots showing change in emissions intensity under increasing renewable generation
- `*_processed_data.csv` - Processed data files for each fuel type/region

## Usage

### Basic Workflow
1. **Run Main Analysis**: `python run_panel_regressions.py`
2. **Generate Figures**: Run individual figure scripts as needed
3. **Review Outputs**: Check `results/main/` and `results/SI/` directories

### Individual Scripts
```bash
# Main analysis
python run_panel_regressions.py

# Generate figures
python generate_figure1.py
python generate_figure2.py
python generate_figure3.py
python generate_figure4.py

# Generate SI figures
python generate_si_figures.py
python generate_si_figure11.py

# Data documentation (read-only)
python generate_combined_emissions.py
```

## Dependencies

### Required Packages
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations
- `statsmodels` - Statistical modeling and regression
- `matplotlib.pyplot` - Plotting and figure generation
- `seaborn` - Statistical data visualization
- `pathlib` - Path manipulation
- `scipy` - Statistical functions
- `openpyxl` - Excel file reading

### Installation
```bash
pip install -r requirements.txt
```

## Data Processing

### CEMS Data
- **Source**: Continuous Emissions Monitoring System hourly data
- **Processing**: Filtered by facility IDs, combined by state and year
- **Coverage**: 2018-2023 for both CAISO and ERCOT regions
- **Facilities**: 308 CAISO facilities, 282 ERCOT facilities

### Generator Data
- **Source**: EIA-860 Form data
- **Processing**: Technology filtering, capacity aggregation
- **Coverage**: Annual data 2018-2023
- **Technologies**: Natural gas, coal, and other thermal technologies

### Control Variables
- **External Generation**: Solar and wind from neighboring areas
- **External Demand**: Electricity demand in neighboring regions
- **Wind Ramp**: Rate of change in wind generation

## Key Findings

### Regional Differences
- **CAISO**: Both solar and wind show significant impacts on thermal plant operations
- **ERCOT**: Wind has stronger impact than solar on thermal plant efficiency

### Emissions Intensity
- **Solar Integration**: Increases emissions intensity in CAISO, minimal effect in ERCOT
- **Wind Integration**: Increases emissions intensity in both regions
- **Mechanism**: Reduced thermal plant capacity factors and partial-load operation

### Policy Implications
- Renewable integration strategies should consider thermal plant efficiency impacts
- Regional differences require tailored approaches to renewable deployment
- Capacity planning should account for efficiency losses in thermal generation

## Troubleshooting

### Common Issues
1. **Missing Data Files**: Ensure all required CSV and Excel files are in `data/processed/`
2. **Memory Issues**: Large datasets may require sufficient RAM
3. **File Paths**: Verify data files are in correct directories
4. **Dependencies**: Ensure all required packages are installed

### Performance Notes
- **Processing Time**: Varies by data size (typically 10-30 minutes for main analysis)
- **Memory Usage**: Peak usage depends on largest input file
- **Output Size**: PNG files can be large due to high DPI and figure dimensions

## License

This code is provided for research reproducibility purposes. Please cite the original paper when using these results in research. 