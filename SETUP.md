# Quick Setup Guide for GitHub Users

## After Cloning This Repository

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Data Files
Ensure the required data files are in `data/processed/`:
- `CISO_control.csv` and `ERCO_control.csv`
- `CISO_merged_H.csv` and `ERCO_merged_H.csv` 
- `260_2023_v2.csv` and `6194_2023.csv`
- `emissions-hourly-*-combined-CAISO.csv` files
- `emissions-hourly-*-combined-ERCOT.csv` files
- `3_1_Generator_Y*.xlsx` files

### 3. Run the Analysis
```bash
# Run main analysis
python run_panel_regressions.py

# Generate all figures
python generate_figure1.py
python generate_figure2.py
python generate_figure3.py
python generate_figure4.py

# Generate SI figures
python generate_si_figures.py
python generate_si_figure11.py
```

## Important Notes

- **Data Files Required**: The analysis scripts need specific data files to run
- **File Sizes**: Some data files may be large (>100MB)
- **Processing Time**: Full analysis may take 10-30 minutes depending on the system
- **Memory**: Ensure sufficient RAM for large datasets

## Expected Outputs

After running, the following will be created:
- `results/main/` - Main regression results and figures
- `results/SI/` - Statistical tests and supplementary information  
- `results/SI_figures/` - Supplementary figures

## Troubleshooting

- **Missing Data**: Check that all CSV/Excel files are in `data/processed/`
- **Import Errors**: Ensure all packages are installed via `pip install -r requirements.txt`
- **Memory Issues**: Close other applications to free up RAM
- **File Paths**: Run scripts from the repository root directory

## Need Help?

If issues are encountered:
1. Check that all data files are present
2. Verify Python packages are installed correctly
3. Ensure running from the repository root
4. Check the main README.md for detailed documentation 