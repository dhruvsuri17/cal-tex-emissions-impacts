#!/usr/bin/env python3
"""
Generate Combined Emissions Files from Raw CEMS Data

This script documents the process of creating the combined emissions files used in the analysis:
- emissions-hourly-*-combined-CAISO.csv
- emissions-hourly-*-combined-ERCOT.csv

These files are created by filtering raw CEMS emissions data to include only specific facility IDs
for CAISO and ERCOT regions, then combining data from multiple states.

IMPORTANT: This script is for documentation purposes only.
The combined files are already available in data/processed/ and should be used directly.

Author: Dhruv Suri
Date: August 2025
"""

import os
import pandas as pd
from pathlib import Path

def generate_ercot_combined_files():
    """
    Generate combined ERCOT emissions files from raw CEMS data.
    
    This function shows how the emissions-hourly-*-combined-ERCOT.csv files were created
    by filtering raw CEMS data to include only specific ERCOT facility IDs.
    
    Note: This is for documentation only - the combined files already exist.
    """
    print("ERCOT Combined Files Generation Process")
    print("=" * 50)
    
    # Define paths (these would be the original raw data locations)
    input_folder = '/Users/dhruvsuri/Code/US_emissions_impacts/data/CEMS/ERCOT'
    output_folder = '/Users/dhruvsuri/Code/US_emissions_impacts/data/fig1'
    
    # ERCOT facility IDs to include (from original analysis)
    ercot_facility_ids = [
        55501, 10298, 10418, 60264, 65535, 4939, 55311, 55168, 55327, 55172, 64383, 55053, 3561, 66561, 52176, 7762, 58378, 3460, 56806, 10184, 
        58151, 56708, 60460, 55299, 55187, 65536, 65537, 6178, 56350, 60122, 50475, 55206, 50026, 65538, 3548, 8063, 55464, 61643, 65539, 58471, 
        56233, 55223, 10261, 10436, 10692, 10554, 55480, 65540, 52120, 56152, 59145, 55226, 66595, 65541, 65542, 3490, 3464, 55086, 55153, 50118, 
        3491, 55144, 66612, 66613, 65261, 65262, 65263, 66593, 65264, 65265, 66549, 66550, 65266, 66563, 66564, 65267, 65268, 65269, 65274, 65026, 
        66562, 66594, 66565, 66567, 66566, 63335, 50043, 55313, 7097, 6181, 55230, 55052, 54817, 3452, 55097, 55365, 3439, 3609, 298, 55154, 55123, 
        65372, 6146, 65543, 55091, 57322, 3492, 3453, 65544, 62762, 60910, 3441, 3611, 6180, 55215, 50815, 66614, 54676, 50109, 55047, 3630, 3494, 60459, 
        4195, 52132, 66596, 58069, 56349, 3628, 66597, 3576, 58005, 59391, 50054, 65573, 55137, 56374, 6243, 3631, 6179, 7325, 6183, 7900, 56611, 65574, 
        10167, 50304, 50127, 3559, 3601, 62548, 59938, 55470, 65575, 4266, 55390,  65576, 3504, 55015, 4937, 3469, 57504, 60468, 58001, 55062, 55132, 52088, 
        50229, 59381, 10243, 63688, 3507, 65578, 7030, 50150, 3612, 50121, 65579, 61241, 61242, 61966, 3443, 10790, 65580, 54520, 3470, 64813, 64814, 64815, 
        64816, 64829, 64883, 64884, 64769, 64770, 64885, 64886, 64771, 64887, 64888, 64772, 64889, 64890, 64891, 64893, 64895, 64896, 64897, 64898, 64899, 64900, 
        64901, 64902, 64922, 64923, 64924, 64925, 64926, 64933, 64934, 64935, 64936, 64937, 64938, 64939, 64940, 64773, 64774, 64942, 64775, 64943, 64944, 64776, 
        64945, 64946, 64777, 64778, 64791, 64792, 64947, 64948, 64949, 64950, 64951, 64952, 64793, 64953, 64794, 64954, 64955, 64956, 64795, 64796, 64957, 64797, 
        64958, 64959, 64960, 64798, 64961, 64962, 64799, 64963, 64800, 64801, 64802, 64803, 64804, 64805, 64806, 64807, 64808, 64809, 64810, 64812, 54330, 65581, 
        56674, 55320, 55139, 59812, 54364
    ]
    
    print(f"ERCOT facility IDs to include: {len(ercot_facility_ids)}")
    print(f"States to combine: tx, ok")
    print(f"Years to process: 2018-2023")
    
    # Process each year
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    states = ['tx', 'ok']
    
    print("\nProcessing steps for each year:")
    print("1. Load emissions-hourly-{year}-{state}.csv for each state")
    print("2. Filter to include only specified facility IDs")
    print("3. Combine data from all states")
    print("4. Save as emissions-hourly-{year}-combined-ERCOT.csv")
    
    # Example of what the process would look like:
    print("\nExample code structure:")
    print("""
    for year in years:
        df_list = []
        for state in states:
            file_path = f'emissions-hourly-{year}-{state}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_list.append(df)
        
        if df_list:
            df_combined = pd.concat(df_list)
            df_filtered = df_combined[df_combined['Facility ID'].isin(ercot_facility_ids)]
            output_path = f'emissions-hourly-{year}-combined-ERCOT.csv'
            df_filtered.to_csv(output_path, index=False)
    """)

def generate_caiso_combined_files():
    """
    Generate combined CAISO emissions files from raw CEMS data.
    
    This function shows how the emissions-hourly-*-combined-CAISO.csv files were created
    by filtering raw CEMS data to include only specific CAISO facility IDs.
    
    Note: This is for documentation only - the combined files already exist.
    """
    print("\nCAISO Combined Files Generation Process")
    print("=" * 50)
    
    # Define paths (these would be the original raw data locations)
    input_folder = '/Users/dhruvsuri/Code/US_emissions_impacts/data/CEMS/CAISO'
    output_folder = '/Users/dhruvsuri/Code/US_emissions_impacts/data/fig1'
    
    # CAISO facility IDs to include (from original analysis)
    caiso_facility_ids = [
        57811, 55184, 315, 62115, 335, 62116, 356, 50748, 63699, 55951, 7450, 57564, 50299, 59800, 66456, 
        62577, 10684, 65205, 65206, 65208, 65204, 65360, 61154, 61153, 65192, 62573, 63605, 50200, 10650, 
        56474, 10649, 56346, 65203, 50170, 50622, 62176, 52096, 54296, 56090, 55295, 57580, 52147, 302, 
        57544, 10262, 55510, 55513, 55508, 55499, 10034, 65202, 65193, 65361, 57460, 57027, 10169, 57573,
        56475, 63439, 55934, 10677, 50003, 10175, 56185, 55540, 57714, 56356, 52086, 52083, 50131, 50750,
        56532, 65211, 61475, 61476, 61464, 61482, 61474, 55625, 55084, 63701, 50851, 58169, 55512, 52081,
        52104, 52082, 58122, 55333, 65200, 54449, 56026, 50493, 55935, 6211, 10776, 57001, 330, 10213,
        55950, 55400, 8076, 58168, 65210, 62671, 62704, 62702, 62703, 62705, 60223, 64756, 63704, 63705,
        63708, 55538, 63604, 63607, 50270, 10052, 55847, 62575, 57809, 10156, 56476, 61993, 63714, 7231,
        55810, 58432, 422, 54749, 55627, 56472, 10115, 56039, 55698, 50541, 55807, 55518, 57977, 50495,
        58223, 58300, 246, 63603, 55541, 60224, 63711, 58301, 66142, 66146, 66148, 62699, 58128, 62698,
        62701, 58132, 62700, 10042, 59280, 50494, 58100, 10496, 52107, 65199, 55811, 10294, 10405, 63712,
        63716, 10720, 55151, 55626, 55542, 66145, 59279, 58302, 54768, 57808, 7451, 57978, 10206, 341,
        55748, 55217, 52077, 56239, 56041, 10342, 57483, 57267, 54912, 59802, 58914, 56471, 52076, 50612,
        55393, 62459, 10501, 56639, 52169, 56473, 56232, 62697, 260, 358, 50674, 50963, 7449, 10427, 
        52078, 50850, 58607, 56914, 350, 54477, 55345, 50464, 57585, 59456, 55985, 56803, 58058, 58198,
        55656, 50849, 6704, 57555, 59803, 58085, 59457, 62458, 56184, 66458, 56134, 54936, 52109, 56143,
        55963, 58297, 58296, 58298, 56467, 54800, 50610, 50865, 50234, 66143, 50061, 10548, 66472, 7232,
        60385, 58625, 50864, 59074, 65196, 63715, 10446, 57482, 58585, 56080, 50985, 50752, 50751, 56144,
        50537, 60698, 59804, 57557, 55182, 61754, 50134, 52085, 66459, 58303, 58190, 56051, 63717, 63638,
        55933, 58619, 57122, 60120, 50064, 50089, 57584, 65191, 50115, 55851, 7436, 58053, 59458, 57515,
        57715, 50216, 54447, 65570, 10685, 54410, 55855, 58299, 59805, 52186, 10349, 55813, 55077
    ]
    
    print(f"CAISO facility IDs to include: {len(caiso_facility_ids)}")
    print(f"States to combine: ca, nv")
    print(f"Years to process: 2018-2023")
    
    # Process each year
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    states = ['ca', 'nv']
    
    print("\nProcessing steps for each year:")
    print("1. Load emissions-hourly-{year}-{state}.csv for each state")
    print("2. Filter to include only specified facility IDs")
    print("3. Combine data from all states")
    print("4. Save as emissions-hourly-{year}-combined-CAISO.csv")
    
    # Example of what the process would look like:
    print("\nExample code structure:")
    print("""
    for year in years:
        df_list = []
        for state in states:
            file_path = f'emissions-hourly-{year}-{state}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_list.append(df)
        
        if df_list:
            df_combined = pd.concat(df_list)
            df_filtered = df_combined[df_combined['Facility ID'].isin(caiso_facility_ids)]
            output_path = f'emissions-hourly-{year}-combined-CAISO.csv'
            df_filtered.to_csv(output_path, index=False)
    """)

def show_current_data_status():
    """
    Show the current status of combined emissions files in the processed data directory.
    """
    print("\nCurrent Data Status")
    print("=" * 50)
    
    processed_dir = Path("data/processed")
    
    if processed_dir.exists():
        print(f"Processed data directory: {processed_dir}")
        
        # Check for combined CAISO files
        caiso_files = list(processed_dir.glob("emissions-hourly-*-combined-CAISO.csv"))
        print(f"\nCAISO combined files found: {len(caiso_files)}")
        for file in sorted(caiso_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.1f} MB)")
        
        # Check for combined ERCOT files
        ercot_files = list(processed_dir.glob("emissions-hourly-*-combined-ERCOT.csv"))
        print(f"\nERCOT combined files found: {len(ercot_files)}")
        for file in sorted(ercot_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.1f} MB)")
        
        # Check for raw CEMS files (if they exist)
        raw_cems_files = list(processed_dir.glob("emissions-hourly-*-*.csv"))
        raw_cems_files = [f for f in raw_cems_files if "combined" not in f.name]
        if raw_cems_files:
            print(f"\nRaw CEMS files found: {len(raw_cems_files)}")
            for file in sorted(raw_cems_files)[:5]:  # Show first 5
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name} ({size_mb:.1f} MB)")
            if len(raw_cems_files) > 5:
                print(f"  ... and {len(raw_cems_files) - 5} more")
    else:
        print("Processed data directory not found")

def main():
    """
    Main function to document the combined emissions file generation process.
    """
    print("COMBINED EMISSIONS FILES GENERATION DOCUMENTATION")
    print("=" * 70)
    print("This script documents how the combined emissions files were created")
    print("from raw CEMS data for the research analysis.\n")
    print("IMPORTANT: The combined files already exist in data/processed/")
    print("This script is for documentation and reproducibility purposes only.\n")
    
    # Document the ERCOT process
    generate_ercot_combined_files()
    
    # Document the CAISO process
    generate_caiso_combined_files()
    
    # Show current data status
    show_current_data_status()
    
    print("\n" + "=" * 70)
    print("DOCUMENTATION COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("- ERCOT: Combines TX and OK state data, filters to 300+ facility IDs")
    print("- CAISO: Combines CA and NV state data, filters to 300+ facility IDs")
    print("- Years: 2018-2023 for comprehensive analysis")
    print("- Output: emissions-hourly-{year}-combined-{REGION}.csv files")
    print("\nThese combined files are used by:")
    print("- run_panel_regressions.py (main analysis)")
    print("- generate_si_figures.py (SI figures)")
    print("- Other analysis scripts")
    print("\nFor actual analysis, use the existing combined files in data/processed/")

if __name__ == "__main__":
    main() 