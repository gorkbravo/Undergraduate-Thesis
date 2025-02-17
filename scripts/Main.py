# main.py

import os
import glob
from Data_Cleaning_Engine import clean_options_data
from Bflys_engine import bfly_main
from SABR_betas_comp_engine import main as sabr_betas_comp_main
from SABR_engine import main as sabr_main
from Futures_curve import main as futures_main
# from Renaming_test import rename_files_in_folder

def main_pipeline():
    print("----- Start Pipeline -----\n")

    # Prompt for global F and T once for the options pipeline
    try:
        F = float(input("Enter forward/spot (F) for cleaning (Black-76 IV): ").strip())
        T = float(input("Enter time to expiration (years) for IV: ").strip())
    except ValueError:
        print("[ERROR] Invalid input for F or T. Exiting pipeline.")
        return

    # 1) Define separate subfolders for raw and cleaned options data
    options_raw_folder = r"C:/Users/User/Desktop/UPF/TGF/Data/OptionsF/Raw"
    options_cleaned_folder = r"C:/Users/User/Desktop/UPF/TGF/Data/OptionsF/Cleaned"

    # 2) List all .csv files in the RAW options folder
    option_files = glob.glob(os.path.join(options_raw_folder, "*.csv"))
    print("Found the following raw option CSV files:", option_files)

    # 3) Process each options file using the global F and T values
    for csv_file_collected in option_files:
        print(f"\n--- Processing Option file: {csv_file_collected} ---")

        # Build the cleaned file path in the CLEANED subfolder
        base_name = os.path.basename(csv_file_collected)  # e.g. "Opt_2025-02-14_raw1.csv"
        cleaned_file = os.path.join(
            options_cleaned_folder,
            base_name.replace(".csv", "_cleaned.csv")      # e.g. "Opt_2025-02-14_raw1_cleaned.csv"
        )

        # Clean the data (creates cleaned_file in the "Cleaned" folder)
        clean_options_data(csv_file_collected, cleaned_file, F, T)

        # 5) Butterfly engine on the cleaned file
        print("\nRunning Bfly Engine...")
        bfly_main(cleaned_file)

        # 6) SABR Betas Comparison
        print("\nRunning SABR Betas Comparison Engine...")
        sabr_betas_comp_main(cleaned_file, F, T)

        # 7) SABR Engine: Prompt for beta for each file
        print("\nRunning SABR Engine...")
        try:
            beta = float(input("Enter beta for SABR: ").strip())
        except ValueError:
            print("[ERROR] Invalid input for beta. Skipping SABR for this file.")
            continue

        sabr_main(cleaned_file, F, T, beta)

        print(f"Completed Option pipeline for {csv_file_collected}\n")

    # 8) Process futures in the same manner 
    futures_folder = r"C:/Users/User/Desktop/UPF/TGF/Data/Futures"
    futures_files = glob.glob(os.path.join(futures_folder, "*.csv"))
    print("Found the following futures CSV files:", futures_files)

    for futures_csv in futures_files:
        print(f"\n--- Processing Futures file: {futures_csv} ---")
        futures_main(futures_csv)
        print(f"Completed Futures pipeline for {futures_csv}\n")

    print("\n----- Full Pipeline Complete! -----")

if __name__ == "__main__":
    main_pipeline()
