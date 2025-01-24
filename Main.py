
from Data_Collection_Engine import main as data_collection_main
from Data_Cleaning_Engine_test import clean_options_data  
from Bflys_engine_test import main as bfly_main
from SABR_betas_comp_engine import main as sabr_betas_comp_main
from SABR_engine import main as sabr_main

def main_pipeline():
    print("1) Running Data Collection Engine...")
    data_collection_main()

    print("\n2) Running Data Cleaning Engine...")
    
    clean_options_data(
       input_file="C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo.csv",
       output_file="C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo_cleaned.csv"
    )

    print("\n3) Running Butterfly (Bfly) Engine...")
    bfly_main()

    print("\n4) Running SABR Betas Comparison Engine...")
    sabr_betas_comp_main()

    print("\n5) Running SABR Engine...")    #ignore 1st outputs, pick best beta from betas_comp then run separately
    sabr_main()

    print("\nPipeline Complete!")

if __name__ == "__main__":
    main_pipeline()
