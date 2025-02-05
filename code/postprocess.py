import numpy as np
import pickle
import zipfile
import pandas as pd


def check_sequence_constraints(v_sequence):
    # Calculate voltage statistics in one pass
    max_v = np.max(v_sequence)
    non_zero_mask = v_sequence != 0
    min_non_zero_v = np.min(v_sequence[non_zero_mask]) if np.any(non_zero_mask) else np.inf
    mean_v = np.mean(v_sequence)
    
    # Check voltage constraints for both sensors at once
    solar_constraints_met = (
        500 < max_v < 1100 and
        -1< min_non_zero_v < 1 and
        100 < mean_v < 200
    )
    
    wd_constraints_met = (
        301 < max_v < 302.9 and
        min_non_zero_v < 50 and
        100 < mean_v < 250
    )
    
    return solar_constraints_met, wd_constraints_met, {
        'max_v': max_v,
        'min_non_zero_v': min_non_zero_v,
        'mean_v': mean_v
    }

def process_test_data(test_zip_path, submission_file_path):
    
    # Read submission file once
    df = pd.read_csv(submission_file_path)
    
    # Convert DataFrame to dictionary for faster lookup
    filename_to_idx = {filename: idx for idx, filename in enumerate(df['filename'])}
    
    # Pre-allocate numpy arrays for modifications
    solar_mask = np.zeros(len(df), dtype=bool)
    wd_mask = np.zeros(len(df), dtype=bool)
    reset_solar_mask = np.zeros(len(df), dtype=bool)
    reset_wd_mask = np.zeros(len(df), dtype=bool)
    
    sequences_checked = 0
    
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        file_list = [f for f in zip_ref.namelist() 
                    if f.startswith('test_X/') 
                    and not f.endswith('/') 
                    and f.endswith('.pkl')]
        print(f"Found {len(file_list)} files in zip")
        
        for file_path in file_list:
            filename = file_path.split('/')[-1]
            if filename not in filename_to_idx:
                continue
                
            idx = filename_to_idx[filename]
            
            with zip_ref.open(file_path, 'r') as file:
                data = pickle.load(file)
                v_sequence = data['v']
            
            sequences_checked += 1
            solar_constraints_met, wd_constraints_met, metrics = check_sequence_constraints(v_sequence)
            
            if solar_constraints_met:
                solar_mask[idx] = True
            elif df.at[idx, 'Solar_Radiance_Sensor'] == 1:
                reset_solar_mask[idx] = True
            
            if wd_constraints_met:
                wd_mask[idx] = True
            elif (df.at[idx, 'Wind_Direction_Sensor'] == 1 and 
                  df.at[idx, 'Sensor'] == 1 and 
                  df.iloc[idx].drop(['filename', 'Wind_Direction_Sensor', 'Sensor']).sum() == 0):
                reset_wd_mask[idx] = True
            
            if sequences_checked % 1000 == 0:
                print(f"Checked {sequences_checked} sequences...")
        
        # Apply all modifications in bulk
        # Reset all relevant columns for solar sensor matches
        if np.any(solar_mask):
            sensor_columns = df.columns.difference(['filename'])
            df.loc[solar_mask, sensor_columns] = 0
            df.loc[solar_mask, ['Solar_Radiance_Sensor', 'Sensor']] = 1
        
        # Reset solar sensor where constraints not met
        if np.any(reset_solar_mask):
            df.loc[reset_solar_mask, 'Solar_Radiance_Sensor'] = 0
        
        # Set wd sensor matches
        if np.any(wd_mask):
            df.loc[wd_mask, sensor_columns] = 0
            df.loc[wd_mask, ['Wind_Direction_Sensor', 'Sensor']] = 1
        
        # Reset wd sensor where constraints not met
        if np.any(reset_wd_mask):
            df.loc[reset_wd_mask, 'Wind_direction_Sensor'] = 0
        
        # Save updated DataFrame
        df.to_csv("test_predictions_pp.csv.gz", index=False)
        print(f"\nUpdated submission file saved to: test_predictions_pp.csv.gz")
        
        print(f"\nProcessing complete:")
        print(f"Total sequences checked: {sequences_checked}")
        print(f"Solar predictions modified: {np.sum(solar_mask) + np.sum(reset_solar_mask)}")
        print(f"Wind direction predictions modified: {np.sum(wd_mask) + np.sum(reset_wd_mask)}")

if __name__ == "__main__":
    process_test_data(
        test_zip_path="../data/test_X_v0.1.0.zip",
        submission_file_path="test_prediction_clean.csv.gz"
    )