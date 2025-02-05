import subprocess
import os
import time
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*80}")
    print(f"Running {description} ({script_name})")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)
    
    try:
        result = subprocess.run(['python', script_name], 
                              check=True,
                              capture_output=True,
                              text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        raise
    except Exception as e:
        print(f"Unexpected error running {script_name}:")
        print(str(e))
        raise

def main():
    start_time = time.time()
    
   
    try:
        # Step 1: Extract features from raw data
        run_script('extract_features.py', 'Feature Extraction')
        
        # Step 2: Train XGBoost model and generate predictions
        run_script('xgb_all.py', 'XGBoost Training and Prediction')
        
        # Step 3: Generate submission file
        run_script('gen_submission_xgb.py', 'Submission Generation')
        
        # Step 4: Post-process predictions
        run_script('postprocess.py', 'Post-processing')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*80)
        print(f"Pipeline completed successfully!")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("="*80)
        
    except Exception as e:
        print("\nPipeline failed!")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
