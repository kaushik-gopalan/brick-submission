import numpy as np
import pandas as pd
import pickle
import zipfile
import os

def is_effectively_integer(x, tolerance=1e-10):
    """Check if a number is effectively an integer within a small tolerance."""
    return abs(round(x) - x) < tolerance

def calculate_integer_percentage(sequence):
    """Calculate the percentage of values in the sequence that are integers."""
    integer_count = sum(1 for x in sequence if is_effectively_integer(x))
    return (integer_count * 100) / len(sequence)

def get_quartile_number(value, boundaries):
    """Determine which quartile a value belongs to (1-4)."""
    for i in range(4):
        if i == 0:
            if boundaries[i] <= value <= boundaries[i + 1]:
                return i + 1
        else:
            if boundaries[i] < value <= boundaries[i + 1]:
                return i + 1
    return 4  # For any value exactly equal to max_val

def calculate_quartile_transitions(sequence, boundaries):
    """Calculate the number of transitions between quartiles per 100 samples."""
    if len(sequence) < 2:
        return {f'Q{i}_to_Q{j}': 0 for i in range(1, 5) for j in range(1, 5) if i != j}
        
    transitions = {f'Q{i}_to_Q{j}': 0 for i in range(1, 5) for j in range(1, 5) if i != j}
    total_samples = len(sequence)
    
    quartile_sequence = [get_quartile_number(val, boundaries) for val in sequence]
    
    for i in range(len(quartile_sequence) - 1):
        current_q = quartile_sequence[i]
        next_q = quartile_sequence[i + 1]
        if current_q != next_q:
            transitions[f'Q{current_q}_to_Q{next_q}'] += 1
    
    scale_factor = 100 / total_samples
    for key in transitions:
        transitions[key] *= scale_factor
        
    return transitions

def calculate_quartile_percentages(sequence):
    """Calculate percentage of values in each quartile."""
    if len(sequence) == 0 or np.all(sequence == sequence[0]):
        return [0, 0, 0, 0]
        
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    range_size = max_val - min_val
    
    if range_size == 0:
        return [0, 0, 0, 0]
        
    boundaries = [min_val + (range_size * i / 4) for i in range(5)]
    
    quartiles = []
    total_points = len(sequence)
    
    for i in range(4):
        if i == 0:
            mask = (sequence >= boundaries[i]) & (sequence <= boundaries[i + 1])
        else:
            mask = (sequence > boundaries[i]) & (sequence <= boundaries[i + 1])
        
        points_in_range = np.sum(mask)
        percentage = (points_in_range * 100) / total_points
        quartiles.append(percentage)
    
    return quartiles

def analyze_sequence(filename, data, labels_row=None):
    """Analyze a single sequence and return its statistics."""
    v_sequence = data['v']
    t_sequence = (data['t'] / np.timedelta64(1, 's')).astype(np.int64)

    delta_t = np.diff(t_sequence)
    min_diff_t = np.min(delta_t)
    mean_diff_t = np.mean(delta_t)
    median_diff_t = np.median(delta_t)
    max_diff_t = np.percentile(delta_t, 99)
    diff_t_30 = np.percentile(delta_t, 30)
    diff_t_70 = np.percentile(delta_t, 70)
    
    integer_percentage = calculate_integer_percentage(v_sequence)
    half_integer_percentage = calculate_integer_percentage(np.array(v_sequence)*2)
    
    
    min_val = np.min(v_sequence)
    max_val = np.max(v_sequence)
    mean_val = np.mean(v_sequence)
    median_val = np.median(v_sequence)
    
    non_zero_vals = v_sequence[v_sequence != 0]
    min_non_zero = np.min(non_zero_vals) if len(non_zero_vals) > 0 else np.nan
    
    percentile_1 = np.percentile(v_sequence, 1)
    percentile_99 = np.percentile(v_sequence, 99)
    
    min_val = np.min(v_sequence)
    max_val = np.max(v_sequence)
    range_size = max_val - min_val
    original_boundaries = [min_val + (range_size * i / 4) for i in range(5)] if range_size > 0 else [min_val] * 5
    
    original_quartiles = calculate_quartile_percentages(v_sequence)
    original_transitions = calculate_quartile_transitions(v_sequence, original_boundaries)
    
    filtered_sequence = v_sequence.copy()
    filtered_sequence[(v_sequence < percentile_1) | (v_sequence > percentile_99)] = np.nan
    filtered_sequence = filtered_sequence[~np.isnan(filtered_sequence)]
    
    if len(filtered_sequence) > 0:
        filtered_min = np.min(filtered_sequence)
        filtered_max = np.max(filtered_sequence)
        filtered_range = filtered_max - filtered_min
        filtered_boundaries = [filtered_min + (filtered_range * i / 4) for i in range(5)] if filtered_range > 0 else [filtered_min] * 5
    else:
        filtered_boundaries = [0] * 5
    
    filtered_quartiles = calculate_quartile_percentages(filtered_sequence)
    filtered_transitions = calculate_quartile_transitions(filtered_sequence, filtered_boundaries)
    
    sequence_diff = np.diff(filtered_sequence)
    total_samples = len(v_sequence)
    positive_changes = np.sum(sequence_diff > 0)
    negative_changes = np.sum(sequence_diff < 0)
    positive_changes_per_100 = (positive_changes * 100) / total_samples if total_samples > 0 else 0
    negative_changes_per_100 = (negative_changes * 100) / total_samples if total_samples > 0 else 0
    
    result_dict = {
        'filename': filename,
        'min_value': min_val,
        'min_non_zero_value': min_non_zero,
        'mean': mean_val,
        'median': median_val,
        'max_value': max_val,
        'percentile_1': percentile_1,
        'percentile_99': percentile_99,
        'integer_percentage': integer_percentage,
        'half_integer_percentage': half_integer_percentage,
        'positive_changes_per_100': positive_changes_per_100,
        'negative_changes_per_100': negative_changes_per_100,
        'original_quartile_1': original_quartiles[0],
        'original_quartile_2': original_quartiles[1],
        'original_quartile_3': original_quartiles[2],
        'original_quartile_4': original_quartiles[3],
        'filtered_quartile_1': filtered_quartiles[0],
        'filtered_quartile_2': filtered_quartiles[1],
        'filtered_quartile_3': filtered_quartiles[2],
        'filtered_quartile_4': filtered_quartiles[3],
        'min_diff_t': min_diff_t,
        'mean_diff_t': mean_diff_t,
        'median_diff_t': median_diff_t,
        'diff_t_30': diff_t_30,
        'diff_t_70': diff_t_70,
        'max_diff_t': max_diff_t
    }
    
    # Add original transitions
    for key, value in original_transitions.items():
        result_dict[f'original_{key}'] = value
        
    # Add filtered transitions
    for key, value in filtered_transitions.items():
        result_dict[f'filtered_{key}'] = value
    
    # Add label information if available (for training data)
    if labels_row is not None:
        cols_with_1 = []
        cols_with_0 = []
        for col in labels_row.index:
            if col != 'filename':
                if labels_row[col] == 1:
                    cols_with_1.append(col)
                elif labels_row[col] == 0:
                    cols_with_0.append(col)
        
        result_dict['columns_with_1'] = ', '.join(cols_with_1)
        result_dict['columns_with_0'] = ', '.join(cols_with_0)
    
    return result_dict

def process_sequences(input_zip, output_file, labels_csv=None):
    """Process sequences from a zip file and save statistics to CSV."""
    results = []
    
    # Read labels if provided (for training data)
    labels_df = pd.read_csv(labels_csv) if labels_csv else None
    
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            # Skip if not in correct folder or not a pickle file
            if not (filename.startswith('test_X/') or filename.startswith('train_X/')) or not filename.endswith('.pkl'):
                continue
            
            try:
                with zip_ref.open(filename) as file:
                    data = pickle.load(file)
                
                # Get base filename without folder path
                base_filename = os.path.basename(filename)
                
                # Get corresponding labels row if processing training data
                labels_row = None
                if labels_df is not None:
                    labels_row = labels_df[labels_df['filename'] == base_filename].iloc[0]
                
                # Analyze sequence
                stats = analyze_sequence(base_filename, data, labels_row)
                results.append(stats)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Create DataFrame and save to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Processed {len(results)} sequences")
        print(f"Results saved to {output_file}")
    else:
        print("No sequences were processed")

if __name__ == "__main__":
    # Process training data
    process_sequences(
        input_zip="../data/train_X_v0.1.0.zip",
        output_file="train_sequence_stats.csv",
        labels_csv="../data/train_y_v0.1.0.csv"
    )
    
    # Process test data
    process_sequences(
        input_zip="../data/test_X_v0.1.0.zip",
        output_file="test_sequence_stats.csv"
    )