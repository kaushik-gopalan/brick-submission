import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

# Training phase
def train_and_save_model(train_data, model_path='xgb_model.pkl', encoder_path='label_encoder.pkl'):
    print("Starting training phase...")
    
    # Get all feature columns except the specified ones
    exclude_columns = ['filename', 'columns_with_1', 'columns_with_0']
    feature_columns = [col for col in train_data.columns if col not in exclude_columns]
    
    X = train_data[feature_columns]
    y = train_data['columns_with_1']
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = np.minimum(total_samples / (len(class_counts) * class_counts),10)
    print(len(class_counts))
    # Label encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Handle categorical columns in features
    encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            encoders[column] = LabelEncoder()
            X[column] = encoders[column].fit_transform(X[column])

    # Convert to float32
    X = X.astype(np.float32)
    
    sample_weights = np.array([class_weights[val] for val in train_data['columns_with_1']])

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X, label=y,feature_weights=sample_weights)

    # Best parameters
    params = {'max_depth': 13,
        'eta': 0.02,
        'subsample': 0.92,
        'colsample_bytree': 0.92,
        'min_child_weight': 1,
        'gamma': 0.0875,
        'objective': 'multi:softmax',
        'num_class': len(class_counts),
        'eval_metric': ['mlogloss'],
     'seed':60}

    # Train model
    print("Training XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000
    )

    # Save model and encoders
    print("Saving model and encoders...")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_columns': feature_columns,
            'target_encoder': le,
            'feature_encoders': encoders
        }, f)

    print("Training phase completed.")

# Prediction phase
def predict_and_save(test_data, model_path='xgb_model.pkl', output_path='predictions_xgb.csv'):
    print("Starting prediction phase...")
    
    # Load model and encoders
    print("Loading model and encoders...")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        feature_columns = saved_data['feature_columns']
        target_encoder = saved_data['target_encoder']
        feature_encoders = saved_data['feature_encoders']

    # Prepare test data
    X_test = test_data[feature_columns].copy()

    # Apply the same encoding as training
    for column in X_test.columns:
        if column in feature_encoders:
            # Handle unseen categories gracefully
            X_test[column] = X_test[column].map(
                lambda x: -1 if x not in feature_encoders[column].classes_ 
                else feature_encoders[column].transform([x])[0]
            )

    # Convert to float32
    X_test = X_test.astype(np.float32)

    # Convert to DMatrix
    dtest = xgb.DMatrix(X_test)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(dtest)
    
    # Convert numeric predictions back to original labels
    predictions_labels = target_encoder.inverse_transform(predictions.astype(int))

    # Create output DataFrame
    output_df = pd.DataFrame({
        'filename': test_data['filename'],
        'predicted_class': predictions_labels.astype(str)
    })

    # Save predictions
    print(f"Saving predictions to {output_path}...")
    output_df.to_csv(output_path, index=False)

    print("Prediction phase completed.")
    return output_df

# Usage example:
if __name__ == "__main__":
    # Train and save model
    print("Loading training data...")
    train_data = pd.read_csv("train_sequence_stats.csv")
    train_and_save_model(train_data)
    
    # Load test data and make predictions
    print("\nLoading test data...")
    test_data = pd.read_csv("test_sequence_stats.csv")
    predictions = predict_and_save(test_data)
    
    # Print first few predictions
    print("\nFirst few predictions:")
    print(predictions.head())