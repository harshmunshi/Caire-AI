import numpy as np
import pandas as pd

def unnormalize_bmi(bmi):
    # Load training data to get BMI statistics
    train_data = pd.read_csv('trainingset.csv')
    bmi_mean = train_data['bmi'].mean()
    bmi_std = train_data['bmi'].std()
    
    # Unnormalize BMI using training set statistics
    unnormalized_bmi = (bmi * bmi_std) + bmi_mean
    return unnormalized_bmi

def get_least_squares_model():
    # Load the data
    data = pd.read_csv("/home/kadmin/harsh/caire/merged_data.csv")
    # Convert gender to binary (0 for Male, 1 for Female)
    data['gender_encoded'] = (data['gender'] == 'Female').astype(int)

    # Create feature matrix X with gender and facial ratios
    X = data[['gender_encoded', 'CJWR', 'WHR', 'ES', 'LF_FH', 'FW_LFH']].values

    # Add column of ones for bias term
    X = np.column_stack([np.ones(X.shape[0]), X])

    # Create target vector y with BMI values
    y = data['bmi'].values

    # Solve least squares equation: beta = (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Print coefficients
    feature_names = ['bias', 'gender', 'CJWR', 'WHR', 'ES', 'LF_FH', 'FW_LFH']
    for name, coef in zip(feature_names, beta):
        print(f"{name}: {coef:.4f}")
    
    return beta

# Inference function
def predict_bmi(features):
    # Add bias term (column of ones) to features
    features = np.column_stack([np.ones(features.shape[0]), features])
    # Make prediction using learned coefficients
    predictions = features @ beta
    return predictions


if __name__ == "__main__":
    beta = get_least_squares_model()
    # Example usage:
    # Load validation/test data
    val_data = pd.read_csv("/home/kadmin/harsh/caire/predicted_bmi.csv")
    val_data['gender_encoded'] = (val_data['gender'] == 'Female').astype(int)

    # Create feature matrix for validation data
    X_val = val_data[['gender_encoded', 'CJWR', 'WHR', 'ES', 'LF_FH', 'FW_LFH']].values

    # Get predictions
    y_val_pred = predict_bmi(X_val)

    # Save predictions
    results = pd.DataFrame({
        'filename': val_data['filename'],
        'predicted_bmi': unnormalize_bmi(y_val_pred),
        'gt_bmi': val_data['gt_bmi'],
        'gender': val_data['gender']
    })
    results.to_csv('predicted_bmi_least_squares.csv', index=False)

