import torch
import pandas as pd
import cv2
import numpy as np
from shallow_model import ShallowModel
import face_alignment
from skimage import io
import os

def unnormalize_bmi(bmi):
    # Load training data to get BMI statistics
    train_data = pd.read_csv('trainingset.csv')
    bmi_mean = train_data['bmi'].mean()
    bmi_std = train_data['bmi'].std()
    
    # Unnormalize BMI using training set statistics
    unnormalized_bmi = (bmi * bmi_std) + bmi_mean
    return unnormalized_bmi

def compute_facial_ratios(landmarks):
    # Extract key points from face_alignment landmarks
    # Note: face_alignment uses different indices than MediaPipe
    chin = landmarks[8]  # Chin point
    left_jaw = landmarks[3]  # Left jaw
    right_jaw = landmarks[13]  # Right jaw
    left_eye = landmarks[36]  # Left eye outer corner
    right_eye = landmarks[45]  # Right eye outer corner
    nose = landmarks[30]  # Nose tip
    left_face = landmarks[0]  # Left face width point 
    right_face = landmarks[16]  # Right face width point
    
    # Calculate distances
    jaw_width = np.linalg.norm(left_jaw - right_jaw)
    chin_jaw_width = np.linalg.norm(chin - ((left_jaw + right_jaw) / 2))
    face_width = np.linalg.norm(left_face - right_face)
    face_height = np.linalg.norm(chin - ((left_eye + right_eye) / 2))
    eye_separation = np.linalg.norm(left_eye - right_eye)
    
    # Compute ratios
    CJWR = chin_jaw_width / jaw_width
    WHR = face_width / face_height
    ES = eye_separation / face_width
    LF_FH = face_width / face_height
    FW_LFH = face_width / face_height
    
    return CJWR, WHR, ES, LF_FH, FW_LFH

def predict_bmi():
    # Initialize face alignment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
    
    # Load validation data
    val_data = pd.read_csv('validationset.csv')
    
    # Load trained model
    model = ShallowModel()
    model.load_state_dict(torch.load('models/final_model.pth'))
    model.eval()
    
    results = []
    
    for _, row in val_data.iterrows():
        filename = row['name']
        gender = row['gender']
        gt_bmi = row['bmi']
        
        # Load and process image
        image_path = os.path.join('src/data/CodingImages/aligned', filename)
        try:
            image = io.imread(image_path)
        except:
            print(f"Could not load image: {image_path}")
            continue
            
        # Get face landmarks
        try:
            landmarks = fa.get_landmarks(image)
            if landmarks is None or len(landmarks) == 0:
                print(f"No face detected in {filename}")
                continue
            landmarks = landmarks[0]  # Get first face if multiple detected
        except:
            print(f"Error detecting landmarks in {filename}")
            continue
        
        # Compute ratios
        CJWR, WHR, ES, LF_FH, FW_LFH = compute_facial_ratios(landmarks)
        
        # Prepare input for model
        gender_encoded = 1 if gender == 'Female' else 0
        features = torch.FloatTensor([[gender_encoded, CJWR, WHR, ES, LF_FH, FW_LFH]])
        
        # Make prediction
        with torch.no_grad():
            predicted_bmi = model(features).item()
            
        results.append({
            'filename': filename,
            'gender': gender,
            'predicted_bmi': unnormalize_bmi(predicted_bmi),
            'predicted_bmi_normalized': predicted_bmi,
            'CJWR': CJWR,
            'WHR': WHR,
            'ES': ES,
            'LF_FH': LF_FH,
            'FW_LFH': FW_LFH,
            'gt_bmi': gt_bmi
        })
        
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('predicted_bmi.csv', index=False)
    
if __name__ == "__main__":
    predict_bmi()

