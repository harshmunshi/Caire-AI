import cv2
import numpy as np
import face_alignment
from typing import Tuple, Optional, List, Union

def get_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect facial landmarks in an image using face_alignment.
    
    Args:
        image: Input image containing the face
        
    Returns:
        Array of facial landmarks or None if no face detected
    """
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")
    try:
        landmarks = fa.get_landmarks(image)
        if landmarks is None or len(landmarks) == 0:
            return None
        return landmarks[0]  # Return first face's landmarks
    except:
        return None

def align_face(image: np.ndarray,
               landmarks: Optional[np.ndarray] = None,
               desired_width: int = 256,
               desired_left_eye: Tuple[float, float] = (0.35, 0.35)) -> Optional[np.ndarray]:
    """
    Aligns and scales a face image based on eye positions from landmarks.
    
    Args:
        image: Input image containing the face
        landmarks: Optional pre-computed facial landmarks. If None, will be detected
        desired_width: Desired width of output image
        desired_left_eye: Relative position where left eye should end up
        
    Returns:
        Aligned and scaled face image, or None if alignment fails
    """
    if image is None:
        return None
        
    # Get landmarks if not provided
    if landmarks is None:
        landmarks = get_landmarks(image)
        if landmarks is None:
            return None
    
    # Extract eye positions from landmarks
    # Landmarks 36-41 are right eye, 42-47 are left eye
    left_eye = landmarks[42:48].mean(axis=0)
    right_eye = landmarks[36:42].mean(axis=0)
    
    # left_eye = tuple(map(int, left_eye))
    # right_eye = tuple(map(int, right_eye))
        
    # Calculate angle between eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Check if face is upside down (angle > 90 or < -90 degrees)
    # if angle > 90:
    #     angle -= 180
    # elif angle < -90:
    #     angle += 180

    # Calculate desired right eye x-coordinate based on left eye position
    desired_right_eye_x = 1.0 - desired_left_eye[0]

    # Calculate distance between eyes and desired distance
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_width
    scale = desired_dist / dist

    # Calculate center point between eyes
    eyes_center = (int((left_eye[0] + right_eye[0]) // 2),
                  int((left_eye[1] + right_eye[1]) // 2))

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Update translation component of matrix
    tX = desired_width * 0.5
    tY = desired_width * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Apply affine transformation
    output = cv2.warpAffine(image, 
                           M, 
                           (desired_width, desired_width),
                           flags=cv2.INTER_CUBIC)

    return output

def align_single_face(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Aligns and scales a single face image.
    """
    landmarks = get_landmarks(image)
    if landmarks is None:
        return None
    return align_face(image, landmarks)
