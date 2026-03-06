import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

# Using OpenCV's face detection instead of face_recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_multiple_persons(frame: np.ndarray) -> Tuple[bool, int]:
    """
    Detect if multiple persons are present in the frame using OpenCV.
    
    Args:
        frame: Input camera frame
        
    Returns:
        Tuple containing (multiple_detected, face_count)
    """
    # Convert to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    face_count = len(faces)
    
    # Return True if more than one face is detected
    return face_count > 1, face_count

def extract_face_encodings(frame: np.ndarray) -> Tuple[List, List]:
    """
    Extract face locations using OpenCV.
    
    Args:
        frame: Input camera frame
        
    Returns:
        Tuple containing (face_locations, empty_list)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Convert to the format expected by other functions (top, right, bottom, left)
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x + w, y + h, x))
    
    # Return locations and empty encodings (not used without face_recognition)
    return face_locations, []

def track_person_consistency(current_encoding: np.ndarray, 
                            reference_encoding: np.ndarray,
                            threshold: float = 0.6) -> bool:
    """
    Simplified version that always returns True without face_recognition.
    """
    return True

def draw_face_boxes(frame: np.ndarray, 
                   face_locations: List[Tuple[int, int, int, int]],
                   is_multiple: bool = False) -> np.ndarray:
    """
    Draw boxes around detected faces with labels.
    
    Args:
        frame: Input camera frame
        face_locations: List of face locations
        is_multiple: Flag indicating if multiple people were detected
        
    Returns:
        Frame with annotated face boxes
    """
    annotated_frame = frame.copy()
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Draw a box around the face
        color = (0, 0, 255) if is_multiple else (0, 255, 0)
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with face number
        label = f"Person {i+1}"
        cv2.putText(annotated_frame, label, (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add warning for multiple faces
    if is_multiple:
        cv2.putText(annotated_frame, "WARNING: Multiple people detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated_frame 