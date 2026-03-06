import mediapipe as mp
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

# Lazy loading helpers to avoid 'module mediapipe has no attribute solutions' error
def get_mp_face_mesh():
    return mp.solutions.face_mesh

def get_mp_drawing():
    return mp.solutions.drawing_utils

def get_mp_drawing_styles():
    return mp.solutions.drawing_styles

# Store the last known landmarks for movement detection
last_landmarks = None
previous_positions = []
MAX_POSITION_HISTORY = 5  # Store last 5 positions for smoother movement detection

def detect_neck_movement(frame: np.ndarray, 
                         threshold: float = 0.35) -> Tuple[bool, float]:
    """
    Detect excessive neck/head movement using facial landmarks.
    
    Args:
        frame: Input camera frame as numpy array
        threshold: Movement threshold value
        
    Returns:
        Tuple containing (movement_detected, movement_ratio)
    """
    global last_landmarks, previous_positions
    
    mp_face_mesh = get_mp_face_mesh()
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get key facial landmarks for movement detection
        # Nose tip
        nose = landmarks[4]
        # Left and right ear
        left_ear = landmarks[127]
        right_ear = landmarks[356]
        # Chin
        chin = landmarks[152]
        
        # Calculate horizontal movement ratio
        horizontal_ratio = abs(left_ear.x - right_ear.x)
        
        # Calculate vertical movement (head tilt)
        vertical_ratio = abs(nose.y - chin.y)
        
        # Get current position data
        current_pos = {
            'nose_x': nose.x,
            'nose_y': nose.y,
            'left_ear_x': left_ear.x,
            'right_ear_x': right_ear.x,
            'chin_y': chin.y,
            'horizontal_ratio': horizontal_ratio,
            'vertical_ratio': vertical_ratio
        }
        
        # Check for movement based on position history
        movement_ratio = 0.0
        if last_landmarks is not None:
            # Calculate movement from last position
            nose_movement = calculate_distance(
                (nose.x, nose.y), 
                (last_landmarks['nose_x'], last_landmarks['nose_y'])
            )
            
            # Ear position change (side-to-side movement)
            ear_distance_change = abs(horizontal_ratio - last_landmarks['horizontal_ratio'])
            
            # Vertical position change (up-down movement)
            vertical_change = abs(vertical_ratio - last_landmarks['vertical_ratio'])
            
            # Combined movement score - weighted for different types of movement
            movement_ratio = (nose_movement * 3.0 + ear_distance_change * 2.0 + vertical_change * 2.0) / 7.0
        
        # Update position history
        previous_positions.append(current_pos)
        if len(previous_positions) > MAX_POSITION_HISTORY:
            previous_positions.pop(0)
            
        # Use the average of recent positions for smoother detection
        last_landmarks = current_pos
        
        # Calculate average movement over the last few frames
        avg_movement = movement_ratio
        if len(previous_positions) > 1:
            movements = []
            for i in range(1, len(previous_positions)):
                prev = previous_positions[i-1]
                curr = previous_positions[i]
                m = calculate_distance((curr['nose_x'], curr['nose_y']), (prev['nose_x'], prev['nose_y']))
                movements.append(m)
            if movements:
                avg_movement = (sum(movements) / len(movements) + movement_ratio) / 2
        
        # Detect movement based on threshold
        movement_detected = avg_movement > threshold
        
        return movement_detected, avg_movement

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def estimate_head_pose(landmarks: List) -> Tuple[float, float, float]:
    """
    Estimate head pose (roll, pitch, yaw) from facial landmarks.
    
    Args:
        landmarks: MediaPipe face landmarks
        
    Returns:
        Tuple of estimated angles (roll, pitch, yaw) in degrees
    """
    # Simplified head pose estimation - would use more complex methods in production
    # This is a basic approximation
    
    # Get key points for pose estimation
    nose = landmarks[4]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    
    # Calculate roll (head tilt)
    eye_dx = right_eye.x - left_eye.x
    eye_dy = right_eye.y - left_eye.y
    roll = np.arctan2(eye_dy, eye_dx) * 180 / np.pi
    
    # Approximate pitch (up/down)
    mouth_mid_y = (left_mouth.y + right_mouth.y) / 2
    eye_mid_y = (left_eye.y + right_eye.y) / 2
    pitch = (mouth_mid_y - eye_mid_y) * 100  # Scaled for visibility
    
    # Approximate yaw (left/right)
    eye_width = abs(right_eye.x - left_eye.x)
    mouth_width = abs(right_mouth.x - left_mouth.x)
    yaw = (eye_width - mouth_width) * 100  # Scaled for visibility
    
    return roll, pitch, yaw

def visualize_landmarks(frame: np.ndarray, 
                       face_landmarks) -> np.ndarray:
    """
    Draw facial landmarks on the frame for visualization.
    
    Args:
        frame: Input camera frame
        face_landmarks: MediaPipe face landmarks
        
    Returns:
        Frame with landmarks visualized
    """
    annotated_frame = frame.copy()
    
    mp_drawing = get_mp_drawing()
    mp_drawing_styles = get_mp_drawing_styles()
    mp_face_mesh = get_mp_face_mesh()
    
    # Draw the face mesh on the frame
    mp_drawing.draw_landmarks(
        image=annotated_frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    
    # Draw the face contours
    mp_drawing.draw_landmarks(
        image=annotated_frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    
    return annotated_frame