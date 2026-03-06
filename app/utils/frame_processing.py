import cv2
import numpy as np
from typing import Tuple, Dict, Any, List

from app.utils.pose_analysis import detect_neck_movement, visualize_landmarks
from app.utils.face_detection import detect_multiple_persons, draw_face_boxes, extract_face_encodings
from app.utils.alerts import continuous_tracking

def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process a single frame from the video feed with all detection algorithms.
    
    This function follows the RORO pattern (Receive Object, Return Object)
    for clean data passing between pipeline stages.
    
    Args:
        frame: Input camera frame as numpy array
        
    Returns:
        Tuple of (processed_frame, metadata_dict)
    """
    if frame is None or frame.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8), {}
    
    # Create a copy for annotation
    processed_frame = frame.copy()
    metadata = {}
    
    try:
        # Detect multiple persons
        multiple_detected, face_count = detect_multiple_persons(frame)
        metadata['multiple_people'] = (multiple_detected, face_count)
        
        # Get face locations for drawing
        face_locations, _ = extract_face_encodings(frame)
        
        # Draw face boxes
        processed_frame = draw_face_boxes(
            processed_frame, 
            face_locations, 
            is_multiple=multiple_detected
        )
        
        # Detect neck movement
        neck_moved, movement_ratio = detect_neck_movement(frame)
        metadata['neck_movement'] = (neck_moved, movement_ratio)
        
        # Add status indicators and metrics on the frame
        add_status_indicators(processed_frame, metadata)
        
    except Exception as e:
        # On error, return original frame with error message
        cv2.putText(processed_frame, f"Error: {str(e)}", 
                   (10, processed_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        metadata['error'] = str(e)
    
    return processed_frame, metadata

def add_status_indicators(frame: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Add status indicators and metrics to the frame.
    
    Args:
        frame: Input frame to annotate
        metadata: Detection metadata
        
    Returns:
        Annotated frame
    """
    height, width = frame.shape[:2]
    
    # Draw rectangle for status area
    cv2.rectangle(frame, (width - 210, 10), (width - 10, 160), (0, 0, 0), -1)
    cv2.rectangle(frame, (width - 210, 10), (width - 10, 160), (255, 255, 255), 1)
    
    # Add title
    cv2.putText(frame, "Proctoring Status", 
               (width - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset = 55
    spacing = 30
    
    # Movement status
    if 'neck_movement' in metadata:
        moved, ratio = metadata['neck_movement']
        status_color = (0, 0, 255) if continuous_tracking['neck_movement']['is_active'] else (0, 255, 0)
        status_text = "HEAD: Warning" if continuous_tracking['neck_movement']['is_active'] else "HEAD: OK"
        cv2.putText(frame, status_text, 
                   (width - 200, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Movement meter
        meter_width = 100
        cv2.rectangle(frame, (width - 200, y_offset + 5), (width - 200 + meter_width, y_offset + 15), (50, 50, 50), -1)
        meter_value = int(ratio * meter_width * 2)
        cv2.rectangle(frame, (width - 200, y_offset + 5), 
                     (width - 200 + meter_value, y_offset + 15), 
                     status_color, -1)
        
        # Add duration indicator if active
        if continuous_tracking['neck_movement']['is_active']:
            duration = continuous_tracking['neck_movement']['duration']
            cv2.putText(frame, f"{duration:.1f}s", 
                      (width - 70, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        y_offset += spacing
    
    # Multiple people status
    if 'multiple_people' in metadata:
        multiple, count = metadata['multiple_people']
        status_color = (0, 0, 255) if continuous_tracking['multiple_people']['is_active'] else (0, 255, 0)
        status_text = f"PEOPLE: {count} (Warning)" if continuous_tracking['multiple_people']['is_active'] else f"PEOPLE: {count} (OK)"
        cv2.putText(frame, status_text, 
                   (width - 200, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Add duration indicator if active
        if continuous_tracking['multiple_people']['is_active']:
            duration = continuous_tracking['multiple_people']['duration']
            cv2.putText(frame, f"{duration:.1f}s", 
                      (width - 70, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        y_offset += spacing
    
    # Absence status
    if continuous_tracking['absence']['is_active']:
        status_color = (0, 0, 255)
        duration = continuous_tracking['absence']['duration']
        cv2.putText(frame, f"ABSENCE: {duration:.1f}s", 
                   (width - 200, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += spacing
    
    # Timestamp
    cv2.putText(frame, "LIVE", 
               (width - 200, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame 