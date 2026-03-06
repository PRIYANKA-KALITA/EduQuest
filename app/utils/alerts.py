from flask_socketio import SocketIO, emit
import time
from typing import Dict, List, Any, Tuple
import numpy as np

# Import pose_analysis and face_detection here to avoid circular imports
from app.utils.pose_analysis import detect_neck_movement
from app.utils.face_detection import detect_multiple_persons

# Store timestamps of suspicious activities
activity_log = {
    'neck_movement': [],
    'multiple_people': [],
    'absence': []
}

# Store continuous activity tracking
continuous_tracking = {
    'neck_movement': {
        'start_time': None,
        'duration': 0,
        'is_active': False
    },
    'multiple_people': {
        'start_time': None,
        'duration': 0,
        'is_active': False
    },
    'absence': {
        'start_time': None,
        'duration': 0,
        'is_active': False
    }
}

# Movement thresholds
MOVEMENT_DURATION_THRESHOLD = 2.0  # Seconds of continuous movement to trigger alert
MULTIPLE_PEOPLE_DURATION_THRESHOLD = 2.0  # Seconds of multiple people to trigger alert
ABSENCE_DURATION_THRESHOLD = 5.0  # Seconds of absence to trigger alert

# Store last frame processed time for absence detection
last_detection_time = time.time()

def check_suspicious_activity(frame: np.ndarray, 
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Check for suspicious activities in the frame and emit alerts.
    
    Args:
        frame: Input camera frame
        metadata: Dictionary of pre-computed metadata (optional)
        
    Returns:
        Dictionary of detected warnings
    """
    global last_detection_time
    warnings = []
    current_time = time.time()
    
    # Use pre-computed data if available, otherwise compute
    if metadata and 'neck_movement' in metadata:
        neck_moved, movement_ratio = metadata['neck_movement']
    else:
        neck_moved, movement_ratio = detect_neck_movement(frame)
        
    if metadata and 'multiple_people' in metadata:
        multiple_people, face_count = metadata['multiple_people']
    else:
        multiple_people, face_count = detect_multiple_persons(frame)
    
    # Track neck movement duration
    track_continuous_activity('neck_movement', neck_moved, current_time, MOVEMENT_DURATION_THRESHOLD)
    
    # Track multiple people duration
    track_continuous_activity('multiple_people', multiple_people, current_time, MULTIPLE_PEOPLE_DURATION_THRESHOLD)
    
    # Check for absence (no faces)
    if face_count == 0:
        track_continuous_activity('absence', True, current_time, ABSENCE_DURATION_THRESHOLD)
    else:
        track_continuous_activity('absence', False, current_time, ABSENCE_DURATION_THRESHOLD)
        last_detection_time = current_time
    
    # Generate warnings based on continuous activity tracking
    if continuous_tracking['neck_movement']['is_active']:
        warnings.append({
            "type": "neck_movement", 
            "message": f"Excessive head movement detected for {continuous_tracking['neck_movement']['duration']:.1f}s",
            "severity": "medium",
            "value": movement_ratio,
            "duration": continuous_tracking['neck_movement']['duration']
        })
        activity_log['neck_movement'].append(current_time)
    
    if continuous_tracking['multiple_people']['is_active']:
        warnings.append({
            "type": "multiple_people", 
            "message": f"{face_count} persons detected in frame for {continuous_tracking['multiple_people']['duration']:.1f}s",
            "severity": "high",
            "count": face_count,
            "duration": continuous_tracking['multiple_people']['duration']
        })
        activity_log['multiple_people'].append(current_time)
    
    if continuous_tracking['absence']['is_active']:
        warnings.append({
            "type": "absence", 
            "message": f"No person detected for {continuous_tracking['absence']['duration']:.1f} seconds",
            "severity": "high",
            "duration": continuous_tracking['absence']['duration']
        })
        activity_log['absence'].append(current_time)
    
    # If any warnings, emit through socketio
    if warnings:
        emit_warning(warnings)
    
    return {"warnings": warnings}

def track_continuous_activity(activity_type: str, is_detected: bool, current_time: float, threshold: float) -> None:
    """
    Track continuous activity over time and set flags when thresholds are exceeded.
    
    Args:
        activity_type: Type of activity to track ('neck_movement', 'multiple_people', 'absence')
        is_detected: Whether the activity is currently detected
        current_time: Current timestamp
        threshold: Duration threshold to trigger alert
    """
    tracking = continuous_tracking[activity_type]
    
    if is_detected:
        # Activity is currently happening
        if tracking['start_time'] is None:
            # Activity just started
            tracking['start_time'] = current_time
            tracking['duration'] = 0
            tracking['is_active'] = False
        else:
            # Activity continuing
            tracking['duration'] = current_time - tracking['start_time']
            
            # Check if duration exceeds threshold
            if tracking['duration'] >= threshold and not tracking['is_active']:
                tracking['is_active'] = True
    else:
        # Activity stopped
        if tracking['is_active']:
            # Only log when we were previously active
            print(f"{activity_type} stopped after {tracking['duration']:.1f}s")
            
        # Reset tracking
        tracking['start_time'] = None
        tracking['duration'] = 0
        tracking['is_active'] = False

def emit_warning(warnings: List[Dict[str, Any]]) -> None:
    """
    Emit warnings through socketio.
    
    Args:
        warnings: List of warning dictionaries
    """
    try:
        # Import socketio here to avoid circular imports
        from app import socketio
        socketio.emit('proctor_alert', {'warnings': warnings})
    except Exception as e:
        print(f"Error emitting alert: {e}")

def get_activity_summary() -> Dict[str, Any]:
    """
    Get a summary of suspicious activities.
    
    Returns:
        Dictionary with activity summary
    """
    current_time = time.time()
    # Only include events from the last 10 minutes
    time_window = current_time - (10 * 60)
    
    recent_neck_movements = [t for t in activity_log['neck_movement'] if t > time_window]
    recent_multiple_people = [t for t in activity_log['multiple_people'] if t > time_window]
    recent_absences = [t for t in activity_log['absence'] if t > time_window]
    
    # Add current active statuses
    summary = {
        "neck_movement_count": len(recent_neck_movements),
        "multiple_people_count": len(recent_multiple_people),
        "absence_count": len(recent_absences),
        "total_violations": len(recent_neck_movements) + len(recent_multiple_people) + len(recent_absences),
        "active_warnings": {
            "neck_movement": continuous_tracking['neck_movement']['is_active'],
            "multiple_people": continuous_tracking['multiple_people']['is_active'],
            "absence": continuous_tracking['absence']['is_active']
        }
    }
    
    return summary 