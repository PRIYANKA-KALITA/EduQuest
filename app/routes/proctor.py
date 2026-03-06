from flask import Blueprint, Response, render_template, current_app, jsonify
import cv2
from typing import Generator, Tuple, Dict, List, Any
import time
import numpy as np
import logging
import base64
import io
from PIL import Image

from app.utils.pose_analysis import detect_neck_movement
from app.utils.face_detection import detect_multiple_persons
from app.utils.alerts import check_suspicious_activity, get_activity_summary
from app.utils.frame_processing import process_frame, add_status_indicators

# Set up logger
logger = logging.getLogger(__name__)

proctor_bp = Blueprint('proctor', __name__, url_prefix='/proctor')

def gen_frames() -> Generator[bytes, None, None]:
    """Generate camera frames with real-time processing."""
    # Initialize with default values
    camera_index = 0
    frame_width = 640
    frame_height = 480
    fps = 24
    
    # Try to get values from app config if available
    try:
        camera_index = current_app.config.get('CAMERA_SOURCE', 0)
        frame_width = current_app.config.get('FRAME_WIDTH', 640)
        frame_height = current_app.config.get('FRAME_HEIGHT', 480)
        fps = current_app.config.get('FPS', 24)
    except (RuntimeError, KeyError):
        # Use defaults if not in app context or keys missing
        pass
    
    # Initialize the camera
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    
    # For demo purposes, simulate warnings after a few frames
    frame_count = 0
    warning_sent = False
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                # If camera read fails, return a blank frame with error message
                blank_frame = create_error_frame(frame_width, frame_height, "Camera not available")
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)  # Wait before trying again
                continue
            
            # Process frame and check for suspicious activity
            processed_frame, metadata = process_frame(frame)
            frame_count += 1
            
            # Check for alerts less frequently to improve performance
            face_detection_frequency = 5
            if frame_count % face_detection_frequency == 0:
                try:
                    check_suspicious_activity(frame, metadata)
                except Exception as e:
                    print(f"Error in suspicious activity check: {e}")
            
            # Use processed frame instead of original
            frame = processed_frame
            
            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Yield the frame in the format expected by multipart HTTP response
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()

def create_error_frame(width: int, height: int, message: str) -> np.ndarray:
    """Create an error message frame when camera is not available."""
    frame = np.zeros((height, width, 3), np.uint8)
    cv2.putText(frame, message, (width // 4, height // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

@proctor_bp.route('/video_feed')
def video_feed() -> Response:
    """Stream video feed with processed frames."""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@proctor_bp.route('/dashboard')
def dashboard() -> str:
    """Render the proctoring dashboard."""
    return render_template('dashboard.html')

@proctor_bp.route('/verify_image', methods=['POST'])
def verify_image():
    """
    Process an image from a POST request and check for suspicious activity.
    
    Expects a base64 encoded image in the request body.
    
    Returns:
        JSON response with detection results
    """
    from flask import request
    
    try:
        # Get image data from request
        data = request.json
        if not data or 'image_data' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No image data provided'
            }), 400
        
        # Decode base64 image
        image_data = data['image_data']
        if image_data.startswith('data:image'):
            # Remove data URL prefix if present
            image_data = image_data.split(',')[1]
        
        # Convert to OpenCV format
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        processed_frame, metadata = process_frame(frame)
        
        # Extract face data for client-side visualization
        face_data = {}
        if 'multiple_people' in metadata:
            _, face_count = metadata['multiple_people']
            face_data['face_count'] = face_count
        
        if 'neck_movement' in metadata:
            moved, ratio = metadata['neck_movement']
            face_data['movement'] = {
                'detected': moved,
                'ratio': ratio
            }
        
        # Extract warnings
        warnings = metadata.get('warnings', [])
        alerts = [w['message'] for w in warnings]
        
        return jsonify({
            'status': 'ok' if not alerts else 'warning',
            'alerts': alerts,
            'face_data': face_data,
            'warning_count': len(alerts)
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}',
            'alerts': ["System error occurred"]
        }), 500

@proctor_bp.route('/camera_status')
def camera_status() -> Dict:
    """Return camera status for health check."""
    return jsonify({"status": "active"})

@proctor_bp.route('/activity_summary')
def activity_summary() -> Dict:
    """Return a summary of suspicious activities."""
    try:
        summary = get_activity_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting activity summary: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error getting activity summary: {str(e)}'
        }), 500

@proctor_bp.route('/toggle_detection', methods=['POST'])
def toggle_detection() -> Dict:
    """Toggle specific detection types on/off."""
    from flask import request
    
    try:
        data = request.json
        if not data or 'type' not in data or 'enabled' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: type and enabled'
            }), 400
            
        detection_type = data['type']
        enabled = data['enabled']
        
        # Store in app config
        if 'DETECTION_SETTINGS' not in current_app.config:
            current_app.config['DETECTION_SETTINGS'] = {}
            
        current_app.config['DETECTION_SETTINGS'][detection_type] = enabled
        
        logger.info(f"Detection type '{detection_type}' set to {enabled}")
        
        return jsonify({
            'status': 'ok',
            'message': f"Detection type '{detection_type}' set to {enabled}"
        })
        
    except Exception as e:
        logger.error(f"Error toggling detection: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error toggling detection: {str(e)}'
        }), 500 