from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime, timedelta
import random
import logging
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import time
from bson import ObjectId

# Global holders for lazy initialization
_mp_face_detection_tool = None
_mp_face_mesh_tool = None

def get_mediapipe_tools():
    """Lazy initialize MediaPipe tools to avoid import issues on production servers"""
    global _mp_face_detection_tool, _mp_face_mesh_tool
    
    if _mp_face_detection_tool is None:
        try:
            from mediapipe.solutions import face_detection as mp_face_detection
            from mediapipe.solutions import face_mesh as mp_face_mesh
            
            _mp_face_detection_tool = mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            _mp_face_mesh_tool = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe tools initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe tools: {e}")
    
    return _mp_face_detection_tool, _mp_face_mesh_tool

from ..mongodb import MongoManager
from ..models.submission import Submission
from ..models.user import User
from ..models.exam import Exam
from ..models.question import Question
from ..models.class_model import Class
# Removed unused import

student_bp = Blueprint('student', __name__, url_prefix='/student')

# Set up logging
logger = logging.getLogger(__name__)

# MediaPipe drawings settings handled via mp.solutions when needed
# (removed top-level initialization)

# Warning tracking
warning_counts = {}

# Enhanced tracking with better verification and cooldown
activity_tracking = {
    'neck_movement': {
        'start_time': None,
        'duration': 0,
        'is_active': False,
        'verification_count': 0,
        'last_warning_time': 0,
        'consecutive_detections': 0
    },
    'multiple_people': {
        'start_time': None,
        'duration': 0,
        'is_active': False,
        'verification_count': 0,
        'last_warning_time': 0,
        'consecutive_detections': 0
    },
    'looking_away': {
        'start_time': None,
        'duration': 0,
        'is_active': False,
        'verification_count': 0,
        'last_warning_time': 0,
        'consecutive_detections': 0
    },
    'camera_blocked': {
        'start_time': None,
        'duration': 0,
        'is_active': False,
        'verification_count': 0,
        'last_warning_time': 0,
        'consecutive_detections': 0
    },
    'no_person_detected': {
        'start_time': None,
        'duration': 0,
        'is_active': False,
        'verification_count': 0,
        'last_warning_time': 0,
        'consecutive_detections': 0
    }
}

# Enhanced thresholds with STRICT verification to stop spam
VERIFICATION_THRESHOLD = 5  # Reduced from 10 - faster detection
WARNING_COOLDOWN = 20.0  # 20 seconds between same type warnings
MOVEMENT_DURATION_THRESHOLD = 3.0  # 3 seconds of movement
MULTIPLE_PEOPLE_DURATION_THRESHOLD = 2.0  # 2 seconds
LOOKING_AWAY_DURATION_THRESHOLD = 3.0  # 3 seconds
CAMERA_BLOCKED_DURATION_THRESHOLD = 2.0  # 2 seconds
NECK_MOVEMENT_THRESHOLD = 0.08  # More sensitive

def get_warning_count(student_id, exam_id):
    """Get the total warning count for a student in an exam"""
    try:
        # Get warnings from session
        warning_key = f'warnings_{student_id}_{exam_id}'
        return session.get(warning_key, 0)
    except Exception as e:
        print(f"Error getting warning count: {e}")
        return 0

def increment_warning_count(user_id, exam_id):
    key = f"{user_id}_{exam_id}"
    warning_counts[key] = warning_counts.get(key, 0) + 1
    return warning_counts[key]

def reset_warning_count(user_id, exam_id):
    key = f"{user_id}_{exam_id}"
    warning_counts[key] = 0
    return 0

def csrf_exempt(f):
    f._csrf_exempt = True
    return f

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'student':
            flash('Access denied. Students only.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

# Global variables for tracking
last_landmarks = None
previous_positions = []
MAX_POSITION_HISTORY = 10

# Eye Aspect Ratio (EAR) calculation for better detection
def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio using 6 eye landmarks"""
    # Vertical eye landmarks
    A = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                      np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    B = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - 
                      np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    # Horizontal eye landmark
    C = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                      np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

def detect_neck_movement(frame, threshold=NECK_MOVEMENT_THRESHOLD):
    """Enhanced neck movement detection using MediaPipe Face Mesh with EAR and head pose"""
    global last_landmarks, previous_positions
    
    try:
        _, face_mesh = get_mediapipe_tools()
        if face_mesh is None:
            return False, 0.0, "Face detection disabled"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, "No face detected"
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Key facial landmarks for comprehensive movement detection
        nose_tip = landmarks[1]  # Nose tip
        left_eye_outer = landmarks[33]  # Left eye outer corner
        right_eye_outer = landmarks[263]  # Right eye outer corner
        chin = landmarks[175]  # Chin
        forehead = landmarks[10]  # Forehead center
        
        # Left eye landmarks for EAR calculation (6 points)
        left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        # Right eye landmarks for EAR calculation (6 points)  
        right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Head pose estimation using key points
        face_2d = []
        face_3d = []
        
        # 3D model points for head pose
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points
        img_h, img_w = frame.shape[:2]
        face_2d.append([nose_tip.x * img_w, nose_tip.y * img_h])
        face_2d.append([chin.x * img_w, chin.y * img_h])
        face_2d.append([left_eye_outer.x * img_w, left_eye_outer.y * img_h])
        face_2d.append([right_eye_outer.x * img_w, right_eye_outer.y * img_h])
        face_2d.append([landmarks[61].x * img_w, landmarks[61].y * img_h])  # Left mouth
        face_2d.append([landmarks[291].x * img_w, landmarks[291].y * img_h])  # Right mouth
        
        face_2d = np.array(face_2d, dtype=np.float64)
        
        # Camera parameters (estimated)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                              [0, focal_length, img_w / 2],
                              [0, 0, 1]])
        
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, cam_matrix, dist_matrix)
        
        # Get rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        
        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Get the y rotation degree
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360
        
        # Calculate movement metrics
        current_pos = {
            'nose_x': nose_tip.x,
            'nose_y': nose_tip.y,
            'ear': avg_ear,
            'x_angle': x_angle,
            'y_angle': y_angle,
            'z_angle': z_angle,
            'timestamp': time.time()
        }
        
        movement_ratio = 0.0
        movement_type = "normal"
        
        if last_landmarks is not None:
            # Calculate position change
            nose_movement = calculate_distance(
                (nose_tip.x, nose_tip.y), 
                (last_landmarks['nose_x'], last_landmarks['nose_y'])
            )
            
            # Calculate angle changes
            angle_change_x = abs(x_angle - last_landmarks['x_angle'])
            angle_change_y = abs(y_angle - last_landmarks['y_angle'])
            angle_change_z = abs(z_angle - last_landmarks['z_angle'])
            
            # EAR change (indicates blinking/eye movement)
            ear_change = abs(avg_ear - last_landmarks['ear'])
            
            # Combined movement score
            movement_ratio = (nose_movement * 2.0 + 
                            angle_change_x * 0.02 + 
                            angle_change_y * 0.02 + 
                            angle_change_z * 0.02 + 
                            ear_change * 0.5)
            
            # Determine movement type based on dominant change
            if angle_change_y > 15:  # Looking left/right
                movement_type = "head_turn"
            elif angle_change_x > 10:  # Looking up/down
                movement_type = "head_tilt"
            elif nose_movement > 0.03:
                movement_type = "head_movement"
            elif ear_change > 0.1:
                movement_type = "eye_movement"
        
        # Store position history
        previous_positions.append(current_pos)
        if len(previous_positions) > MAX_POSITION_HISTORY:
            previous_positions.pop(0)
        
        last_landmarks = current_pos
        
        # Enhanced movement detection
        movement_detected = movement_ratio > threshold
        
        return movement_detected, movement_ratio, movement_type
        
    except Exception as e:
        logger.error(f"Error in neck movement detection: {str(e)}")
        return False, 0.0, "detection_error"

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def detect_multiple_persons(frame):
    """Enhanced multiple person detection with better face counting"""
    try:
        face_detection, _ = get_mediapipe_tools()
        if face_detection is None:
            return False, 0, 0.0, False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        face_count = 0
        confidence_scores = []
        
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                if confidence > 0.4:  # Lower threshold for better detection
                    face_count += 1
                    confidence_scores.append(confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Return multiple people detection AND if NO person is detected
        multiple_people_detected = face_count > 1
        no_person_detected = face_count == 0
        
        return multiple_people_detected, face_count, avg_confidence, no_person_detected
        
    except Exception as e:
        logger.error(f"Error in multiple person detection: {str(e)}")
        return False, 0, 0.0, False

def detect_looking_away(frame):
    """Enhanced looking away detection using iris tracking and gaze estimation"""
    try:
        _, face_mesh = get_mediapipe_tools()
        if face_mesh is None:
            return False, 0.0, "Face detection disabled"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, "no_face"
        
        landmarks = results.multi_face_landmarks[0].landmark
        img_h, img_w = frame.shape[:2]
        
        # Left eye landmarks
        left_eye_landmarks = [landmarks[i] for i in [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]]
        # Right eye landmarks  
        right_eye_landmarks = [landmarks[i] for i in [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]
        
        # Calculate eye centers
        left_eye_center_x = sum([lm.x for lm in left_eye_landmarks]) / len(left_eye_landmarks)
        left_eye_center_y = sum([lm.y for lm in left_eye_landmarks]) / len(left_eye_landmarks)
        
        right_eye_center_x = sum([lm.x for lm in right_eye_landmarks]) / len(right_eye_landmarks)
        right_eye_center_y = sum([lm.y for lm in right_eye_landmarks]) / len(right_eye_landmarks)
        
        # Use specific landmarks for pupil/iris estimation
        left_pupil_x = landmarks[159].x  # Left eye center landmark
        left_pupil_y = landmarks[159].y
        right_pupil_x = landmarks[386].x  # Right eye center landmark  
        right_pupil_y = landmarks[386].y
        
        # Calculate gaze direction (pupil position relative to eye center)
        left_gaze_x = left_pupil_x - left_eye_center_x
        left_gaze_y = left_pupil_y - left_eye_center_y
        
        right_gaze_x = right_pupil_x - right_eye_center_x
        right_gaze_y = right_pupil_y - right_eye_center_y
        
        # Average gaze direction
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
        avg_gaze_y = (left_gaze_y + right_gaze_y) / 2
        
        # Calculate gaze intensity (distance from center)
        gaze_intensity = np.sqrt(avg_gaze_x**2 + avg_gaze_y**2)
        
        # Determine gaze direction
        direction = "center"
        if abs(avg_gaze_x) > abs(avg_gaze_y):
            if avg_gaze_x > 0.015:  # Looking right
                direction = "right"
            elif avg_gaze_x < -0.015:  # Looking left
                direction = "left"
        else:
            if avg_gaze_y > 0.01:  # Looking down
                direction = "down"
            elif avg_gaze_y < -0.01:  # Looking up
                direction = "up"
        
        # Enhanced head pose for additional validation
        nose_tip = landmarks[1]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        
        # Face asymmetry (indicates head turn)
        face_asymmetry = abs((nose_tip.x - left_eye_outer.x) - (right_eye_outer.x - nose_tip.x))
        
        # Combined looking away score
        looking_away_score = gaze_intensity + face_asymmetry * 0.5
        
        # Threshold for looking away (more sensitive)
        looking_away_threshold = 0.025
        looking_away = looking_away_score > looking_away_threshold and direction != "center"
        
        return looking_away, looking_away_score, direction
        
    except Exception as e:
        logger.error(f"Error in looking away detection: {str(e)}")
        return False, 0.0, "detection_error"

def detect_camera_blocked(frame):
    """Enhanced camera blocking detection with multiple metrics"""
    try:
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple brightness metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_brightness = np.min(gray)
        max_brightness = np.max(gray)
        
        # HSV analysis
        mean_saturation = np.mean(hsv[:,:,1])
        mean_value = np.mean(hsv[:,:,2])
        
        # Edge detection for texture analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # Multiple blocking conditions (more sensitive)
        conditions = {
            'very_dark': mean_brightness < 25,
            'uniform_dark': std_brightness < 10 and mean_brightness < 30,
            'completely_black': max_brightness < 20,
            'very_bright': mean_brightness > 230,
            'uniform_bright': std_brightness < 15 and mean_brightness > 220,
            'low_saturation': mean_saturation < 20,
            'no_texture': edge_density < 0.01,
            'extreme_values': mean_value < 30 or mean_value > 240
        }
        
        # Count active conditions
        active_conditions = sum(conditions.values())
        
        # Camera is blocked if multiple conditions are met
        is_blocked = active_conditions >= 2
        
        # Determine blocking type
        blocking_type = "none"
        if conditions['completely_black'] or conditions['very_dark']:
            blocking_type = "dark_blocking"
        elif conditions['very_bright'] or conditions['uniform_bright']:
            blocking_type = "bright_blocking"
        elif conditions['no_texture']:
            blocking_type = "covered"
        elif conditions['low_saturation']:
            blocking_type = "partial_block"
        
        return is_blocked, mean_brightness, blocking_type
        
    except Exception as e:
        logger.error(f"Error in camera blocking detection: {str(e)}")
        return False, 0.0, "detection_error"

def track_activity_with_verification(activity_type, is_detected, current_time, threshold, additional_info=""):
    """Enhanced activity tracking with verification and STRICT cooldown"""
    tracking = activity_tracking[activity_type]
    
    # STRICT COOLDOWN CHECK - if we recently warned, don't process anything
    if current_time - tracking['last_warning_time'] < WARNING_COOLDOWN:
        return False  # Don't process, still in cooldown
    
    if is_detected:
        tracking['consecutive_detections'] += 1
        
        if tracking['start_time'] is None:
            # Activity just started
            tracking['start_time'] = current_time
            tracking['duration'] = 0
            tracking['verification_count'] = 1
            tracking['is_active'] = False
        else:
            # Activity continuing
            tracking['duration'] = current_time - tracking['start_time']
            tracking['verification_count'] += 1
            
            # Check if we have enough verification and duration AND not in cooldown
            if (tracking['verification_count'] >= VERIFICATION_THRESHOLD and 
                tracking['duration'] >= threshold and 
                not tracking['is_active'] and
                tracking['consecutive_detections'] >= VERIFICATION_THRESHOLD):
                
                tracking['is_active'] = True
                tracking['last_warning_time'] = current_time
                logger.warning(f" VERIFIED WARNING {activity_type}: {tracking['verification_count']} confirmations over {tracking['duration']:.1f}s - {additional_info}")
                return True  # Warning should be generated
    else:
        # Activity stopped - reset tracking but keep cooldown
        if tracking['is_active']:
            logger.info(f" {activity_type} stopped after {tracking['duration']:.1f}s")
        
        tracking['start_time'] = None
        tracking['duration'] = 0
        tracking['verification_count'] = 0
        tracking['consecutive_detections'] = 0
        tracking['is_active'] = False
    
    return False  # No warning should be generated

@student_bp.route('/dashboard')
@login_required
@student_required
def dashboard():
    exams = MongoManager.get_available_exams_for_student(current_user.id)
    submissions = MongoManager.get_student_submissions(current_user.id)
    recent_submissions = submissions[:5] if submissions else []
    
    total_exams_taken = len(submissions)
    graded_submissions = [s for s in submissions if s.is_graded and s.score is not None]
    average_score = sum(s.score for s in graded_submissions) / len(graded_submissions) if graded_submissions else 0.0
    pending_reviews = len([s for s in submissions if not s.is_graded])
    
    if recent_submissions:
        for submission in recent_submissions:
            try:
                exam = MongoManager.get_exam_by_id(submission.exam_id)
                submission.exam = exam
            except Exception as e:
                logger.error(f"Error fetching exam for submission {submission.id}: {str(e)}")
                class PlaceholderExam:
                    def __init__(self):
                        self.title = "Exam Not Found"
                        self.id = submission.exam_id
                submission.exam = PlaceholderExam()
    
    stats = {
        'total_exams_taken': total_exams_taken,
        'average_score': average_score,
        'pending_reviews': pending_reviews
    }
    
    return render_template('student/dashboard.html',
                         available_exams=exams,
                         recent_submissions=recent_submissions,
                         stats=stats)

@student_bp.route('/exams')
@login_required
@student_required
def exams():
    exams = MongoManager.get_available_exams_for_student(current_user.id)
    return render_template('student/exams.html', exams=exams)

@student_bp.route('/exam/<exam_id>')
@login_required
@student_required
def exam_details(exam_id):
    exam = MongoManager.get_exam_by_id(exam_id)
    if not exam:
        flash('Exam not found.', 'error')
        return redirect(url_for('student.exams'))
    
    existing_submission = MongoManager.get_student_submission_for_exam(current_user.id, exam_id)
    if existing_submission:
        return redirect(url_for('student.submission_result', submission_id=existing_submission.id))
    
    return render_template('student/exam_details.html', exam=exam)

@student_bp.route('/exam/<exam_id>/take')
@login_required
@student_required
def take_exam(exam_id):
    reset_warning_count(current_user.id, exam_id)
    
    exam = MongoManager.get_exam_by_id(exam_id)
    if not exam:
        flash('Exam not found.', 'error')
        return redirect(url_for('student.dashboard'))
    
    if not exam.is_active_for_student():
        flash('This exam is not currently active.', 'error')
        return redirect(url_for('student.dashboard'))
    
    existing_submission = MongoManager.get_student_submission_for_exam(current_user.id, exam_id)
    if existing_submission:
        flash('You have already submitted this exam.', 'warning')
        return redirect(url_for('student.submission_result', submission_id=existing_submission.id))
    
    # Verify this student's class has access to this exam
    if not MongoManager.student_has_access_to_exam(current_user.id, exam_id):
        flash('You do not have access to this exam.', 'error')
        return redirect(url_for('student.dashboard'))
    
    exam_session_key = f'exam_{exam_id}_questions'
    all_questions = exam.questions
    if not all_questions:
        flash('This exam has no questions. Please contact your teacher.', 'warning')
        return redirect(url_for('student.dashboard'))
    
    print(f"Exam has {len(all_questions)} total questions")
    
    # Get number of questions to display (set by teacher)
    num_questions_to_display = exam.num_questions_to_display or len(all_questions)
    num_questions_to_display = min(num_questions_to_display, len(all_questions))
    
    print(f"Will display {num_questions_to_display} questions for this student")
    
    # Check if this student already has selected questions for this exam
    if exam_session_key in session:
        session_questions = session[exam_session_key]
        print(f"Using existing question set from session ({len(session_questions)} questions)")
        
        # Convert session data back to Question objects if needed
        questions_to_display = []
        for q_data in session_questions:
            if isinstance(q_data, dict):
                # Convert dictionary back to Question object
                from app.models.question import Question
                question = Question.from_db(q_data)
                if question:
                    questions_to_display.append(question)
                    print(f"Successfully processed question {question.id}")
                else:
                    print(f"Failed to process question data: {q_data}")
            else:
                # Already a Question object
                questions_to_display.append(q_data)
                print(f"Successfully processed question {q_data.id}")
        
        print(f"Exam loaded with {len(questions_to_display)} questions")
    else:
        # Generate random question set for this student
        if num_questions_to_display < len(all_questions):
            # Use student ID as seed for consistent randomization per student
            random.seed(f"{current_user.id}_{exam_id}")
            questions_to_display = random.sample(all_questions, num_questions_to_display)
            print(f"Generated random question set for student {current_user.id}")
        else:
            # If displaying all questions, no need to randomize
            questions_to_display = all_questions
            print(f"Using all questions (no randomization needed)")
        
        # Store the selected questions in session for this student (as dictionaries for serialization)
        session[exam_session_key] = [q.to_dict() if hasattr(q, 'to_dict') else q for q in questions_to_display]
        session.modified = True
        print(f"Saved question set to session")

    # Only clear answers and start time if this is a fresh exam start
    start_time_key = f'exam_{exam_id}_start_time'
    if start_time_key not in session:
        # Fresh start - clear any old data and set start time
        session.pop(f'exam_{exam_id}_answers', None)
        session[start_time_key] = datetime.utcnow().isoformat()
        session.modified = True
    
    # Get saved answers (will be empty for fresh start, preserved for continuation)
    saved_answers = session.get(f'exam_{exam_id}_answers', {})
    
    return render_template('student/take_exam.html',
                         exam=exam,
                         questions=questions_to_display,
                         total_questions=len(questions_to_display),
                         current_warning_count=get_warning_count(current_user.id, exam_id),
                         saved_answers=saved_answers)

@student_bp.route('/submission/<submission_id>')
@login_required
@student_required
def submission_result(submission_id):
    try:
        submission = MongoManager.get_submission_by_id(submission_id)
        if not submission or submission.student_id != current_user.id:
            flash('Submission not found.', 'error')
            return redirect(url_for('student.submissions'))
        
        exam = MongoManager.get_exam_by_id(submission.exam_id)
        if not exam:
            flash('Exam not found.', 'error')
            return redirect(url_for('student.submissions'))
        
        # Handle simple dictionary format: question_id -> answer_text
        processed_answers = []
        
        if submission.answers and isinstance(submission.answers, dict):
            for question_id, answer_text in submission.answers.items():
                # Find the corresponding question in the exam
                question_data = None
                for question in exam.questions:
                    if isinstance(question, dict):
                        q_id = str(question.get('_id', question.get('id', '')))
                    else:
                        q_id = str(getattr(question, '_id', getattr(question, 'id', '')))
                    
                    if q_id == question_id:
                        question_data = question
                        break
                
                # Extract actual student answer from nested structure if needed
                actual_student_answer = answer_text
                if isinstance(answer_text, dict):
                    actual_student_answer = answer_text.get('student_answer', answer_text.get('answer', ''))
                    if isinstance(actual_student_answer, dict):
                        actual_student_answer = actual_student_answer.get('student_answer', actual_student_answer.get('answer', ''))
                
                # Create processed answer with question details
                if question_data:
                    marks_per_q = exam.get_marks_per_question()
                    if isinstance(question_data, dict):
                        processed_answer = {
                            'question_id': question_id,
                            'question_text': question_data.get('question_text', 'Question not available'),
                            'student_answer': str(actual_student_answer).strip() if actual_student_answer else '',
                            'marks': marks_per_q,
                            'options': question_data.get('options', []),
                            'correct_answer': question_data.get('correct_answer', ''),
                            'model_answer': question_data.get('model_answer', ''),
                            'question_type': question_data.get('question_type', 'subjective')
                        }
                    else:
                        processed_answer = {
                            'question_id': question_id,
                            'question_text': getattr(question_data, 'question_text', 'Question not available'),
                            'student_answer': str(actual_student_answer).strip() if actual_student_answer else '',
                            'marks': marks_per_q,
                            'options': getattr(question_data, 'options', []),
                            'correct_answer': getattr(question_data, 'correct_answer', ''),
                            'model_answer': getattr(question_data, 'model_answer', ''),
                            'question_type': getattr(question_data, 'question_type', 'subjective')
                        }
                    processed_answers.append(processed_answer)
        
        # Create a simple answers dictionary for template compatibility
        simple_answers = {}
        for answer in processed_answers:
            simple_answers[answer['question_id']] = answer['student_answer']
        
        # Replace the nested answers with processed flat answers for template
        submission.answers = simple_answers
        
        # Also store processed answers for detailed display
        submission.processed_answers = processed_answers
        
        return render_template('student/submission_result.html',
                             exam=exam,
                             submission=submission)
    
    except Exception as e:
        logger.error(f"Error loading submission {submission_id}: {str(e)}")
        flash('Error loading submission details.', 'error')
        return redirect(url_for('student.submissions'))

@student_bp.route('/submission/<submission_id>/view')
@login_required
@student_required
def view_submission(submission_id):
    try:
        submission = MongoManager.get_submission_by_id(submission_id)
        if not submission or submission.student_id != current_user.id:
            flash('Submission not found.', 'error')
            return redirect(url_for('student.submissions'))
        
        exam = MongoManager.get_exam_by_id(submission.exam_id)
        if not exam:
            flash('Exam not found.', 'error')
            return redirect(url_for('student.submissions'))
        
        # Use the EXACT same logic as teacher route for processing submission answers
        processed_answers = []
        
        # Get raw database data directly like teacher route does
        from app.mongodb import mongo
        from bson import ObjectId
        raw_submission_data = mongo.db.submissions.find_one({'_id': ObjectId(submission_id)})
        
        if raw_submission_data:
            # Use the raw database answers directly (like teacher route)
            raw_answers = raw_submission_data.get('answers', {})
            
            if raw_answers and isinstance(raw_answers, dict):
                # Handle dictionary format (key-value pairs) - EXACT teacher logic
                for answer_key, answer_value in raw_answers.items():
                    # Handle both formats: "answer_questionId" and just "questionId"
                    if answer_key.startswith('answer_'):
                        question_id = answer_key.replace('answer_', '')
                    else:
                        question_id = answer_key
                    
                    # Find the corresponding question in the exam
                    question_data = None
                    for question in exam.questions:
                        if isinstance(question, dict):
                            q_id = str(question.get('_id', question.get('id', '')))
                        else:
                            q_id = str(getattr(question, '_id', getattr(question, 'id', '')))
                        
                        if q_id == question_id:
                            question_data = question
                            break
                    
                    # Extract actual student answer from nested structure - EXACT teacher logic
                    actual_student_answer = answer_value
                    if isinstance(answer_value, dict):
                        actual_student_answer = answer_value.get('student_answer', answer_value.get('answer', ''))
                        # If still a dict, try to extract further
                        if isinstance(actual_student_answer, dict):
                            actual_student_answer = actual_student_answer.get('student_answer', actual_student_answer.get('answer', ''))
                    
                    # Create processed answer with question details - EXACT teacher logic
                    if question_data:
                        marks_per_q = exam.get_marks_per_question()
                        if isinstance(question_data, dict):
                            processed_answer = {
                                'question_id': question_id,
                                'question_text': question_data.get('question_text', 'Question not available'),
                                'student_answer': actual_student_answer,
                                'marks': marks_per_q,
                                'options': question_data.get('options', []),
                                'correct_answer': question_data.get('correct_answer', ''),
                                'model_answer': question_data.get('model_answer', ''),
                                'question_type': question_data.get('question_type', 'subjective')
                            }
                        else:
                            processed_answer = {
                                'question_id': question_id,
                                'question_text': getattr(question_data, 'question_text', 'Question not available'),
                                'student_answer': actual_student_answer,
                                'marks': marks_per_q,
                                'options': getattr(question_data, 'options', []),
                                'correct_answer': getattr(question_data, 'correct_answer', ''),
                                'model_answer': getattr(question_data, 'model_answer', ''),
                                'question_type': getattr(question_data, 'question_type', 'subjective')
                            }
                        processed_answers.append(processed_answer)
            
            elif isinstance(raw_answers, list):
                # Handle list format (already processed answers with question data)
                processed_answers = raw_answers
        
        # Create a simple answers dictionary for template compatibility
        simple_answers = {}
        for answer in processed_answers:
            simple_answers[answer['question_id']] = answer['student_answer']
        
        # Replace the nested answers with processed flat answers for template
        submission.answers = simple_answers
        
        # Also store processed answers for detailed display
        submission.processed_answers = processed_answers
        
        return render_template('student/submission_result.html',
                             exam=exam,
                             submission=submission)
    
    except Exception as e:
        logger.error(f"Error viewing submission {submission_id}: {str(e)}")
        flash('Error loading submission details.', 'error')
        return redirect(url_for('student.submissions'))

@student_bp.route('/profile')
@login_required
@student_required
def profile():
    return render_template('student/profile.html')

@student_bp.route('/submissions')
@login_required
@student_required
def submissions():
    submissions = MongoManager.get_student_submissions(current_user.id)
    
    if submissions:
        for submission in submissions:
            try:
                exam = MongoManager.get_exam_by_id(submission.exam_id)
                submission.exam = exam
            except Exception as e:
                logger.error(f"Error fetching exam for submission {submission.id}: {str(e)}")
                class PlaceholderExam:
                    def __init__(self):
                        self.title = "Exam Not Found"
                        self.id = submission.exam_id
                submission.exam = PlaceholderExam()
    
    return render_template('student/submissions.html', submissions=submissions)

@student_bp.route('/results')
@login_required
@student_required
def results():
    # Get only graded submissions for results page
    submissions = MongoManager.get_graded_submissions(current_user.id)
    
    print(f"\n=== RESULTS DEBUG ===")
    print(f"Found {len(submissions) if submissions else 0} graded submissions")
    
    if submissions:
        for submission in submissions:
            try:
                exam = MongoManager.get_exam_by_id(submission.exam_id)
                submission.exam = exam
                
                # Calculate grade based on percentage using 3-tier system
                if hasattr(submission, 'score') and submission.score is not None:
                    # Calculate percentage from score
                    if hasattr(submission, 'max_score') and submission.max_score and submission.max_score > 0:
                        percentage = (submission.score / submission.max_score) * 100
                    elif hasattr(submission, 'percentage') and submission.percentage is not None:
                        percentage = submission.percentage
                    else:
                        # Fallback: assume score is already a percentage
                        percentage = submission.score
                    
                    # Apply 3-tier grading system
                    if percentage >= 70:
                        submission.grade = 'A'
                        submission.grade_color = 'success'
                        submission.pass_status = 'PASS'
                    elif percentage >= 35:
                        submission.grade = 'B' 
                        submission.grade_color = 'warning'
                        submission.pass_status = 'PASS'
                    else:
                        submission.grade = 'F'
                        submission.grade_color = 'danger'
                        submission.pass_status = 'FAIL'
                    
                    # Store calculated percentage for template use
                    submission.calculated_percentage = percentage
                    
                    print(f"Submission {submission.id}: score={submission.score}, max_score={submission.max_score}, percentage={percentage}, grade={submission.grade}")
                else:
                    submission.grade = 'Pending'
                    submission.grade_color = 'secondary'
                    submission.pass_status = 'Pending'
                    submission.calculated_percentage = 0
                    
                    print(f"Submission {submission.id}: No score - set to Pending")
                    
            except Exception as e:
                logger.error(f"Error fetching exam for submission {submission.id}: {str(e)}")
                class PlaceholderExam:
                    def __init__(self):
                        self.title = "Exam Not Found"
                        self.id = submission.exam_id
                submission.exam = PlaceholderExam()
                submission.grade = 'Error'
                submission.grade_color = 'danger'
                submission.pass_status = 'Error'
    
    return render_template('student/results.html', submissions=submissions)

@student_bp.route('/settings')
@login_required
@student_required
def settings():
    return render_template('student/settings.html')

# Profile update route removed - settings page is now read-only

@student_bp.route('/exam/<exam_id>/submit', methods=['POST'])
@login_required
@student_required
def submit_exam(exam_id):
    try:
        exam = MongoManager.get_exam_by_id(exam_id)
        if not exam:
            return jsonify({'success': False, 'message': 'Exam not found'}), 404
        
        existing_submission = MongoManager.get_student_submission_for_exam(current_user.id, exam_id)
        if existing_submission:
            # Already submitted — tell the frontend to redirect gracefully
            sub_id = str(existing_submission.id) if hasattr(existing_submission, 'id') else str(existing_submission)
            return jsonify({
                'success': True,
                'already_submitted': True,
                'message': 'Exam already submitted.',
                'redirect_url': url_for('student.submission_result', submission_id=sub_id)
            })
        
        print(f"\n=== SUBMIT EXAM DEBUG START ===")
        print(f"Exam ID: {exam_id}")
        print(f"User ID: {current_user.id}")
        
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data:
            print("ERROR: No data provided in request")
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        user_answers = data.get('answers', {})
        print(f"Answers from request: {user_answers}")
        print(f"Number of answers from request: {len(user_answers)}")
        
        # Also merge any answers saved in session (from auto-save)
        answers_key = f'exam_{exam_id}_answers'
        session_answers = session.get(answers_key, {})
        print(f"Session answers key: {answers_key}")
        print(f"Session answers: {session_answers}")
        print(f"Number of session answers: {len(session_answers)}")
        
        # Convert session answers to the format expected by the submission
        # Session answers have keys like 'answer_123', we need just '123'
        for key, value in session_answers.items():
            if key.startswith('answer_'):
                question_id = key.replace('answer_', '')
                if question_id not in user_answers:  # Don't override form submission
                    user_answers[question_id] = value if value is not None else ''
                    print(f"Added session answer: {question_id} = {value}")
        
        # Ensure we have answers for all questions in the exam
        exam_session_key = f'exam_{exam_id}_questions'
        session_questions = session.get(exam_session_key, exam.questions)
        
        print(f" Session questions count: {len(session_questions)}")
        print(f" Total exam questions: {len(exam.questions)}")
        print(f" Using {'session-specific' if exam_session_key in session else 'all exam'} questions")
        
        # Add empty answers for questions that weren't answered
        for question in session_questions:
            # Handle both dict and Question object formats
            if isinstance(question, dict):
                question_id = str(question.get('_id') or question.get('id', ''))
            else:
                question_id = str(question.id)
            
            if question_id and question_id not in user_answers:
                user_answers[question_id] = ''
                print(f"Added empty answer for question: {question_id}")
        
        print(f"Final user_answers after merging: {user_answers}")
        print(f"Total final answers: {len(user_answers)}")
        print(f"Questions in exam: {len(session_questions)}")
        
        start_time_str = session.get(f'exam_{exam_id}_start_time')
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
        else:
            start_time = datetime.utcnow()
        
        end_time = datetime.utcnow()
        time_taken = int((end_time - start_time).total_seconds())
        
        score = None
        is_graded = False
        
        # Calculate marks per question based on total questions in this session
        num_session_questions = len(session_questions)
        if num_session_questions > 0:
            # Try to get marks per question from exam settings
            marks_per_question = exam.get_marks_per_question()
            # Fallback if marks per question is not set or zero
            if marks_per_question <= 0:
                marks_per_question = (exam.total_marks / num_session_questions) if num_session_questions > 0 else 0
        else:
            marks_per_question = 0
            
        total_marks = marks_per_question * num_session_questions
        percentage = 0
        grade = None
        pass_fail = None
        answer_details = []
        
        if exam.exam_format == 'objective':
            score, is_graded = auto_grade_objective_exam(exam, user_answers, session_questions)
            
            # Ensure score is not None
            current_score = score if score is not None else 0
            
            # Use pre-calculated values
            percentage = (current_score / total_marks * 100) if total_marks > 0 else 0
            
            # Grade calculation
            if percentage < 35:
                grade = 'F'
                pass_fail = 'FAIL'
                grade_color = '#dc3545'  # Red
            elif percentage < 70:
                grade = 'B'
                pass_fail = 'PASS'
                grade_color = '#ffc107'  # Yellow
            else:
                grade = 'A'
                pass_fail = 'PASS'
                grade_color = '#28a745'  # Green
            
            # Prepare detailed answer comparison
            answer_details = []
            for question in session_questions:
                # Handle both dict and Question object formats
                if isinstance(question, dict):
                    question_id = str(question.get('_id') or question.get('id', ''))
                    question_text = question.get('question_text', '')
                    correct_answer = question.get('correct_answer')
                    question_type = question.get('question_type', 'multiple_choice')
                    marks = marks_per_question
                else:
                    question_id = str(question.id)
                    question_text = question.question_text
                    correct_answer = getattr(question, 'correct_answer', None)
                    question_type = getattr(question, 'question_type', 'multiple_choice')
                    marks = marks_per_question
                
                student_answer = user_answers.get(question_id, '')
                
                # Proper answer comparison
                is_correct = False
                if student_answer and correct_answer:
                    if question_type == 'multiple_choice':
                        try:
                            # Robust numeric check for IDs like '1' vs 1
                            val_std = float(student_answer)
                            val_cor = float(correct_answer)
                            is_correct = val_std == val_cor
                        except (ValueError, TypeError):
                            is_correct = str(student_answer).strip() == str(correct_answer).strip()
                    else:
                        is_correct = str(student_answer).strip().lower() == str(correct_answer).strip().lower()
                
                answer_details.append({
                    'question_number': len(answer_details) + 1,
                    'question_text': question_text[:100] + '...' if len(question_text) > 100 else question_text,
                    'student_answer': student_answer or 'Not answered',
                    'correct_answer': correct_answer or 'Not set',
                    'is_correct': is_correct,
                    'marks': marks,
                    'awarded_marks': marks if is_correct else 0
                })
            
            # Get violation count
            violation_count = get_warning_count(current_user.id, exam_id)
        else:
            # For subjective exams, no auto-grading
            grade = 'Pending'
            pass_fail = 'Pending Grading'
            grade_color = '#6c757d'  # Gray
            answer_details = []
            violation_count = get_warning_count(current_user.id, exam_id)
        
        # Calculate totals from session questions (the ones actually displayed to student)
        questions_answered = len([k for k, v in user_answers.items() if v and str(v).strip()])
        
        # SIMPLE FLAT STRUCTURE: Just save the answers as simple key-value pairs
        formatted_answers = {}
        for question_id, answer_text in user_answers.items():
            # Save ALL answers (empty or not) for consistent structure
            formatted_answers[question_id] = str(answer_text).strip() if answer_text else ''
        
        print(f"   Formatted answers (simple): {formatted_answers}")
        
        submission_data = {
            'student_id': current_user.id,
            'exam_id': exam_id,
            'answers': formatted_answers,  # Properly formatted answers
            'questions': session_questions,  # Store ONLY the questions that were actually displayed to this student
            'submitted_at': end_time,
            'time_taken': time_taken,
            'score': score or 0,
            'max_score': total_marks,  # This is calculated from session_questions, not all exam questions
            'is_graded': is_graded,
            'warning_count': get_warning_count(current_user.id, exam_id),
            'questions_answered': len([v for v in formatted_answers.values() if v and str(v).strip()]),  # Count only non-empty answers
            'detailed_results': answer_details if exam.exam_format == 'objective' else [],
            'total_marks': total_marks,  # Same as max_score for consistency
            'percentage': (score / total_marks * 100) if total_marks > 0 and score is not None else 0,
            'grade': grade if exam.exam_format == 'objective' else None,
            'pass_fail': pass_fail if exam.exam_format == 'objective' else None
        }
        
        print(f"Submission data answers: {submission_data['answers']}")
        print(f"Submission data score: {submission_data['score']}")
        print(f"Submission data detailed_results: {len(submission_data.get('detailed_results', []))} items")
        
        # Create submission object safely
        try:
            submission = Submission(**submission_data)
            print(f"Created Submission object with answers: {len(submission.answers)} items")
            submission_id = MongoManager.create_submission(submission)
            print(f"MongoManager.create_submission returned: {submission_id}")
        except Exception as create_error:
            print(f"Error creating submission: {str(create_error)}")
            # Fallback: try with dictionary directly
            submission_id = MongoManager.create_submission(submission_data)
            print(f"Fallback create_submission returned: {submission_id}")
        
        if submission_id:
            # Handle both Submission object and ObjectId returns
            if hasattr(submission_id, 'id'):
                final_submission_id = str(submission_id.id)
            else:
                final_submission_id = str(submission_id)
            
            session.pop(exam_session_key, None)
            session.pop(f'exam_{exam_id}_answers', None)
            session.pop(f'exam_{exam_id}_start_time', None)
            session.modified = True
            
            reset_warning_count(current_user.id, exam_id)
            
            return jsonify({
                'success': True,
                'message': 'Exam submitted successfully!',
                'submission_id': final_submission_id,
                'redirect_url': url_for('student.submission_result', submission_id=final_submission_id),
                'score': score,
                'is_graded': is_graded,
                'grade': grade,
                'pass_fail': pass_fail,
                'grade_color': grade_color,
                'answer_details': answer_details,
                'violation_count': violation_count,
                'is_objective': exam.exam_format == 'objective',
                'total_questions': len(session_questions),
                'total_marks': total_marks,
                'percentage': submission_data['percentage']
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to submit exam'}), 500
    
    except Exception as e:
        logger.error(f"Error submitting exam {exam_id} for user {current_user.id}: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while submitting the exam'}), 500

@student_bp.route('/exam/<exam_id>/get-session-answers', methods=['GET'])
@login_required
@student_required
def get_session_answers(exam_id):
    """Get saved answers from session"""
    try:
        session_key = f'exam_{exam_id}_answers'
        saved_answers = session.get(session_key, {})
        
        print(f" Getting session answers for exam {exam_id}")
        print(f" Session key: {session_key}")
        print(f" Found {len(saved_answers)} saved answers")
        
        return jsonify({
            'success': True,
            'answers': saved_answers
        })
    except Exception as e:
        print(f" Error getting session answers: {str(e)}")
        return jsonify({
            'success': False,
            'answers': {},
            'error': str(e)
        })

@student_bp.route('/exam/<exam_id>/save-answer', methods=['POST'])
@login_required
@student_required
def save_answer(exam_id):
    try:
        print(f"\n=== SAVE ANSWER DEBUG ===")
        print(f"Exam ID: {exam_id}")
        print(f"User ID: {current_user.id}")
        
        # Get data from request
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data:
            print("ERROR: No data provided")
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        question_id = data.get('question_id')
        answer = data.get('answer')
        
        print(f"Question ID: {question_id}, Answer: {answer}")
        
        if question_id is None:
            print("ERROR: Question ID missing")
            return jsonify({'success': False, 'message': 'Question ID required'}), 400
        
        # Save to session
        answers_key = f'exam_{exam_id}_answers'
        if answers_key not in session:
            session[answers_key] = {}
        
        session[answers_key][f'answer_{question_id}'] = answer
        session.modified = True
        
        print(f"Saved to session: {answers_key}")
        print(f"Total saved answers: {len(session[answers_key])}")
        print("=== SAVE SUCCESS ===")
        
        return jsonify({'success': True, 'message': 'Answer saved'})
    
    except Exception as e:
        print(f"=== SAVE ERROR ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'success': False, 'message': f'Save failed: {str(e)}'}), 500

@student_bp.route('/exam/<exam_id>/auto_submit', methods=['POST'])
@login_required
@student_required
def auto_submit_exam(exam_id):
    try:
        data = request.get_json() or {}
        reason = data.get('reason', 'Excessive proctoring violations')
        warning_count = data.get('warning_count', 0)
        violation_history = data.get('violation_history', [])
        
        logger.info(f"Auto-submitting exam {exam_id} for user {current_user.id} due to: {reason}")
        
        exam = MongoManager.get_exam_by_id(exam_id)
        if not exam:
            return jsonify({'success': False, 'message': 'Exam not found'}), 404
        
        existing_submission = MongoManager.get_student_submission_for_exam(current_user.id, exam_id)
        if existing_submission:
            return jsonify({'success': False, 'message': 'Exam already submitted'}), 400
        
        answers_key = f'exam_{exam_id}_answers'
        session_answers = session.get(answers_key, {})
        
        # Convert session answers to simple format: question_id -> answer_text
        user_answers = {}
        for key, value in session_answers.items():
            if key.startswith('answer_'):
                question_id = key.replace('answer_', '')
                user_answers[question_id] = str(value).strip() if value else ''
        
        exam_session_key = f'exam_{exam_id}_questions'
        session_questions = session.get(exam_session_key, exam.questions)
        
        start_time_str = session.get(f'exam_{exam_id}_start_time')
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
        else:
            start_time = datetime.utcnow() - timedelta(minutes=30)
        
        end_time = datetime.utcnow()
        time_taken = int((end_time - start_time).total_seconds())
        
        score = None
        is_graded = False
        
        if exam.exam_format == 'objective':
            score, is_graded = auto_grade_objective_exam(exam, user_answers, session_questions)
        
        # Get violation history from session if not provided
        if not violation_history:
            violation_history_key = f'violation_history_{current_user.id}_{exam_id}'
            violation_history = session.get(violation_history_key, [])
        
        # Get final warning count
        final_warning_count = warning_count or get_warning_count(current_user.id, exam_id)
        
        submission_data = {
            'student_id': current_user.id,
            'exam_id': exam_id,
            'answers': user_answers,
            'submitted_at': end_time,
            'started_at': start_time,
            'time_taken': time_taken,
            'score': score,
            'is_graded': is_graded,
            'warning_count_at_submission': final_warning_count,
            'questions_answered': len([k for k, v in user_answers.items() if v and str(v).strip()]),
            'auto_submitted': True,
            'auto_submit_reason': reason,
            'submission_reason': f"{reason} ({final_warning_count} violations)",
            'violation_details': violation_history  # Store detailed violation history
        }
        
        submission = Submission(**submission_data)
        submission_id = MongoManager.create_submission(submission)
        
        if submission_id:
            # Clean up session data
            session.pop(exam_session_key, None)
            session.pop(answers_key, None)
            session.pop(f'exam_{exam_id}_start_time', None)
            session.pop(f'warnings_{current_user.id}_{exam_id}', None)
            session.pop(f'violation_history_{current_user.id}_{exam_id}', None)
            session.modified = True
            
            reset_warning_count(current_user.id, exam_id)
            
            logger.info(f"Exam {exam_id} auto-submitted for user {current_user.id} with {final_warning_count} violations")
            
            return jsonify({
                'success': True,
                'message': f'Exam auto-submitted due to {final_warning_count} proctoring violations',
                'submission_id': str(submission_id),
                'redirect_url': url_for('student.submission_result', submission_id=submission_id),
                'warning_count': final_warning_count,
                'violation_count': len(violation_history)
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to auto-submit exam'}), 500
    
    except Exception as e:
        logger.error(f"Error auto-submitting exam {exam_id} for user {current_user.id}: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred during auto-submission'}), 500

@student_bp.route('/exam/<exam_id>/procrastination-check', methods=['POST'])
@csrf_exempt
@login_required
@student_required
def procrastination_check(exam_id):
    """Enhanced procrastination detection with face analysis, copy/paste blocking, and tab switching detection"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        user_id = current_user.id
        current_time = time.time()
        
        # Get current warning count
        current_warnings = get_warning_count(user_id, exam_id)
        
        # Log incoming violation data for debugging
        total_violations_reported = data.get('totalViolations', 0)
        logger.info(f" Procrastination Check - User: {user_id}, Exam: {exam_id}")
        logger.info(f" Current warnings: {current_warnings}/20")
        logger.info(f" Total violations reported: {total_violations_reported}")
        logger.info(f" Face data: {data.get('faceData', {})}")
        logger.info(f" Browser data: {data.get('browserData', {})}")
        logger.info(f" Input data: {data.get('inputData', {})}")
        
        # Initialize violation flags
        violations = []
        should_increment_warning = False
        auto_submit = False
        
        # 1. FACE DETECTION VIOLATIONS
        face_data = data.get('faceData', {})
        num_faces = face_data.get('numFaces', 0)
        pose_direction = face_data.get('poseDirection', 'unknown')
        camera_blocked = face_data.get('cameraBlocked', False)
        
        logger.info(f" Face Detection Check: faces={num_faces}, pose={pose_direction}, blocked={camera_blocked}")
        
        # No face detected
        if num_faces == 0 or camera_blocked:
            violations.append({
                'type': 'CAMERA_ISSUE',
                'message': 'No face detected or camera blocked',
                'severity': 'HIGH',
                'timestamp': current_time
            })
            should_increment_warning = True
            logger.info(" VIOLATION: No face detected - incrementing warning")
        
        # Multiple faces detected
        elif num_faces > 1:
            violations.append({
                'type': 'MULTIPLE_FACES',
                'message': f'{num_faces} faces detected - only one person allowed',
                'severity': 'HIGH',
                'timestamp': current_time
            })
            should_increment_warning = True
            logger.info(f" VIOLATION: {num_faces} faces detected - incrementing warning")
        
        # Looking away detection (only left/right) - MOST COMMON VIOLATION
        elif pose_direction in ['left', 'right']:
            violations.append({
                'type': 'LOOKING_AWAY',
                'message': f'Looking {pose_direction} - please focus on exam',
                'severity': 'HIGH',  # Changed to HIGH for better counting
                'timestamp': current_time
            })
            should_increment_warning = True
            logger.info(f" VIOLATION: Looking {pose_direction} - incrementing warning")
        
        # 2. BROWSER ACTIVITY VIOLATIONS
        browser_data = data.get('browserData', {})
        
        # Tab switching detection
        if browser_data.get('tabSwitched', False):
            violations.append({
                'type': 'TAB_SWITCH',
                'message': 'Tab switching detected',
                'severity': 'HIGH',
                'timestamp': current_time
            })
            should_increment_warning = True
        
        # Window focus lost
        if browser_data.get('windowFocusLost', False):
            violations.append({
                'type': 'WINDOW_FOCUS',
                'message': 'Window focus lost',
                'severity': 'MEDIUM',
                'timestamp': current_time
            })
            should_increment_warning = True
        
        # 3. INPUT ACTIVITY VIOLATIONS
        input_data = data.get('inputData', {})
        
        # Copy/paste detection
        if input_data.get('copyAttempt', False):
            violations.append({
                'type': 'COPY_ATTEMPT',
                'message': 'Copy operation blocked',
                'severity': 'HIGH',
                'timestamp': current_time
            })
            should_increment_warning = True
        
        if input_data.get('pasteAttempt', False):
            violations.append({
                'type': 'PASTE_ATTEMPT',
                'message': 'Paste operation blocked',
                'severity': 'HIGH',
                'timestamp': current_time
            })
            should_increment_warning = True
        
        # Text selection blocking
        if input_data.get('textSelectionAttempt', False):
            violations.append({
                'type': 'TEXT_SELECTION',
                'message': 'Text selection blocked',
                'severity': 'MEDIUM',
                'timestamp': current_time
            })
            should_increment_warning = True  # NOW COUNT text selection as violation
        
        # Suspicious key combinations
        if input_data.get('suspiciousKeys', False):
            violations.append({
                'type': 'SUSPICIOUS_KEYS',
                'message': 'Suspicious key combination detected',
                'severity': 'MEDIUM',
                'timestamp': current_time
            })
            should_increment_warning = True
        
        # 4. WARNING MANAGEMENT
        new_warning_count = current_warnings
        if should_increment_warning:
            logger.info(f" INCREMENTING WARNING: Current={current_warnings}, Violations={len(violations)}")
            new_warning_count = increment_warning_count(user_id, exam_id)
            logger.info(f" NEW WARNING COUNT: {new_warning_count}/20")
        
        # 5. AUTO-SUBMIT CHECK (20 warnings threshold)
        if new_warning_count >= 20:
            auto_submit = True
            violations.append({
                'type': 'AUTO_SUBMIT',
                'message': f'Maximum warnings reached ({new_warning_count}/20) - Auto-submitting exam',
                'severity': 'CRITICAL',
                'timestamp': current_time
            })
        
        # 6. STORE VIOLATIONS IN DATABASE
        if violations:
            from app.mongodb import mongo
            violation_record = {
                'user_id': user_id,
                'exam_id': exam_id,
                'violations': violations,
                'warning_count': new_warning_count,
                'timestamp': datetime.utcnow(),
                'auto_submit_triggered': auto_submit
            }
            mongo.db.procrastination_violations.insert_one(violation_record)
        
        # 7. PREPARE RESPONSE
        response_data = {
            'success': True,
            'violations': violations,
            'warningCount': new_warning_count,
            'maxWarnings': 20,
            'autoSubmit': auto_submit,
            'message': f'Violations detected: {len(violations)}' if violations else 'No violations detected'
        }
        
        logger.info(f" RESPONSE DATA: {response_data}")
        logger.info(f" Final Warning Count: {new_warning_count}/20")
        
        # Add specific messages for different violation types
        if auto_submit:
            response_data['autoSubmitMessage'] = 'Maximum violations reached. Exam will be auto-submitted.'
        elif violations:
            high_severity_violations = [v for v in violations if v['severity'] == 'HIGH']
            if high_severity_violations:
                response_data['alertMessage'] = f"Warning {new_warning_count}/20: {high_severity_violations[0]['message']}"
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in procrastination check: {str(e)}")
        return jsonify({'success': False, 'message': 'Error processing procrastination check'}), 500

@student_bp.route('/increment_warning', methods=['POST'])
@csrf_exempt
@login_required
@student_required
def increment_warning():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        exam_id = data.get('exam_id')
        if not exam_id:
            return jsonify({'success': False, 'message': 'Exam ID required'}), 400
        
        if data.get('verified', False):
            new_count = increment_warning_count(current_user.id, exam_id)
            logger.info(f"Warning incremented for user {current_user.id}, exam {exam_id}. New count: {new_count}")
            
            return jsonify({
                'success': True,
                'warning_count': new_count,
                'action': 'warning' if new_count < 20 else 'auto_submit'
            })
        else:
            current_count = get_warning_count(current_user.id, exam_id)
            return jsonify({
                'success': True,
                'warning_count': current_count,
                'action': 'ignored'
            })
    
    except Exception as e:
        logger.error(f"Error incrementing warning: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to increment warning'}), 500

def auto_grade_objective_exam(exam, user_answers, session_questions=None):
    try:
        print(f"=== AUTO GRADE DEBUG START ===")
        print(f"Exam format: {exam.exam_format}")
        print(f"User answers: {user_answers}")
        
        if exam.exam_format != 'objective':
            return None, False
        
        questions_to_grade = session_questions if session_questions else exam.questions
        total_questions = len(questions_to_grade)
        print(f"Total questions to grade: {total_questions}")
        
        if total_questions == 0:
            return 0.0, True
        
        correct_answers = 0
        total_marks = 0
        earned_marks = 0
        
        for question in questions_to_grade:
            # Handle both dict and Question object formats
            if isinstance(question, dict):
                question_id = str(question.get('_id') or question.get('id', ''))
                question_marks = exam.get_marks_per_question() or (exam.total_marks / len(questions_to_grade) if len(questions_to_grade) > 0 else 0)
                correct_answer = question.get('correct_answer')
                question_type = question.get('question_type', 'multiple_choice')
            else:
                question_id = str(question.id)
                question_marks = exam.get_marks_per_question() or (exam.total_marks / len(questions_to_grade) if len(questions_to_grade) > 0 else 0)
                correct_answer = getattr(question, 'correct_answer', None)
                question_type = getattr(question, 'question_type', 'multiple_choice')
            
            print(f"Processing question ID: {question_id}")
            total_marks += question_marks
            
            # Get user answer using question ID
            user_answer = user_answers.get(question_id)
            print(f"User answer for question {question_id}: {user_answer}")
            
            if user_answer is None or str(user_answer).strip() == '':
                print(f"No answer provided for question {question_id}")
                continue
            
            print(f"Correct answer for question {question_id}: {correct_answer}")
            
            if correct_answer is None:
                print(f"No correct answer configured for question {question_id}")
                continue
            
            # Compare answers
            # Compare answers
            if question_type == 'multiple_choice':
                try:
                    # Attempt to convert to float/int for numeric comparison (handles '1' vs 1)
                    ua_val = float(user_answer)
                    ca_val = float(correct_answer)
                    is_correct = ua_val == ca_val
                except (ValueError, TypeError):
                    # Fallback to string comparison
                    is_correct = str(user_answer).strip() == str(correct_answer).strip()
                
                if is_correct:
                    correct_answers += 1
                    earned_marks += question_marks
                    print(f" Correct answer for question {question_id}")
                else:
                    print(f" Wrong answer for question {question_id} (User: {user_answer}, Correct: {correct_answer})")
            else:
                if str(user_answer).strip().lower() == str(correct_answer).strip().lower():
                    correct_answers += 1
                    earned_marks += question_marks
                    print(f" Correct answer for question {question_id}")
                else:
                    print(f" Wrong answer for question {question_id}")
        
        # Calculate final score
        final_score = earned_marks  # Return actual marks earned, not percentage
        
        print(f"=== AUTO GRADE RESULTS ===")
        print(f"Correct answers: {correct_answers}/{total_questions}")
        print(f"Marks earned: {earned_marks}/{total_marks}")
        print(f"Final score: {final_score}")
        print(f"=== AUTO GRADE DEBUG END ===")
        
        return final_score, True
    
    except Exception as e:
        print(f"=== AUTO GRADE ERROR ===")
        print(f"Error in auto_grade_objective_exam: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

@student_bp.route('/exam/<exam_id>/violation', methods=['POST'])
@csrf_exempt
@login_required
@student_required
def report_violation(exam_id):
    """Handle violation reports from the advanced proctoring system"""
    try:
        data = request.get_json()
        violation_type = data.get('violation_type')
        message = data.get('message')
        timestamp = data.get('timestamp')
        warning_count = data.get('warning_count', 0)
        
        # Log the violation
        logger.info(f"Violation reported for exam {exam_id}, user {current_user.id}: {violation_type} - {message}")
        
        # Update warning count in session/database
        warning_key = f'warnings_{current_user.id}_{exam_id}'
        if warning_key not in session:
            session[warning_key] = 0
        
        session[warning_key] = warning_count
        session.modified = True
        
        # Store violation details (you can expand this to store in database)
        violation_history_key = f'violation_history_{current_user.id}_{exam_id}'
        if violation_history_key not in session:
            session[violation_history_key] = []
        
        session[violation_history_key].append({
            'type': violation_type,
            'message': message,
            'timestamp': timestamp,
            'warning_count': warning_count
        })
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': 'Violation recorded',
            'current_warnings': warning_count
        })
        
    except Exception as e:
        logger.error(f"Error reporting violation for exam {exam_id}: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to record violation'}), 500

@student_bp.route('/debug/test-save', methods=['GET', 'POST'])
@csrf_exempt
@login_required
@student_required
def debug_test_save():
    """Debug endpoint to test save functionality"""
    try:
        debug_info = {
            'method': request.method,
            'user_id': str(current_user.id),
            'user_name': current_user.name,
            'session_keys': list(session.keys()),
            'request_headers': dict(request.headers),
            'request_args': dict(request.args),
            'request_form': dict(request.form),
            'content_type': request.content_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if request.method == 'POST':
            if request.content_type == 'application/json':
                json_data = request.get_json()
                debug_info['json_data'] = json_data
            else:
                debug_info['form_data'] = dict(request.form)
        
        print(f"=== DEBUG TEST SAVE ===")
        print(f"Debug info: {debug_info}")
        
        return jsonify({
            'success': True,
            'message': 'Debug endpoint working',
            'debug_info': debug_info
        })
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(f"=== DEBUG TEST ERROR ===")
        print(f"Error info: {error_info}")
        
        return jsonify({
            'success': False,
            'message': 'Debug endpoint error',
            'error_info': error_info
        }), 500

@student_bp.route('/debug/test-page')
def debug_test_page():
    """Debug test page to verify F12 access and console logging"""
    return render_template('debug_test.html')

@student_bp.route('/debug/test-questions/<exam_id>')
@login_required
@student_required
def debug_test_questions(exam_id):
    """Debug endpoint to test random question generation"""
    try:
        exam = MongoManager.get_exam_by_id(exam_id)
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404
        
        all_questions = exam.questions
        num_questions_to_display = exam.num_questions_to_display or len(all_questions)
        num_questions_to_display = min(num_questions_to_display, len(all_questions))
        
        # Generate random questions for this student
        random.seed(f"{current_user.id}_{exam_id}")
        if num_questions_to_display < len(all_questions):
            selected_questions = random.sample(all_questions, num_questions_to_display)
        else:
            selected_questions = all_questions
        
        # Extract question IDs and texts for debugging
        question_info = []
        for i, q in enumerate(selected_questions):
            if isinstance(q, dict):
                question_info.append({
                    'index': i + 1,
                    'id': str(q.get('_id') or q.get('id', '')),
                    'text': q.get('question_text', '')[:100] + '...' if len(q.get('question_text', '')) > 100 else q.get('question_text', ''),
                    'type': q.get('question_type', 'unknown'),
                    'marks': q.get('marks', 1)
                })
            else:
                question_info.append({
                    'index': i + 1,
                    'id': str(q.id) if hasattr(q, 'id') else 'unknown',
                    'text': (q.question_text[:100] + '...' if len(q.question_text) > 100 else q.question_text) if hasattr(q, 'question_text') else 'No text',
                    'type': getattr(q, 'question_type', 'unknown'),
                    'marks': getattr(q, 'marks', 1)
                })
        
        return jsonify({
            'student_id': str(current_user.id),
            'exam_id': exam_id,
            'exam_title': exam.title,
            'total_questions_in_exam': len(all_questions),
            'questions_to_display': num_questions_to_display,
            'selected_questions': question_info,
            'seed_used': f"{current_user.id}_{exam_id}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@student_bp.route('/debug/test-submit', methods=['POST'])
@csrf_exempt
@login_required
@student_required
def debug_test_submit():
    """Debug endpoint to test submission functionality"""
    try:
        print(f"\n=== DEBUG TEST SUBMIT ===")
        print(f"Method: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"User: {current_user.id}")
        
        # Get data
        if request.content_type == 'application/json':
            data = request.get_json()
            print(f"JSON Data: {data}")
        else:
            data = dict(request.form)
            print(f"Form Data: {data}")
        
        # Test answer processing
        test_answers = data.get('answers', {})
        print(f"Test answers: {test_answers}")
        print(f"Number of answers: {len(test_answers)}")
        
        # Create a simple test submission
        test_submission_data = {
            'student_id': current_user.id,
            'exam_id': '507f1f77bcf86cd799439011',  # Dummy exam ID
            'answers': test_answers,
            'submitted_at': datetime.utcnow(),
            'time_taken': 300,
            'score': None,
            'is_graded': False,
            'warning_count': 0,
            'questions_answered': len([k for k, v in test_answers.items() if v and str(v).strip()])
        }
        
        print(f"Test submission data: {test_submission_data}")
        
        return jsonify({
            'success': True,
            'message': 'Debug test successful',
            'received_answers': test_answers,
            'processed_data': test_submission_data,
            'user_id': str(current_user.id)
        })
        
    except Exception as e:
        print(f"Debug test error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
