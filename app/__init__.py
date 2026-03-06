from flask import Flask, session, request
from flask_login import LoginManager
# from flask_wtf.csrf import CSRFProtect  # Disabled for procrastination detection
from flask_session import Session
from flask_socketio import SocketIO
from .mongodb import mongo, init_mongodb, MongoManager
from .config.mongodb import MongoConfig
from .config import Config
from .models import User
import os
import jinja2
from markupsafe import Markup

login_manager = LoginManager()
# csrf = CSRFProtect()  # Disabled
sess = Session()
socketio = SocketIO()

@login_manager.user_loader
def load_user(user_id):
    return MongoManager.get_user_with_class_info(user_id)

def create_app():
    app = Flask(__name__)
    
    # Set secret key
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
    
    # Load configuration
    app.config.from_object(Config)
    
    # Set proctoring configuration
    app.config['CAMERA_INDEX'] = 0
    app.config['FRAME_WIDTH'] = 640
    app.config['FRAME_HEIGHT'] = 480
    app.config['FPS'] = 24
    app.config['PROCTORING_THRESHOLDS'] = {
        'MAX_HEAD_YAW': 25,          # degrees
        'MAX_HEAD_PITCH': 15,        # degrees
        'MAX_HEAD_ROLL': 15,         # degrees
        'MAX_ABSENCE_TIME': 5,       # seconds
        'FACE_CONFIDENCE': 0.8,
        'NECK_MOVEMENT_THRESHOLD': 0.35,
        'FACE_DETECTION_FREQUENCY': 5, # frames
        'EYE_ASPECT_RATIO': 0.2,      # closed eye threshold
        # New procrastination monitoring thresholds
        'MOVEMENT_DURATION_THRESHOLD': 2.0,  # Seconds of continuous movement to trigger alert
        'MULTIPLE_PEOPLE_DURATION_THRESHOLD': 2.0,  # Seconds of multiple people to trigger alert
        'ABSENCE_DURATION_THRESHOLD': 5.0,  # Seconds of absence to trigger alert
    }
    
    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Initialize MongoDB
    if not init_mongodb(app):
        print("WARNING: Failed to initialize MongoDB connection. Check MONGO_URI env variable.")
    
    # Configure MongoDB indexes (non-fatal)
    with app.app_context():
        if not MongoConfig.init_indexes(mongo):
            print("WARNING: Failed to initialize MongoDB indexes. App will start but may have DB issues.")
    
    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Initialize session handling (uses cookie-based sessions for production speed)
    sess.init_app(app)
    
    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins="*", async_mode='gevent')
    
    # Register SocketIO event handlers
    @socketio.on('exam_warning')
    def handle_exam_warning(data):
        print(f"Exam warning received: {data}")
        # Broadcast warning to all connected clients
        socketio.emit('proctor_alert', {
            'warnings': [{
                'type': data.get('type', 'general'),
                'message': data.get('message', 'Warning detected'),
                'severity': data.get('severity', 'medium'),
                'timestamp': data.get('timestamp'),
                'exam_id': data.get('exam_id')
            }]
        })
    
    # Add context processor for csrf_token (disabled for procrastination detection)
    @app.context_processor
    def inject_csrf_token():
        """Provide a dummy csrf_token function since CSRFProtect is disabled"""
        def csrf_token():
            return 'csrf-disabled-for-procrastination-detection'
        return dict(csrf_token=csrf_token)
    
    # Add template filters
    @app.template_filter('nl2br')
    def nl2br_filter(s):
        if s:
            return Markup(s.replace('\n', '<br>'))
        return ''
    
    @app.template_filter('safe_string')
    def safe_string_filter(s):
        """Safely convert any value to a string and escape problematic characters"""
        try:
            if s is None:
                return ''
            # Convert to string and escape quotes and backslashes
            s_str = str(s)
            # Replace problematic characters that could break JSON
            s_str = s_str.replace('\\', '\\\\')  # Escape backslashes
            s_str = s_str.replace('"', '\\"')    # Escape double quotes
            s_str = s_str.replace("'", "\\'")    # Escape single quotes
            s_str = s_str.replace('\n', '\\n')   # Escape newlines
            s_str = s_str.replace('\r', '\\r')   # Escape carriage returns
            s_str = s_str.replace('\t', '\\t')   # Escape tabs
            return s_str
        except Exception:
            return ''
    
    # Register blueprints
    from .routes.auth import auth_bp
    from .routes.teacher import teacher_bp
    from .routes.student import student_bp
    from .routes.proctor import proctor_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(proctor_bp, url_prefix='/proctor')
    
    # Ensure session is configured
    @app.before_request
    def make_session_permanent():
        session.permanent = True
    
    return app 