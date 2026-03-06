from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User
from app.mongodb import MongoManager, mongo
from flask_wtf import FlaskForm

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/')
def index():
    return render_template('index.html')

@auth_bp.route('/test_db')
def test_db():
    try:
        # Test MongoDB connection
        mongo.db.command('ping')
        connection_status = "MongoDB connection successful"
    except Exception as e:
        connection_status = f"MongoDB connection failed: {str(e)}"

    try:
        # Get all users
        users = list(mongo.db.users.find())
        users_info = []
        for user in users:
            # Remove password hash from display
            user_info = {
                'id': str(user['_id']),
                'username': user.get('username'),
                'email': user.get('email'),
                'role': user.get('role'),
                'is_active': user.get('is_active', True)
            }
            users_info.append(user_info)
    except Exception as e:
        users_info = f"Error getting users: {str(e)}"

    return {
        'connection_status': connection_status,
        'users': users_info
    }

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('teacher.dashboard' if current_user.is_teacher() else 'student.dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        if not all([email, password, role]):
            flash('All fields are required.', 'error')
            return render_template('auth/login.html')
        
        user = MongoManager.get_user_by_email(email)
        
        if user and user.role == role and user.check_password(password):
            # Block students who haven't been assigned to a class yet
            if user.is_student():
                class_data = mongo.db.classes.find_one({'students': user._id})
                if not class_data:
                    # Don't log them in — redirect back with error flag
                    return redirect(url_for('auth.login', no_class=1))

            login_user(user)
            return redirect(url_for('teacher.dashboard' if user.is_teacher() else 'student.dashboard'))
        
        flash('Invalid email or password.', 'error')
    
    no_class = request.args.get('no_class')
    return render_template('auth/login.html', no_class=no_class)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('teacher.dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.', 'error')
            return render_template('auth/register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('auth/register.html')
        
        if MongoManager.get_user_by_email(email):
            flash('Email already registered.', 'error')
            return render_template('auth/register.html')
        
        user = MongoManager.create_user(
            username=username,
            email=email,
            password=password,
            role='teacher'
        )
        
        if user:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))
        
        flash('Registration failed. Please try again.', 'error')
    
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
