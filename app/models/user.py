from datetime import datetime
from flask_login import UserMixin
from bson import ObjectId
from werkzeug.security import check_password_hash, generate_password_hash
import hashlib
import base64
import hmac

class User(UserMixin):
    def __init__(self, id, username, email, role, password_hash=None, is_active=True, 
                 student_id=None, phone_number=None, college_name=None, created_at=None):
        self._id = ObjectId(id) if isinstance(id, str) else id
        self.id = str(self._id)  # Convert ObjectId to string for Flask-Login
        self.username = username
        self.email = email
        self.role = role
        self.password_hash = password_hash
        self._is_active = is_active
        self.student_id = student_id
        self.phone_number = phone_number
        self.college_name = college_name
        self.created_at = created_at or datetime.utcnow()
        self.class_info = None  # Add class_info attribute

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        self._is_active = value

    @property
    def class_name(self):
        """Get the student's class name"""
        return self.class_info['name'] if self.class_info else 'Not Assigned'

    @property
    def class_id(self):
        """Get the student's class ID"""
        return self.class_info['id'] if self.class_info else None

    @staticmethod
    def from_db(user_data):
        """Create a User instance from database data"""
        if not user_data:
            return None
            
        try:
            return User(
                id=user_data['_id'],
                username=user_data['username'],
                email=user_data['email'],
                role=user_data['role'],
                password_hash=user_data.get('password_hash'),
                is_active=user_data.get('is_active', True),
                student_id=user_data.get('student_id'),
                phone_number=user_data.get('phone_number'),
                college_name=user_data.get('college_name'),
                created_at=user_data.get('created_at')
            )
        except Exception as e:
            print(f"Error creating user from DB data: {str(e)}")
            print(f"User data: {user_data}")
            return None

    def check_password(self, password):
        """Check if the provided password matches the stored hash.
        Handles both scrypt and pbkdf2 hashes."""
        if not self.password_hash or not password:
            print("No password hash found or empty password provided")
            return False
        
        try:
            # Check if this is a scrypt hash that needs migration
            if self.password_hash.startswith('scrypt:'):
                print("Checking scrypt password")
                return self._check_scrypt_password(password)
            # Otherwise use standard Werkzeug password check
            print("Checking pbkdf2 password")
            result = check_password_hash(self.password_hash, password)
            print(f"Password check result: {result}")
            return result
        except Exception as e:
            print(f"Password verification error: {str(e)}")
            return False

    def _check_scrypt_password(self, password):
        """Legacy method to check scrypt passwords"""
        try:
            method, salt, hashval = self.password_hash.split('$', 2)
            if not method.startswith('scrypt:'):
                print("Invalid scrypt hash format")
                return False
                
            print(f"Scrypt parameters - Method: {method}")
            # Parse scrypt parameters
            _, n, r, p = method.split(':')
            n, r, p = int(n), int(r), int(p)
            
            # Calculate scrypt hash
            password_bytes = password.encode()
            salt_bytes = salt.encode()
            maxmem = 132 * n * r * p
            
            calculated_hash = hashlib.scrypt(
                password_bytes, 
                salt=salt_bytes,
                n=n, r=r, p=p,
                maxmem=maxmem
            ).hex()
            
            result = hmac.compare_digest(calculated_hash, hashval)
            print(f"Scrypt verification result: {result}")
            return result
        except Exception as e:
            print(f"Scrypt password check error: {str(e)}")
            return False

    @staticmethod
    def set_password(password):
        """Generate password hash using pbkdf2:sha256"""
        if not password:
            raise ValueError("Password cannot be empty")
        return generate_password_hash(password, method='pbkdf2:sha256')

    def needs_password_migration(self):
        """Check if the password hash needs to be migrated"""
        return self.password_hash and self.password_hash.startswith('scrypt:')

    def is_teacher(self):
        return self.role == 'teacher'

    def is_student(self):
        return self.role == 'student'

    def is_admin(self):
        return self.role == 'admin'

    def to_dict(self):
        """Convert User object to dictionary for database storage"""
        data = {
            '_id': self._id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'password_hash': self.password_hash,
            'is_active': self._is_active,
            'created_at': self.created_at
        }
        
        # Add student-specific fields if they exist
        if self.student_id:
            data['student_id'] = self.student_id
        if self.phone_number:
            data['phone_number'] = self.phone_number
        if self.college_name:
            data['college_name'] = self.college_name
            
        return data

    def get_id(self):
        return str(self._id)

    def __repr__(self):
        return f'<User {self.username}>' 