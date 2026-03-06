from datetime import datetime
from bson import ObjectId
from .question import Question

# These classes are now just data containers or can be removed if not used directly.
# All database operations should use MongoManager from app/mongodb.py

class Exam:
    def __init__(self, id=None, title=None, description=None, duration=None, 
                 teacher_id=None, start_time=None, end_time=None, questions=None,
                 is_active=True, exam_type='scheduled', exam_format='objective',
                 total_marks=0, num_questions_to_display=None, class_id=None):
        self.id = id
        self.title = title
        self.description = description
        self.duration = duration
        self.teacher_id = teacher_id
        self.start_time = start_time
        self.end_time = end_time
        self.questions = questions or []
        self.is_active = is_active
        self.exam_type = exam_type
        self.exam_format = exam_format
        self.total_marks = total_marks
        self.num_questions_to_display = num_questions_to_display
        self.class_id = class_id

    @staticmethod
    def from_db(data):
        if not data:
            return None
        
        try:
            # Convert ObjectId to string for id and teacher_id
            exam_id = str(data.pop('_id')) if '_id' in data else data.get('id')
            teacher_id = str(data['teacher_id']) if 'teacher_id' in data else None
            class_id = str(data['class_id']) if 'class_id' in data and data['class_id'] else None
            
            # Handle questions separately to avoid conversion issues
            questions_data = data.get('questions', [])
            processed_questions = []
            
            for q in questions_data:
                if isinstance(q, dict):
                    # Keep as dict for better compatibility
                    processed_questions.append(q)
                else:
                    # Convert Question object to dict if needed
                    try:
                        processed_questions.append(q.to_dict() if hasattr(q, 'to_dict') else q)
                    except Exception:
                        processed_questions.append(q)
            
            # Create Exam instance
            exam = Exam(
                id=exam_id,
                title=data.get('title'),
                description=data.get('description'),
                duration=data.get('duration'),
                teacher_id=teacher_id,
                start_time=data.get('start_time'),
                end_time=data.get('end_time'),
                questions=processed_questions,  # Use processed questions
                is_active=data.get('is_active', True),
                exam_type=data.get('exam_type', 'scheduled'),
                exam_format=data.get('exam_format', 'objective'),
                total_marks=data.get('total_marks', 0),
                num_questions_to_display=data.get('num_questions_to_display'),
                class_id=class_id
            )
            return exam
        except Exception as e:
            print(f"Error creating Exam from database data: {str(e)}")
            return None

    def to_dict(self):
        """Convert exam to dictionary format"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'duration': self.duration,
            'teacher_id': self.teacher_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'questions': self.questions,
            'is_active': self.is_active,
            'exam_type': self.exam_type,
            'exam_format': self.exam_format,
            'total_marks': self.total_marks,
            'num_questions_to_display': self.num_questions_to_display
        }

    def is_active_for_student(self):
        """Check if exam is active for student"""
        try:
            if not self.is_active:
                return False
                
            now = datetime.utcnow()
            
            # Check if dates are valid
            if not self.end_time:
                return False
            
            # Ensure dates are datetime objects
            end_time = self.end_time
            if isinstance(end_time, str):
                end_time = parse_datetime(end_time)
            if not end_time:
                return False
            
            if self.exam_type == 'self_paced':
                # For self-paced exams, only check end time
                return now <= end_time
            else:
                # For scheduled exams, check both start and end time
                start_time = self.start_time
                if isinstance(start_time, str):
                    start_time = parse_datetime(start_time)
                if not start_time:
                    return False
                    
                return start_time <= now <= end_time
        except Exception as e:
            print(f"Error in is_active_for_student: {str(e)}")
            return False
    
    def is_objective(self):
        """Check if exam is objective type"""
        return self.exam_format == 'objective'
    
    def is_subjective(self):
        """Check if exam is subjective type"""
        return self.exam_format == 'subjective'

    def get_total_marks(self):
        """Calculate total marks for the exam"""
        if not self.questions:
            return 0
        return sum(q.get('marks', 0) if isinstance(q, dict) else q.marks for q in self.questions)

    def get_marks_per_question(self):
        """Calculate marks per question based on total marks and questions to display"""
        if not self.num_questions_to_display or self.num_questions_to_display <= 0:
            return 0
        return self.total_marks / self.num_questions_to_display

    def get_question_count(self):
        """Get total number of questions"""
        return len(self.questions) if self.questions else 0
    
    def get_formatted_duration(self):
        """Get formatted duration string"""
        if not self.duration:
            return "No time limit"
        hours = self.duration // 60
        minutes = self.duration % 60
        if hours and minutes:
            return f"{hours} hour{'s' if hours > 1 else ''} {minutes} minute{'s' if minutes > 1 else ''}"
        elif hours:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            return f"{minutes} minute{'s' if minutes > 1 else ''}"
    
    def get_status(self):
        """Get exam status"""
        if not self.is_active:
            return "Inactive"
            
        now = datetime.utcnow()
        
        if self.exam_type == 'self_paced':
            if now > self.end_time:
                return "Expired"
            return "Active"
        else:
            if now < self.start_time:
                return "Upcoming"
            elif now > self.end_time:
                return "Completed"
            else:
                return "In Progress"
    
    def validate_questions(self):
        """Validate exam questions"""
        if not self.questions:
            return False
            
        try:
            for q in self.questions:
                if isinstance(q, dict):
                    # Check required fields
                    if not q.get('question_text'):
                        return False
                    if q.get('question_type') == 'multiple_choice' and not q.get('options'):
                        return False
                else:
                    # Question object validation
                    if not q.question_text:
                        return False
                    if q.question_type == 'multiple_choice' and not q.options:
                        return False
            return True
        except Exception:
            return False

def parse_datetime(dt_str):
    """Parse datetime string to datetime object"""
    if not dt_str:
        return None
    if isinstance(dt_str, datetime):
        return dt_str
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception:
        return None

class Submission:
    pass 

