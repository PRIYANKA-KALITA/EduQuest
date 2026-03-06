import os
from flask_pymongo import PyMongo
from bson import ObjectId
from datetime import datetime
from .models import User, Exam, Question, Submission, Class
from pymongo.errors import ConnectionFailure
import traceback

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize PyMongo
mongo = PyMongo()

def init_mongodb(app):
    try:
        uri = os.getenv("MONGO_URI")
        app.config["MONGO_URI"] = uri
        mongo.init_app(app)
        with app.app_context():
            # Test connection and log database name
            mongo.db.command('ping')
            print(f"Connected to MongoDB: {mongo.db.name}")
        return True
    except Exception as e:
        print(f"MongoDB Connection Error: {str(e)}")
        return False

class MongoManager:
    @staticmethod
    def init_app(app):
        mongo.init_app(app)

    @staticmethod
    def delete_all_questions(exam_id):
        """Delete all questions from an exam."""
        try:
            # Convert exam_id to ObjectId if it's a string
            if isinstance(exam_id, str):
                exam_id = ObjectId(exam_id)
            
            # Update the exam to remove all questions
            result = mongo.db.exams.update_one(
                {'_id': exam_id},
                {'$set': {'questions': []}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error in delete_all_questions: {str(e)}")
            return False

    @staticmethod
    def update_exam_ai_status(exam_id, status, error=None):
        """Update the AI generation status of an exam."""
        try:
            if isinstance(exam_id, str):
                exam_id = ObjectId(exam_id)
            mongo.db.exams.update_one(
                {'_id': exam_id},
                {'$set': {'ai_status': {
                    'status': status,
                    'error': error,
                    'updated_at': datetime.utcnow()
                }}}
            )
            return True
        except Exception as e:
            print(f"Error in update_exam_ai_status: {str(e)}")
            return False

    @staticmethod
    def get_exam_ai_status(exam_id):
        """Get the current AI generation status of an exam."""
        try:
            if isinstance(exam_id, str):
                exam_id = ObjectId(exam_id)
            exam = mongo.db.exams.find_one({'_id': exam_id}, {'ai_status': 1})
            return exam.get('ai_status') if exam else None
        except Exception as e:
            print(f"Error in get_exam_ai_status: {str(e)}")
            return None

    @staticmethod
    def _convert_to_json_serializable(obj):
        """Convert MongoDB document to JSON serializable format."""
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: MongoManager._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MongoManager._convert_to_json_serializable(item) for item in obj]
        return obj

    # User Operations
    @staticmethod
    def count_students():
        """Count the total number of students in the system"""
        return mongo.db.users.count_documents({'role': 'student'})

    @staticmethod
    def count_teacher_students(teacher_id):
        """Count students that belong to this teacher (in classes or created by them)"""
        try:
            teacher_oid = ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id
            # Students in teacher's classes
            teacher_classes = list(mongo.db.classes.find({'teacher_id': teacher_oid}, {'students': 1}))
            student_ids = set()
            for cls in teacher_classes:
                for sid in cls.get('students', []):
                    student_ids.add(str(sid))
            # Also count students created by this teacher but not in any class
            created_by = list(mongo.db.users.find(
                {'role': 'student', 'created_by': teacher_oid}, {'_id': 1}
            ))
            for s in created_by:
                student_ids.add(str(s['_id']))
            return len(student_ids)
        except Exception as e:
            print(f"Error in count_teacher_students: {str(e)}")
            return 0

    @staticmethod
    def get_all_students():
        """Get all students from the database with their class information"""
        try:
            # Get all students
            students_data = list(mongo.db.users.find({'role': 'student'}))
            
            # Get all classes
            classes = list(mongo.db.classes.find())
            
            # Create a map of student IDs to their classes
            student_classes = {}
            for class_obj in classes:
                for student_id in class_obj.get('students', []):
                    if isinstance(student_id, ObjectId):
                        student_classes[str(student_id)] = {
                            'id': str(class_obj['_id']),
                            'name': class_obj['name']
                        }
            
            # Add class information to each student
            students = []
            for student_data in students_data:
                student = User.from_db(student_data)
                student_id = str(student_data['_id'])
                if student_id in student_classes:
                    student.class_info = student_classes[student_id]
                else:
                    student.class_info = None
                students.append(student)
            
            return students
        except Exception as e:
            print(f"Error getting students: {str(e)}")
            return []

    @staticmethod
    def get_teacher_students(teacher_id):
        """Get all students belonging to this teacher — in classes OR created by them (unassigned)"""
        try:
            teacher_oid = ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id
            
            # Get all classes for this teacher
            teacher_classes = list(mongo.db.classes.find({'teacher_id': teacher_oid}))
            
            # Build class map: student_id -> class info
            class_map = {}
            enrolled_ids = set()
            for cls in teacher_classes:
                cls_info = {'id': str(cls['_id']), 'name': cls['name']}
                for sid in cls.get('students', []):
                    enrolled_ids.add(str(sid))
                    class_map[str(sid)] = cls_info
            
            # Also find students created by this teacher but not in any class
            unassigned_data = list(mongo.db.users.find({
                'role': 'student',
                'created_by': teacher_oid
            }))
            unassigned_ids = {str(s['_id']) for s in unassigned_data if str(s['_id']) not in enrolled_ids}
            
            # Fetch enrolled students
            all_ids = [ObjectId(sid) for sid in enrolled_ids] + [ObjectId(sid) for sid in unassigned_ids]
            if not all_ids:
                return []
            
            students_data = list(mongo.db.users.find({
                '_id': {'$in': all_ids},
                'role': 'student'
            }))
            
            students = []
            for student_data in students_data:
                student = User.from_db(student_data)
                student.class_info = class_map.get(str(student_data['_id']))  # None if unassigned
                students.append(student)
            
            return students
        except Exception as e:
            print(f"Error in get_teacher_students: {str(e)}")
            return []

    @staticmethod
    def get_teacher_classes(teacher_id):
        """Get all classes for a teacher"""
        classes_data = mongo.db.classes.find({'teacher_id': ObjectId(teacher_id)})
        return [Class(class_data) for class_data in classes_data]

    @staticmethod
    def count_teacher_exams(teacher_id):
        """Count the total number of exams created by a teacher"""
        return mongo.db.exams.count_documents({'teacher_id': ObjectId(teacher_id)})

    @staticmethod
    def count_active_teacher_exams(teacher_id):
        """Count the number of active exams for a teacher"""
        now = datetime.utcnow()
        # Count exams that are active and haven't expired yet
        return mongo.db.exams.count_documents({
            'teacher_id': ObjectId(teacher_id),
            'is_active': True,
            'end_time': {'$gte': now}  # Only check if end time is in the future
        })

    @staticmethod
    def count_completed_teacher_exams(teacher_id):
        """Count completed exams for a teacher"""
        now = datetime.utcnow()
        return mongo.db.exams.count_documents({
            'teacher_id': ObjectId(teacher_id),
            'end_time': {'$lt': now}
        })

    @staticmethod
    def count_pending_reviews(teacher_id):
        """Count submissions pending review for a teacher's exams"""
        # Get all exam IDs for this teacher
        exam_ids = [exam['_id'] for exam in mongo.db.exams.find({'teacher_id': ObjectId(teacher_id)})]
        
        # Count ungraded submissions for these exams
        # Handle both False and None values for is_graded field
        return mongo.db.submissions.count_documents({
            'exam_id': {'$in': exam_ids},
            '$or': [
                {'is_graded': False},
                {'is_graded': {'$exists': False}},
                {'is_graded': None}
            ]
        })

    @staticmethod
    def get_recent_submissions_for_teacher(teacher_id, limit=5):
        """Get recent submissions for a teacher's exams"""
        # First get the teacher's exam IDs
        exam_ids = [exam['_id'] for exam in mongo.db.exams.find({'teacher_id': ObjectId(teacher_id)})]
        if not exam_ids:
            return []
        
        # Then get recent submissions for these exams
        submissions_data = mongo.db.submissions.find(
            {'exam_id': {'$in': exam_ids}}
        ).sort('submitted_at', -1).limit(limit)
        
        submissions = []
        for sub_data in submissions_data:
            submission = Submission.from_db(sub_data)
            if submission:
                # Add student and exam data for display
                try:
                    student = MongoManager.get_user_with_class_info(submission.student_id)
                    exam = MongoManager.get_exam_by_id(submission.exam_id)
                    submission.student = student
                    submission.exam = exam
                except Exception as e:
                    print(f"Error loading student/exam data for submission {submission.id}: {str(e)}")
                    # Create placeholder objects
                    class PlaceholderStudent:
                        def __init__(self):
                            self.username = "Unknown Student"
                            self.class_name = "N/A"
                            self.student_id = "N/A"
                            self.id = None
                    class PlaceholderExam:
                        def __init__(self):
                            self.title = "Unknown Exam"
                    submission.student = PlaceholderStudent()
                    submission.exam = PlaceholderExam()
                
                submissions.append(submission)
        
        return submissions

    @staticmethod
    def create_user(username, email, password, role):
        try:
            # Generate password hash
            password_hash = User.set_password(password)
            
            user_dict = {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'role': role,
                'created_at': datetime.utcnow(),
                'is_active': True
            }
                    
            result = mongo.db.users.insert_one(user_dict)
            if result.inserted_id:
                user_dict['_id'] = result.inserted_id
                return User.from_db(user_dict)
            return None
        except Exception as e:
            print(f"Create User Error: {str(e)}")
            return None

    @staticmethod
    def get_user_by_id(user_id):
        try:
            user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
            return User.from_db(user_data) if user_data else None
        except Exception as e:
            print(f"Get User Error: {str(e)}")
            return None

    @staticmethod
    def get_user_with_class_info(user_id):
        """Get user by ID with class information populated"""
        try:
            user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
            if not user_data:
                return None
            
            user = User.from_db(user_data)
            if user and user.role == 'student':
                # Get the student's class information
                class_data = mongo.db.classes.find_one({'students': ObjectId(user_id)})
                if class_data:
                    user.class_info = {
                        'id': str(class_data['_id']),
                        'name': class_data['name']
                    }
                else:
                    user.class_info = None
            
            return user
        except Exception as e:
            print(f"Get User with Class Info Error: {str(e)}")
            return None

    @staticmethod
    def get_user_by_email(email):
        try:
            user_data = mongo.db.users.find_one({'email': email})
            return User.from_db(user_data) if user_data else None
        except Exception as e:
            print(f"Get User Error: {str(e)}")
            return None

    @staticmethod
    def get_user_by_username(username):
        try:
            user_data = mongo.db.users.find_one({'username': username})
            return User.from_db(user_data) if user_data else None
        except Exception as e:
            print(f"Get User Error: {str(e)}")
            return None

    @staticmethod
    def get_user_by_student_id(student_id):
        try:
            user_data = mongo.db.users.find_one({'student_id': student_id})
            return User.from_db(user_data) if user_data else None
        except Exception as e:
            print(f"Get User Error: {str(e)}")
            return None

    @staticmethod
    def create_student(student_data):
        """Create a new student user"""
        try:
            # Ensure required fields
            if not all(k in student_data for k in ['username', 'email', 'student_id', 'password']):
                return None, "Missing required fields in student_data"
                
            # Create the MongoDB document
            student_dict = {
                'username': student_data['username'],
                'email': student_data['email'],
                'student_id': student_data['student_id'],
                'phone_number': student_data.get('phone_number', '') or '',
                'college_name': student_data.get('college_name', '') or '',
                'role': 'student',
                'password_hash': User.set_password(student_data['password']),
                'created_at': datetime.utcnow(),
                'is_active': True,
                '_id': ObjectId(),  # Pre-generate the ID
                # Track which teacher created this student
                'created_by': ObjectId(student_data['created_by']) if student_data.get('created_by') else None
            }
            
            # Insert the document
            try:
                print(f"DEBUG: Attempting to insert student into {mongo.db.name}.users")
                result = mongo.db.users.insert_one(student_dict)
                if not result.acknowledged:
                    return None, "MongoDB insert not acknowledged"
                    
                print(f"Student created with ID: {result.inserted_id} in {mongo.db.name}")
                return User.from_db(student_dict), None
            except Exception as db_error:
                error_msg = str(db_error)
                print(f"Database error while creating student: {error_msg}")
                if "duplicate key" in error_msg.lower():
                    if "email" in error_msg: return None, "Email address already in use."
                    if "username" in error_msg: return None, "Username already in use."
                    if "student_id" in error_msg: return None, "Student ID already exists."
                return None, f"Database error: {error_msg}"
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error in create_student: {error_msg}")
            return None, error_msg

    @staticmethod
    def update_user(user_id, update_data):
        """Update user information"""
        try:
            # Convert user_id to ObjectId if it's a string
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            elif hasattr(user_id, 'id'):
                # If it's a user object, get the ID
                user_id = ObjectId(user_id.id)
                
            # Remove None values and empty strings from update_data
            clean_update_data = {}
            for key, value in update_data.items():
                if value is not None and value != '':
                    clean_update_data[key] = value
            
            if not clean_update_data:
                print("No valid data to update")
                return False
            
            # Add updated timestamp
            clean_update_data['updated_at'] = datetime.utcnow()
            
            # Update the user
            result = mongo.db.users.update_one(
                {'_id': user_id},
                {'$set': clean_update_data}
            )
            
            success = result.modified_count > 0
            print(f"User update result: {success}, modified_count: {result.modified_count}")
            return success
            
        except Exception as e:
            print(f"Error updating user: {str(e)}")
            return False

    # Exam Operations
    @staticmethod
    def create_exam(title, description, duration, teacher_id, start_time, end_time, total_marks, exam_type='scheduled', exam_format='objective', num_questions_to_display=None, class_id=None):
        try:
            exam_dict = {
                'title': title,
                'description': description,
                'duration': duration,
                'teacher_id': ObjectId(teacher_id),
                'start_time': start_time,
                'end_time': end_time,
                'created_at': datetime.utcnow(),
                'is_active': True,
                'questions': [],
                'exam_type': exam_type,
                'exam_format': exam_format,
                'total_marks': total_marks,
                'num_questions_to_display': num_questions_to_display,
                'class_id': ObjectId(class_id) if class_id else None
            }
            result = mongo.db.exams.insert_one(exam_dict)
            if result.inserted_id:
                exam_dict['_id'] = result.inserted_id
                return Exam.from_db(exam_dict)
            return None
        except Exception as e:
            print(f"Error creating exam: {str(e)}")
            return None

    @staticmethod
    def get_exam_by_id(exam_id):
        """Get exam by ID with proper question handling"""
        try:
            print(f"\n=== Getting exam with ID: {exam_id} ===")
            
            # Convert string ID to ObjectId if needed
            if isinstance(exam_id, str):
                exam_id = ObjectId(exam_id)
            
            # Get exam document
            exam_data = mongo.db.exams.find_one({'_id': exam_id})
            
            if not exam_data:
                print("No exam found with the given ID")
                return None
            
            print(f"Found exam: {exam_data.get('title')}")
            
            # Convert to Exam object using from_db method, but handle questions separately
            exam_data_copy = exam_data.copy()
            
            # Store questions separately and clear from exam_data temporarily
            questions_data = exam_data_copy.pop('questions', [])
            
            # Create exam object without questions first
            exam = Exam.from_db(exam_data_copy)
            if not exam:
                print("Failed to create exam object")
                return None
            
            # Now handle questions separately with better error handling
            processed_questions = []
            if questions_data:
                print(f"Processing {len(questions_data)} questions")
                for i, question in enumerate(questions_data):
                    try:
                        # Ensure question has an ID
                        if '_id' not in question and 'id' not in question:
                            question['_id'] = ObjectId()
                        
                        # Validate required fields
                        if not question.get('question_text'):
                            print(f"Skipping question {i}: Missing question text")
                            continue
                        
                        # Ensure marks is an integer
                        question['marks'] = int(question.get('marks', 1))
                        
                        # Set default question type if missing
                        if 'question_type' not in question:
                            question['question_type'] = 'multiple_choice'
                        
                        # For multiple choice questions, ensure options exist
                        if question['question_type'] == 'multiple_choice':
                            if 'options' not in question or not question['options']:
                                question['options'] = ['Option 1', 'Option 2', 'Option 3', 'Option 4']
                            if 'correct_answer' not in question:
                                question['correct_answer'] = 0
                        
                        processed_questions.append(question)
                        print(f"Successfully processed question {i+1}")
                    
                    except Exception as qe:
                        print(f"Error processing question {i}: {str(qe)}")
                        continue
                
            # Set the processed questions
            exam.questions = processed_questions
            print(f"Exam loaded with {len(processed_questions)} questions")
            
            # Ensure num_questions_to_display is properly set
            if not hasattr(exam, 'num_questions_to_display') or exam.num_questions_to_display is None:
                exam.num_questions_to_display = len(processed_questions)
            elif exam.num_questions_to_display > len(processed_questions):
                exam.num_questions_to_display = len(processed_questions)
            
            print(f"Will display {exam.num_questions_to_display} questions")
            return exam
            
        except Exception as e:
            print(f"Error getting exam: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def get_teacher_exams(teacher_id, limit=None, sort_by='created_at', sort_order=-1):
        """Get all exams for a teacher, optionally limited and sorted"""
        try:
            # Convert teacher_id to ObjectId
            teacher_id = ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id
            
            # Build query
            query = {'teacher_id': teacher_id}
            
            # Get exams from MongoDB
            cursor = mongo.db.exams.find(query).sort(sort_by, sort_order)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to Exam objects
            exams = []
            for exam_data in cursor:
                try:
                    exam = Exam.from_db(exam_data)
                    if exam:
                        exams.append(exam)
                except Exception as e:
                    print(f"Error converting exam data: {str(e)}")
                    print(f"Exam data: {exam_data}")
                    continue
            
            return exams
            
        except Exception as e:
            print(f"Error in get_teacher_exams: {str(e)}")
            return []

    @staticmethod
    def update_exam(exam_id, update_data):
        try:
            print(f"\n=== Updating exam {exam_id} ===")
            
            # Convert exam_id to ObjectId
            exam_id = ObjectId(exam_id) if isinstance(exam_id, str) else exam_id
            print(f"Converted exam_id to ObjectId: {exam_id}")
            
            # Convert class_id to ObjectId if provided
            if 'class_id' in update_data:
                update_data['class_id'] = ObjectId(update_data['class_id']) if update_data['class_id'] else None
                print(f"Class ID in update data: {update_data['class_id']}")
            
            # Ensure dates are datetime objects
            for field in ['start_time', 'end_time']:
                if field in update_data and update_data[field] and not isinstance(update_data[field], datetime):
                    try:
                        update_data[field] = datetime.fromisoformat(str(update_data[field]))
                    except ValueError:
                        print(f"Error converting {field} to datetime")
                        return False
            
            # Remove None values from update_data
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            print(f"Final update data to be applied: {update_data}")
            
            # Update the exam
            result = mongo.db.exams.update_one(
                {'_id': exam_id},
                {'$set': update_data}
            )
            
            success = result.acknowledged
            print(f"Update result - acknowledged: {success}, modified_count: {result.modified_count}")
            
            if success:
                print("Exam updated successfully")
                # Verify the update
                updated_exam = mongo.db.exams.find_one({'_id': exam_id})
                if updated_exam:
                    print("Update verified in database")
                    return True
            
            print("Failed to update exam or verify update")
            return False
            
        except Exception as e:
            print(f"Error updating exam: {str(e)}")
            print(f"Stack trace: {e.__traceback__}")
            return False

    @staticmethod
    def delete_exam(exam_id):
        """Delete an exam and all its associated data"""
        try:
            # Convert exam_id to ObjectId if it's a string
            exam_id = ObjectId(exam_id) if isinstance(exam_id, str) else exam_id
            
            # Delete all submissions for this exam
            mongo.db.submissions.delete_many({'exam_id': exam_id})
            
            # Delete the exam itself
            result = mongo.db.exams.delete_one({'_id': exam_id})
            
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting exam: {str(e)}")
            return False

    @staticmethod
    def add_question(exam_id, question_data):
        """Add a question to an exam."""
        try:
            print(f"\n=== Adding question to exam {exam_id} ===")
            print(f"Question data: {question_data}")
            
            # Generate new ObjectId for the question
            question_id = ObjectId()
            
            # Ensure all required fields are present
            formatted_question = {
                '_id': question_id,
                'question_text': question_data.get('question_text', ''),
                'question_type': question_data.get('question_type', 'multiple_choice'),
                'marks': int(question_data.get('marks', 1))
            }
            
            # Add type-specific fields
            if question_data.get('question_type') == 'multiple_choice':
                formatted_question.update({
                    'options': question_data.get('options', []),
                    'correct_answer': question_data.get('correct_answer')
                })
            else:  # subjective
                formatted_question['model_answer'] = question_data.get('model_answer', '')
            
            print(f"Formatted question: {formatted_question}")
            
            # Update exam with new question
            result = mongo.db.exams.update_one(
                {'_id': ObjectId(exam_id)},
                {'$push': {'questions': formatted_question}}
            )
            
            success = result.modified_count > 0
            if success:
                print(f"Successfully added question to exam")
                # Verify the question was added
                exam = mongo.db.exams.find_one({'_id': ObjectId(exam_id)})
                if exam and 'questions' in exam:
                    print(f"Exam now has {len(exam['questions'])} questions")
            else:
                    print("Warning: Could not verify question was added")
            return success
                
        except Exception as e:
            print(f"Error adding question: {str(e)}")
            print(f"Stack trace: {e.__traceback__}")
            return False

    @staticmethod
    def update_question(exam_id, question_id, update_data):
        """Update a question in an exam."""
        try:
            # Convert IDs to ObjectId
            exam_id = ObjectId(exam_id)
            question_id = ObjectId(question_id)
            
            # Update the specific question in the array
            result = mongo.db.exams.update_one(
                {
                    '_id': exam_id,
                    'questions._id': question_id
                },
                {
                    '$set': {
                        'questions.$': {
                            '_id': question_id,
                            **update_data
                        }
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating question: {str(e)}")
            return False

    @staticmethod
    def delete_question(exam_id, question_id):
        """Delete a question from an exam."""
        try:
            result = mongo.db.exams.update_one(
                {'_id': ObjectId(exam_id)},
                {'$pull': {'questions': {'_id': ObjectId(question_id)}}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error deleting question: {str(e)}")
            return False

    # Submission Operations
    @staticmethod
    def save_submission(submission):
        """Save or update a submission in the database with proper answer handling"""
        try:
            print("\n=== SAVE SUBMISSION DEBUG START ===")
            
            # 1. Convert submission to dict
            try:
                print("1. Converting submission to dict")
                submission_dict = submission.to_dict()
                print(f"   Submission dict created with {len(submission_dict.get('answers', {}))} answers")
            except Exception as e:
                print(f"Error converting submission to dict: {str(e)}")
                return False

            # 2. Ensure proper ObjectId conversion
            try:
                print("2. Converting IDs to ObjectId")
                submission_dict['exam_id'] = ObjectId(submission_dict['exam_id']) if submission_dict.get('exam_id') else None
                submission_dict['student_id'] = ObjectId(submission_dict['student_id']) if submission_dict.get('student_id') else None
                print(f"   Converted exam_id: {submission_dict['exam_id']}")
                print(f"   Converted student_id: {submission_dict['student_id']}")
            except Exception as e:
                print(f"Error converting IDs to ObjectId: {str(e)}")
                return False

            # 3. Validate required fields
            if not submission_dict.get('exam_id') or not submission_dict.get('student_id'):
                print("Error: Missing required exam_id or student_id")
                return False

            # 4. Process answers to ensure proper format
            try:
                print("3. Processing answers for database storage")
                answers = submission_dict.get('answers', {})
                if answers:
                    # Ensure all answer data is properly structured
                    processed_answers = {}
                    for question_id, answer_data in answers.items():
                        if isinstance(answer_data, dict):
                            # Ensure all required fields are present
                            processed_answer = {
                                'student_answer': answer_data.get('student_answer', ''),
                                'timestamp': answer_data.get('timestamp', datetime.utcnow().isoformat()),
                                'question_type': answer_data.get('question_type', 'unknown'),
                                'marks': float(answer_data.get('marks', 1)),
                                'awarded_marks': float(answer_data.get('awarded_marks', 0)),
                                'is_correct': bool(answer_data.get('is_correct', False))
                            }
                            # Add optional fields if present
                            if 'correct_answer' in answer_data:
                                processed_answer['correct_answer'] = answer_data['correct_answer']
                            if 'question_text' in answer_data:
                                processed_answer['question_text'] = answer_data['question_text']
                            if 'options' in answer_data:
                                processed_answer['options'] = answer_data['options']
                            if 'graded_by' in answer_data:
                                processed_answer['graded_by'] = answer_data['graded_by']
                            if 'graded_at' in answer_data:
                                processed_answer['graded_at'] = answer_data['graded_at']
                            processed_answers[str(question_id)] = processed_answer
                        else:
                            # Handle legacy format
                            processed_answers[str(question_id)] = {
                                'student_answer': str(answer_data),
                                'timestamp': datetime.utcnow().isoformat(),
                                'question_type': 'unknown',
                                'marks': 1.0,
                                'awarded_marks': 0.0,
                                'is_correct': False
                            }
                    
                    submission_dict['answers'] = processed_answers
                    print(f"   Processed {len(processed_answers)} answers")
                else:
                    submission_dict['answers'] = {}
            except Exception as e:
                print(f"Error processing answers: {str(e)}")
                return False

            # 5. Create update query
            try:
                print("4. Preparing database update")
                query = {
                    'student_id': submission_dict['student_id'],
                    'exam_id': submission_dict['exam_id']
                }
                print(f"   Query: {query}")
            except Exception as e:
                print(f"Error preparing update query: {str(e)}")
                return False

            # 6. Perform the update
            try:
                print("5. Executing database update")
                result = mongo.db.submissions.update_one(
                    query,
                    {'$set': submission_dict},
                    upsert=True
                )
                print(f"   Update result - Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
                
                # If this was a new document, add the _id to the submission
                if result.upserted_id and not submission.id:
                    submission.id = str(result.upserted_id)
                    print(f"   Set new submission ID: {submission.id}")
            except Exception as e:
                print(f"Error executing database update: {str(e)}")
                print(f"Error details: {traceback.format_exc()}")
                return False

            print("=== SAVE SUBMISSION DEBUG END ===")
            return True
            
        except Exception as e:
            print(f"=== UNEXPECTED ERROR IN SAVE SUBMISSION ===")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return False

    @staticmethod
    def create_submission(submission_data):
        """Create a new submission with proper answer processing and grading"""
        try:
            print("\n=== CREATE SUBMISSION DEBUG START ===")
            
            # Handle both Submission objects and dictionaries
            if hasattr(submission_data, 'to_dict'):
                submission_dict = submission_data.to_dict()
                print(f"Converting Submission object to dict")
            else:
                submission_dict = submission_data.copy() if isinstance(submission_data, dict) else submission_data
                print(f"Using provided dict")
            
            print(f"Creating submission with {len(submission_dict.get('answers', {}))} answers")
            
            # Validate required fields
            if not submission_dict.get('exam_id') or not submission_dict.get('student_id'):
                print("ERROR: Missing required exam_id or student_id")
                return None
            
            # Convert IDs to ObjectId if they're strings
            if isinstance(submission_dict['exam_id'], str):
                submission_dict['exam_id'] = ObjectId(submission_dict['exam_id'])
            if isinstance(submission_dict['student_id'], str):
                submission_dict['student_id'] = ObjectId(submission_dict['student_id'])
            
            print(f"Converted IDs - Exam: {submission_dict['exam_id']}, Student: {submission_dict['student_id']}")
            
            # Ensure timestamps are datetime objects
            now = datetime.utcnow()
            
            if isinstance(submission_dict.get('submitted_at'), str):
                try:
                    submission_dict['submitted_at'] = datetime.fromisoformat(submission_dict['submitted_at'])
                except:
                    submission_dict['submitted_at'] = now
            elif not submission_dict.get('submitted_at'):
                submission_dict['submitted_at'] = now
                
            if isinstance(submission_dict.get('started_at'), str):
                try:
                    submission_dict['started_at'] = datetime.fromisoformat(submission_dict['started_at'])
                except:
                    submission_dict['started_at'] = now
            elif not submission_dict.get('started_at'):
                submission_dict['started_at'] = now
                
            if isinstance(submission_dict.get('graded_at'), str):
                try:
                    submission_dict['graded_at'] = datetime.fromisoformat(submission_dict['graded_at'])
                except:
                    submission_dict['graded_at'] = None
            
            # Set default values for missing fields
            submission_dict.setdefault('is_graded', False)
            submission_dict.setdefault('auto_submitted', False)
            submission_dict.setdefault('auto_graded', False)
            submission_dict.setdefault('detailed_results', [])
            submission_dict.setdefault('violation_details', [])
            submission_dict.setdefault('warning_count_at_submission', 0)
            submission_dict.setdefault('score', None)
            submission_dict.setdefault('max_score', None)
            submission_dict.setdefault('percentage', None)
            submission_dict.setdefault('pass_fail', None)
            # Process questions to ensure proper format
            questions = submission_dict.get('questions', [])
            if questions:
                processed_questions = []
                for question in questions:
                    if isinstance(question, dict):
                        # Keep question data as-is for database storage
                        processed_questions.append(question)
                    else:
                        # Convert Question object to dict if needed
                        if hasattr(question, 'to_dict'):
                            processed_questions.append(question.to_dict())
                        else:
                            processed_questions.append(question)
                submission_dict['questions'] = processed_questions
                print(f" Processed {len(processed_questions)} questions for submission")
            else:
                submission_dict['questions'] = []
                print(" No questions provided for submission")
            
            # Process answers to ensure proper format
            answers = submission_dict.get('answers', {})
            print(f"Raw answers received: {answers}")
            print(f"Raw answers type: {type(answers)}")
            print(f"Raw answers length: {len(answers) if answers else 0}")
            
            if answers:
                processed_answers = {}
                for question_id, answer_data in answers.items():
                    print(f"Processing question {question_id}: {answer_data} (type: {type(answer_data)})")
                    
                    if isinstance(answer_data, dict):
                        # Ensure all required fields are present with proper types
                        processed_answer = {
                            'student_answer': str(answer_data.get('student_answer', '')),
                            'timestamp': answer_data.get('timestamp', now.isoformat()),
                            'question_type': answer_data.get('question_type', 'unknown'),
                            'marks': float(answer_data.get('marks', 1)),
                            'awarded_marks': float(answer_data.get('awarded_marks', 0)),
                            'is_correct': bool(answer_data.get('is_correct', False))
                        }
                        
                        # Add optional fields if present
                        optional_fields = ['correct_answer', 'question_text', 'options', 'graded_by', 'graded_at']
                        for field in optional_fields:
                            if field in answer_data:
                                processed_answer[field] = answer_data[field]
                                
                        processed_answers[str(question_id)] = processed_answer
                        print(f" Processed complex answer for {question_id}")
                    else:
                        # Handle simple format (just the answer string)
                        processed_answers[str(question_id)] = {
                            'student_answer': str(answer_data) if answer_data is not None else '',
                            'timestamp': now.isoformat(),
                            'question_type': 'unknown',
                            'marks': 1.0,
                            'awarded_marks': 0.0,
                            'is_correct': False
                        }
                        print(f" Processed simple answer for {question_id}: '{answer_data}'")
                
                submission_dict['answers'] = processed_answers
                print(f" Final processed answers: {len(processed_answers)} items")
                print(f" Sample processed answer: {list(processed_answers.values())[0] if processed_answers else 'None'}")
            else:
                print(" No answers to process")
                submission_dict['answers'] = {}
            
            # Insert into database
            print("Inserting submission into database...")
            result = mongo.db.submissions.insert_one(submission_dict)
            
            if result.inserted_id:
                submission_dict['_id'] = result.inserted_id
                created_submission = Submission.from_db(submission_dict)
                print(f" Successfully created submission with ID: {created_submission.id}")
                print(f"   Score: {created_submission.score}/{created_submission.max_score}")
                print(f"   Graded: {created_submission.is_graded}")
                print(f"   Auto-submitted: {created_submission.auto_submitted}")
                return created_submission
            else:
                print(" Failed to insert submission - no ID returned")
                return None
                
        except Exception as e:
            print(f" Error creating submission: {str(e)}")
            import traceback
            print(f"Error traceback: {traceback.format_exc()}")
            return None
        finally:
            print("=== CREATE SUBMISSION DEBUG END ===")

    @staticmethod
    def auto_grade_submission(submission, exam_questions):
        """Auto-grade a submission for objective questions"""
        try:
            print(f"\n=== AUTO GRADING SUBMISSION {submission.id} ===")
            
            if not exam_questions:
                print("No exam questions provided for grading")
                return False
            
            # Create a mapping of question IDs to question data
            questions_map = {}
            for question in exam_questions:
                question_id = str(question.get('id') or question.get('_id', ''))
                if question_id:
                    questions_map[question_id] = question
            
            total_marks = 0
            earned_marks = 0
            graded_answers = {}
            
            # Grade each answer
            for question_id, answer_data in submission.answers.items():
                if question_id not in questions_map:
                    print(f"Question {question_id} not found in exam questions")
                    continue
                
                question = questions_map[question_id]
                question_marks = float(question.get('marks', 1))
                question_type = question.get('question_type', question.get('type', 'unknown'))
                total_marks += question_marks
                
                student_answer = answer_data.get('student_answer', '')
                awarded_marks = 0
                is_correct = False
                
                if question_type == 'multiple_choice':
                    # Auto-grade objective questions
                    correct_answer = question.get('correct_answer', '')
                    is_correct = str(student_answer).strip() == str(correct_answer).strip()
                    awarded_marks = question_marks if is_correct else 0
                    print(f"Question {question_id}: Student='{student_answer}', Correct='{correct_answer}', Correct={is_correct}")
                else:
                    # Subjective questions need manual grading
                    awarded_marks = answer_data.get('awarded_marks', 0)
                    is_correct = awarded_marks > 0
                
                earned_marks += awarded_marks
                
                # Update answer data
                graded_answer = answer_data.copy()
                graded_answer.update({
                    'marks': question_marks,
                    'awarded_marks': awarded_marks,
                    'is_correct': is_correct,
                    'question_type': question_type,
                    'correct_answer': question.get('correct_answer', ''),
                    'question_text': question.get('question_text', '')
                })
                graded_answers[question_id] = graded_answer
            
            # Update submission with grading results
            submission.answers = graded_answers
            submission.score = earned_marks
            submission.max_score = total_marks
            submission.percentage = round((earned_marks / total_marks * 100), 2) if total_marks > 0 else 0
            submission.is_graded = True
            submission.graded_at = datetime.utcnow()
            submission.auto_graded = True
            submission.graded_by = "Auto-grading system"
            submission.pass_fail = "PASS" if submission.percentage >= 50 else "FAIL"
            
            print(f"Grading completed: {earned_marks}/{total_marks} ({submission.percentage}%) - {submission.pass_fail}")
            
            # Save the graded submission
            return MongoManager.save_submission(submission)
            
        except Exception as e:
            print(f"Error auto-grading submission: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return False

    @staticmethod
    def get_student_submissions(student_id, sort_by_date=True):
        """Get all submissions for a student, optionally sorted by date"""
        query = {'student_id': ObjectId(student_id)}
        cursor = mongo.db.submissions.find(query)
        if sort_by_date:
            cursor = cursor.sort('submitted_at', -1)
        return [Submission.from_db(sub) for sub in cursor]

    @staticmethod
    def get_graded_submissions(student_id):
        """Get all graded submissions for a student"""
        query = {
            'student_id': ObjectId(student_id),
            'is_graded': True
        }
        cursor = mongo.db.submissions.find(query).sort('submitted_at', -1)
        return [Submission.from_db(sub) for sub in cursor]

    @staticmethod
    def get_student_submission_for_exam(student_id, exam_id):
        """Get a student's submission for a specific exam"""
        try:
            print(f"\n=== Getting student submission - Student: {student_id}, Exam: {exam_id} ===")
            
            # Convert IDs to ObjectId
            student_oid = ObjectId(student_id) if isinstance(student_id, str) else student_id
            exam_oid = ObjectId(exam_id) if isinstance(exam_id, str) else exam_id
            
            print(f"Looking up submission with student_id={student_oid}, exam_id={exam_oid}")
            
            # Find submission in database
            submission_data = mongo.db.submissions.find_one({
                'student_id': student_oid,
                'exam_id': exam_oid
            })
            
            if submission_data:
                print(f"Found submission: ID={submission_data['_id']}")
                
                # Get exam questions if not in submission
                if 'questions' not in submission_data or not submission_data['questions']:
                    print("Retrieving exam questions for submission...")
                    exam_data = mongo.db.exams.find_one({'_id': exam_oid})
                    if exam_data and 'questions' in exam_data:
                        submission_data['questions'] = exam_data['questions']
                        print(f"Added {len(exam_data['questions'])} questions to submission")
                
                # Convert to Submission object
                submission = Submission.from_db(submission_data)
                print(f"Converted to Submission object: ID={submission.id}, Answers count={len(submission.answers)}")
                return submission
            else:
                print("No submission found")
                return None
                
        except Exception as e:
            print(f"Error getting student submission: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    @staticmethod
    def get_exam_submissions(exam_id):
        submissions_data = mongo.db.submissions.find({'exam_id': ObjectId(exam_id)})
        return [Submission.from_db(sub_data) for sub_data in submissions_data]

    @staticmethod
    def count_exam_submissions(exam_id):
        """Count total submissions for an exam"""
        try:
            return mongo.db.submissions.count_documents({'exam_id': ObjectId(exam_id)})
        except Exception as e:
            print(f"Error counting submissions: {str(e)}")
            return 0

    @staticmethod
    def count_completed_submissions(exam_id):
        """Count completed submissions for an exam"""
        try:
            return mongo.db.submissions.count_documents({
                'exam_id': ObjectId(exam_id),
                'is_submitted': True
            })
        except Exception as e:
            print(f"Error counting completed submissions: {str(e)}")
            return 0

    @staticmethod
    def get_exam_average_score(exam_id):
        """Calculate average score for an exam"""
        try:
            pipeline = [
                {'$match': {
                    'exam_id': ObjectId(exam_id),
                    'is_submitted': True,
                    'score': {'$exists': True}
                }},
                {'$group': {
                    '_id': None,
                    'average_score': {'$avg': '$score'}
                }}
            ]
            result = list(mongo.db.submissions.aggregate(pipeline))
            return result[0]['average_score'] if result else 0
        except Exception as e:
            print(f"Error calculating average score: {str(e)}")
            return 0

    @staticmethod
    def get_recent_submissions_for_exam(exam_id, limit=10):
        """Get recent submissions for an exam"""
        try:
            submissions = list(mongo.db.submissions.find(
                {'exam_id': ObjectId(exam_id)},
                sort=[('submitted_at', -1)]
            ).limit(limit))
            
            # Add student names to submissions
            for submission in submissions:
                student = mongo.db.users.find_one({'_id': submission['student_id']})
                if student:
                    submission['student_name'] = student.get('username', 'Unknown')
            
            return submissions
        except Exception as e:
            print(f"Error getting recent submissions: {str(e)}")
            return []

    @staticmethod
    def update_submission_score(submission_id, score):
        """Update submission score and mark as graded"""
        try:
            # Convert submission_id to ObjectId if it's a string
            if isinstance(submission_id, str):
                submission_id = ObjectId(submission_id)
            
            # Validate score
            if not isinstance(score, (int, float)):
                print(f"Invalid score type: {type(score)}")
                return False
            
            if score < 0 or score > 100:
                print(f"Score out of range: {score}")
                return False
            
            # Get the current submission
            submission = mongo.db.submissions.find_one({'_id': submission_id})
            if not submission:
                print("Submission not found")
                return False
            
            # Update submission with score and grading information
            update_data = {
                'score': float(score),
                'is_graded': True,
                'graded_at': datetime.utcnow()
            }
            
            # If graded_by is not already set, mark as manually graded
            if not submission.get('graded_by'):
                update_data['graded_by'] = 'Manual grading'
            
            result = mongo.db.submissions.update_one(
                {'_id': submission_id},
                {'$set': update_data}
            )
            
            success = result.modified_count > 0
            print(f"Submission score update result: {success}, score: {score}")
            return success
            
        except Exception as e:
            print(f"Error updating submission score: {str(e)}")
            return False

    @staticmethod
    def create_class(class_obj):
        """Create a new class"""
        class_dict = class_obj.to_dict()
        result = mongo.db.classes.insert_one(class_dict)
        class_dict['_id'] = result.inserted_id
        return Class(class_dict)

    @staticmethod
    def get_class_by_id(class_id):
        """Get class by ID"""
        try:
            class_data = mongo.db.classes.find_one({'_id': ObjectId(class_id)})
            return Class(class_data) if class_data else None
        except Exception as e:
            print(f"Error getting class: {str(e)}")
            return None

    @staticmethod
    def delete_class(class_id):
        """Delete a class"""
        result = mongo.db.classes.delete_one({'_id': ObjectId(class_id)})
        return result.deleted_count > 0

    @staticmethod
    def add_student_to_class(class_id, student_id):
        """Add a student to a class"""
        try:
            if not class_id or not student_id:
                print("Error: class_id or student_id is missing")
                return False

            # Convert IDs to ObjectId
            try:
                class_id = ObjectId(class_id) if isinstance(class_id, str) else class_id
                student_id = ObjectId(student_id) if isinstance(student_id, str) else student_id
            except Exception as oid_err:
                print(f"Invalid ID format: {str(oid_err)}")
                return False
            
            print(f"Adding student {student_id} to class {class_id} in {mongo.db.name}")
            
            # Verify class exists
            class_data = mongo.db.classes.find_one({'_id': class_id})
            if not class_data:
                print("Class not found")
                return False
            
            # Verify student exists
            student = mongo.db.users.find_one({'_id': student_id, 'role': 'student'})
            if not student:
                print("Student not found or not a student")
                return False
            
            # Check if student is already in class
            if student_id in class_data.get('students', []):
                print("Student already in class")
                return False
            
            # Add student to class
            result = mongo.db.classes.update_one(
                {'_id': class_id},
                {'$addToSet': {'students': student_id}}  # Use addToSet to prevent duplicates
            )
            
            success = result.modified_count > 0
            print(f"Add student to class result: {success}")
            return success
            
        except Exception as e:
            print(f"Error adding student to class: {str(e)}")
            return False

    @staticmethod
    def remove_student_from_class(class_id, student_id):
        """Remove a student from a class"""
        result = mongo.db.classes.update_one(
            {'_id': ObjectId(class_id)},
            {'$pull': {'students': ObjectId(student_id)}}
        )
        return result.modified_count > 0

    @staticmethod
    def get_exam_analytics(teacher_id):
        """Get analytics for a teacher's exams"""
        # Get all exams for the teacher
        exams = list(mongo.db.exams.find({'teacher_id': ObjectId(teacher_id)}))
        
        # Get all submissions for these exams
        exam_ids = [exam['_id'] for exam in exams]
        submissions = list(mongo.db.submissions.find({'exam_id': {'$in': exam_ids}}))
        
        # Calculate statistics
        total_exams = len(exams)
        total_submissions = len(submissions)
        
        # Calculate average scores for graded submissions
        graded_submissions = [sub for sub in submissions if sub.get('is_graded', False)]
        if graded_submissions:
            avg_score = sum(sub.get('score', 0) for sub in graded_submissions) / len(graded_submissions)
        else:
            avg_score = 0
            
        return {
            'total_exams': total_exams,
            'total_submissions': total_submissions,
            'avg_score': round(avg_score, 2)
        }

    @staticmethod
    def delete_student(student_id):
        """Delete a student and their related data"""
        # Delete the student
        mongo.db.users.delete_one({'_id': ObjectId(student_id)})
        
        # Delete their submissions
        mongo.db.submissions.delete_many({'student_id': ObjectId(student_id)})
        
        # Remove them from all classes
        mongo.db.classes.update_many(
            {'students': ObjectId(student_id)},
            {'$pull': {'students': ObjectId(student_id)}}
        )

    @staticmethod
    def find_exams(teacher_id=None):
        """Find exams with optional teacher filter"""
        query = {}
        if teacher_id:
            query['teacher_id'] = ObjectId(teacher_id)
        exams_data = mongo.db.exams.find(query)
        return [Exam.from_db(exam) for exam in exams_data]

    @staticmethod
    def find_submissions(student_id=None, exam_id=None):
        """Find submissions with optional filters"""
        query = {}
        if student_id:
            query['student_id'] = ObjectId(student_id)
        if exam_id:
            if isinstance(exam_id, list):
                query['exam_id'] = {'$in': [ObjectId(eid) for eid in exam_id]}
            else:
                query['exam_id'] = ObjectId(exam_id)
        submissions_data = mongo.db.submissions.find(query)
        return [Submission.from_db(sub) for sub in submissions_data]

    @staticmethod
    def get_class_students(class_id):
        """Get all students in a class"""
        try:
            class_data = mongo.db.classes.find_one({'_id': ObjectId(class_id)})
            if not class_data or 'students' not in class_data:
                return []
            
            student_ids = class_data['students']
            students_data = mongo.db.users.find({
                '_id': {'$in': student_ids},
                'role': 'student'
            })
            
            students = []
            for student_data in students_data:
                student = User.from_db(student_data)
                student.class_info = {
                    'id': str(class_data['_id']),
                    'name': class_data['name']
                }
                students.append(student)
            
            return students
        except Exception as e:
            print(f"Error getting class students: {str(e)}")
            return []

    @staticmethod
    def get_student_class(student_id):
        """Get the class a student belongs to"""
        try:
            # Convert student_id to ObjectId
            student_id = ObjectId(student_id)
            
            # Find the class that contains this student
            class_data = mongo.db.classes.find_one({'students': student_id})
            return Class(class_data) if class_data else None
        except Exception as e:
            print(f"Error getting student class: {str(e)}")
            return None

    @staticmethod
    def migrate_user_password(user_id, new_password):
        """Migrate a user's password hash to pbkdf2:sha256"""
        try:
            user = MongoManager.get_user_by_id(user_id)
            if not user:
                return False
                
            # Generate new password hash using pbkdf2
            new_hash = User.set_password(new_password)
            
            # Update the user's password hash
            result = mongo.db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'password_hash': new_hash}}
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"Error migrating password: {str(e)}")
            return False

    @staticmethod
    def migrate_all_passwords():
        """Find all users with scrypt hashes and mark them for password reset"""
        try:
            # Find all users with scrypt password hashes
            users = mongo.db.users.find({
                'password_hash': {'$regex': '^scrypt:'}
            })
            
            # Mark these users for password reset
            result = mongo.db.users.update_many(
                {'password_hash': {'$regex': '^scrypt:'}},
                {'$set': {'needs_password_reset': True}}
            )
            
            return result.modified_count
        except Exception as e:
            print(f"Error marking users for password migration: {str(e)}")
            return 0

    @staticmethod
    def check_password_migration_needed(user_id):
        """Check if a user needs password migration"""
        try:
            user = MongoManager.get_user_by_id(user_id)
            return user and user.needs_password_migration()
        except Exception as e:
            print(f"Error checking password migration status: {str(e)}")
            return False

    @staticmethod
    def remove_student_from_all_classes(student_id):
        """Remove a student from all classes"""
        try:
            student_id = ObjectId(student_id)
            result = mongo.db.classes.update_many(
                {'students': student_id},
                {'$pull': {'students': student_id}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error removing student from classes: {str(e)}")
            return False

    @staticmethod
    def get_submission_by_id(submission_id):
        """Get a submission by its ID with proper format handling"""
        try:
            print(f"\n=== Getting submission by ID: {submission_id} ===")
            
            submission_data = mongo.db.submissions.find_one({'_id': ObjectId(submission_id)})
            if submission_data:
                print(f"Found submission: {submission_data['_id']}")
                
                # Ensure questions are present
                if 'questions' not in submission_data or not submission_data['questions']:
                    print("Retrieving exam questions for submission...")
                    exam_data = mongo.db.exams.find_one({'_id': submission_data['exam_id']})
                    if exam_data and 'questions' in exam_data:
                        submission_data['questions'] = exam_data['questions']
                        print(f"Added {len(exam_data['questions'])} questions to submission")
                
                submission = Submission.from_db(submission_data)
                print(f"Converted to Submission object: Score={submission.score}/{submission.max_score}")
                return submission
            else:
                print("Submission not found")
                return None
                
        except Exception as e:
            print(f"Error getting submission by ID: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    @staticmethod
    def get_random_exam_questions(exam_id, num_questions):
        """Get random questions for an exam"""
        try:
            exam = mongo.db.exams.find_one({'_id': ObjectId(exam_id)})
            if not exam or 'questions' not in exam:
                return []
            
            # Get all questions from the exam
            all_questions = exam.get('questions', [])
            
            # If requested number is greater than available questions, return all
            if num_questions >= len(all_questions):
                return [Question.from_db(q) for q in all_questions]
            
            # Randomly select questions
            from random import sample
            selected_questions = sample(all_questions, num_questions)
            
            # Convert to Question objects
            questions = []
            for q in selected_questions:
                try:
                    question = Question.from_db(q)
                    if question:
                        questions.append(question)
                except Exception as e:
                    print(f"Error converting question: {str(e)}")
                    continue
            
            return questions
            
        except Exception as e:
            print(f"Error getting random questions: {str(e)}")
            return []

    @staticmethod
    def update_exam_question_settings(exam_id, num_questions_to_display):
        """Update exam settings for number of questions to display"""
        try:
            result = mongo.db.exams.update_one(
                {'_id': ObjectId(exam_id)},
                {'$set': {'num_questions_to_display': num_questions_to_display}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating exam question settings: {str(e)}")
            return False

    @staticmethod
    def get_available_exams_for_student(student_id):
        """Get all available exams for a student — scoped to their class only"""
        try:
            now = datetime.utcnow()
            
            # Convert student_id to ObjectId
            student_id = ObjectId(student_id) if isinstance(student_id, str) else student_id
            
            # Get student's class
            class_data = mongo.db.classes.find_one({'students': student_id})
            
            # Base query
            query = {
                'is_active': True,
                'exam_type': 'self_paced',
                'end_time': {'$gte': now}
            }
            
            if class_data:
                # Student is in a class — show only exams assigned to that class
                query['class_id'] = class_data['_id']
            else:
                # Student has no class — show no exams (they need to be enrolled)
                return []
            
            # Get exams from MongoDB
            exams_data = list(mongo.db.exams.find(query))
            
            # Convert to Exam objects, skipping already submitted ones
            exams = []
            for exam_data in exams_data:
                try:
                    if MongoManager.has_student_submitted_exam(student_id, exam_data['_id']):
                        continue
                    exam = Exam.from_db(exam_data)
                    if exam:
                        exams.append(exam)
                except Exception as e:
                    print(f"Error converting exam data: {str(e)}")
                    continue
            
            return exams
            
        except Exception as e:
            print(f"Error getting available exams: {str(e)}")
            return []

    @staticmethod
    def get_upcoming_exams_for_student(student_id):
        """Get upcoming exams for a student"""
        try:
            now = datetime.utcnow()
            # Get student's class
            class_data = mongo.db.classes.find_one({'students': ObjectId(student_id)})
            
            # Base query for upcoming exams
            query = {
                'is_active': True,
                'exam_type': 'scheduled',  # Only scheduled exams can be upcoming
                'start_time': {'$gt': now}
            }
            
            # If student is in a class, include exams for that class
            if class_data:
                query['$or'] = [
                    {'class_id': class_data['_id']},
                    {'class_id': None}  # Include exams not assigned to any class
                ]
            else:
                query['class_id'] = None
            
            exams_data = mongo.db.exams.find(query).sort('start_time', 1)
            return [Exam.from_db(exam_data) for exam_data in exams_data]
            
        except Exception as e:
            print(f"Error getting upcoming exams: {str(e)}")
            return []

    @staticmethod
    def student_has_access_to_exam(student_id, exam_id):
        """Check if a student has access to an exam (must be in the assigned class)"""
        try:
            # Get the exam
            exam_data = mongo.db.exams.find_one({'_id': ObjectId(exam_id)})
            if not exam_data:
                return False
            
            # Exam must be assigned to a class — no class = no student access
            if not exam_data.get('class_id'):
                return False
            
            # Check if student is in the exam's class
            class_data = mongo.db.classes.find_one({
                '_id': exam_data['class_id'],
                'students': ObjectId(student_id)
            })
            return bool(class_data)
            
        except Exception as e:
            print(f"Error checking exam access: {str(e)}")
            return False

    @staticmethod
    def has_student_submitted_exam(student_id, exam_id):
        """Check if a student has already submitted an exam"""
        try:
            submission = mongo.db.submissions.find_one({
                'student_id': ObjectId(student_id),
                'exam_id': ObjectId(exam_id)
            })
            return bool(submission)
            
        except Exception as e:
            print(f"Error checking exam submission: {str(e)}")
            return False

    @staticmethod
    def get_active_teacher_exams(teacher_id, limit=10):
        """Get active exams for a teacher with detailed information"""
        now = datetime.utcnow()
        try:
            # Convert teacher_id to ObjectId
            teacher_id = ObjectId(teacher_id) if isinstance(teacher_id, str) else teacher_id
            
            # Find active exams that haven't expired
            cursor = mongo.db.exams.find({
                'teacher_id': teacher_id,
                'is_active': True,
                'end_time': {'$gte': now}
            }).sort('created_at', -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to Exam objects
            exams = []
            for exam_data in cursor:
                try:
                    exam = Exam.from_db(exam_data)
                    if exam:
                        exams.append(exam)
                except Exception as e:
                    print(f"Error converting exam data: {str(e)}")
                    continue
            
            return exams
            
        except Exception as e:
            print(f"Error in get_active_teacher_exams: {str(e)}")
            return []

    @staticmethod
    def get_warning_count(student_id, exam_id):
        """Get the total warning count for a student in an exam"""
        try:
            # Count warnings from the warnings collection
            warning_count = mongo.db.warnings.count_documents({
                'student_id': ObjectId(student_id),
                'exam_id': ObjectId(exam_id)
            })
            return warning_count
        except Exception as e:
            print(f"Error getting warning count: {str(e)}")
            return 0
