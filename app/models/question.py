from bson import ObjectId
from datetime import datetime

class Question:
    def __init__(self, question_text, question_type, marks, options=None, 
                 correct_answer=None, model_answer=None, id=None):
        self._id = ObjectId(id) if isinstance(id, str) else id or ObjectId()
        self.id = str(self._id)
        self.question_text = question_text
        self.question_type = question_type  # 'multiple_choice' or 'subjective'
        self.marks = marks
        self.options = options if question_type == 'multiple_choice' else None
        self.correct_answer = correct_answer if question_type == 'multiple_choice' else None
        self.model_answer = model_answer if question_type == 'subjective' else None

    @staticmethod
    def from_db(question_data):
        """Create a Question instance from database data"""
        if not question_data:
            return None
            
        try:
            # Handle both _id and id fields
            question_id = question_data.get('_id') or question_data.get('id')
            
            # Get required fields with defaults
            question_text = question_data.get('question_text', '')
            question_type = question_data.get('question_type', 'multiple_choice')
            marks = int(question_data.get('marks', 1))
            
            # Get type-specific fields
            options = None
            correct_answer = None
            model_answer = None
            
            if question_type == 'multiple_choice':
                options = question_data.get('options', [])
                correct_answer = question_data.get('correct_answer')
            else:  # subjective
                model_answer = question_data.get('model_answer', '')
            
            return Question(
                id=question_id,
                question_text=question_text,
                question_type=question_type,
                marks=marks,
                options=options,
                correct_answer=correct_answer,
                model_answer=model_answer
            )
        except Exception as e:
            print(f"Error creating Question from data: {str(e)}")
            print(f"Question data: {question_data}")
            return None
    
    def to_dict(self):
        """Convert Question instance to dictionary for database storage"""
        question_dict = {
            '_id': self._id,
            'question_text': self.question_text,
            'question_type': self.question_type,
            'marks': self.marks
        }
        
        if self.question_type == 'multiple_choice':
            question_dict.update({
                'options': self.options,
                'correct_answer': self.correct_answer
            })
        else:  # subjective
            question_dict['model_answer'] = self.model_answer
            
        return question_dict

    def is_multiple_choice(self):
        return self.question_type == 'multiple_choice'

    def is_subjective(self):
        return self.question_type == 'subjective'

    def check_answer(self, student_answer):
        """Check if the answer is correct"""
        if self.is_multiple_choice():
            try:
                student_choice = int(student_answer)
                return student_choice == self.correct_answer
            except (ValueError, TypeError):
                return False
        return None  # Subjective questions need manual grading
    
    def get_marks(self):
        return self.marks
    
    def __repr__(self):
        return f'<Question {self.id}: {self.question_type}>' 