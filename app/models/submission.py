from datetime import datetime
from bson import ObjectId

class Submission:
    def __init__(self, id=None, exam_id=None, student_id=None, answers=None, questions=None, submitted_at=None, 
                 score=None, is_graded=False, graded_at=None, started_at=None,
                 warning_count_at_submission=0, auto_submitted=False, 
                 submission_reason=None, graded_by=None, max_score=None,
                 percentage=None, detailed_results=None, questions_graded=None,
                 auto_graded=False, auto_submit_reason=None, pass_fail=None,
                 violation_details=None, **kwargs):
        self.id = str(id) if id else None
        self.exam_id = str(exam_id) if exam_id else None
        self.student_id = str(student_id) if student_id else None
        self.answers = answers or {}
        self.questions = questions or []
        self.submitted_at = submitted_at or datetime.utcnow()
        self.score = score
        self.is_graded = is_graded
        self.graded_at = graded_at
        self.started_at = started_at
        self.warning_count_at_submission = warning_count_at_submission or 0
        self.auto_submitted = auto_submitted or False
        self.submission_reason = submission_reason
        self.graded_by = graded_by
        # Auto-grading specific attributes
        self.max_score = max_score
        self.percentage = percentage
        self.detailed_results = detailed_results or []
        self.questions_graded = questions_graded
        self.auto_graded = auto_graded or False
        self.auto_submit_reason = auto_submit_reason
        self.pass_fail = pass_fail
        # Advanced proctoring violation details
        self.violation_details = violation_details or []
        
        # Handle any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @staticmethod
    def from_db(submission_data):
        if not submission_data:
            return None
        return Submission(
            id=submission_data['_id'],
            exam_id=submission_data['exam_id'],
            student_id=submission_data['student_id'],
            answers=submission_data['answers'],
            questions=submission_data.get('questions', []),
            submitted_at=submission_data.get('submitted_at'),
            score=submission_data.get('score'),
            is_graded=submission_data.get('is_graded', False),
            graded_at=submission_data.get('graded_at'),
            started_at=submission_data.get('started_at'),
            warning_count_at_submission=submission_data.get('warning_count_at_submission', 0),
            auto_submitted=submission_data.get('auto_submitted', False),
            submission_reason=submission_data.get('submission_reason'),
            graded_by=submission_data.get('graded_by'),
            # Auto-grading specific attributes
            max_score=submission_data.get('max_score'),
            percentage=submission_data.get('percentage'),
            detailed_results=submission_data.get('detailed_results', []),
            questions_graded=submission_data.get('questions_graded'),
            auto_graded=submission_data.get('auto_graded', False),
            auto_submit_reason=submission_data.get('auto_submit_reason'),
            pass_fail=submission_data.get('pass_fail'),
            violation_details=submission_data.get('violation_details', [])
        )

    def to_dict(self):
        """Convert submission to dictionary for MongoDB storage"""
        result = {
            'exam_id': ObjectId(self.exam_id) if self.exam_id else None,
            'student_id': ObjectId(self.student_id) if self.student_id else None,
            'answers': self.answers,
            'questions': self.questions,
            'submitted_at': self.submitted_at,
            'score': self.score,
            'is_graded': self.is_graded,
            'graded_at': self.graded_at,
            'started_at': self.started_at,
            'warning_count_at_submission': self.warning_count_at_submission,
            'auto_submitted': self.auto_submitted,
            'submission_reason': self.submission_reason,
            'graded_by': ObjectId(self.graded_by) if self.graded_by else None,
            'max_score': self.max_score,
            'percentage': self.percentage,
            'detailed_results': self.detailed_results,
            'questions_graded': self.questions_graded,
            'auto_graded': self.auto_graded,
            'auto_submit_reason': self.auto_submit_reason,
            'pass_fail': self.pass_fail,
            'violation_details': self.violation_details
        }
        
        # Only include _id if it exists
        if self.id:
            result['_id'] = ObjectId(self.id) if isinstance(self.id, str) else self.id
            
        return result

    def grade(self, score):
        self.score = score
        self.is_graded = True
        self.graded_at = datetime.utcnow()

    def __repr__(self):
        return f'<Submission {self.id} - Exam {self.exam_id}>' 