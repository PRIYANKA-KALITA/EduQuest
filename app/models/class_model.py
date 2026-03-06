from datetime import datetime
from bson import ObjectId

class Class:
    def __init__(self, data=None, **kwargs):
        if data is None:
            data = kwargs
        
        self.id = str(data.get('_id', ObjectId()))
        self.name = data.get('name')
        self.description = data.get('description')
        self.created_at = data.get('created_at', datetime.utcnow())
        self.teacher_id = str(data.get('teacher_id'))
        self.students = data.get('students', [])

    @staticmethod
    def from_mongo(data):
        return Class(data)

    def to_dict(self):
        return {
            '_id': ObjectId(self.id) if self.id else ObjectId(),
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'teacher_id': ObjectId(self.teacher_id),
            'students': self.students
        } 