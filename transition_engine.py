from database import db
def transitions_for(user_id): return db.transitions(user_id)
