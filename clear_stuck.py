from app import app, db
from models import VideoRecord

with app.app_context():
    stuck_records = VideoRecord.query.filter_by(status='processing').all()
    print(f"Found {len(stuck_records)} stuck records.")
    for record in stuck_records:
        db.session.delete(record)
    db.session.commit()
    print("Deleted stuck records.")
