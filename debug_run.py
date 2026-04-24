import traceback
from ai_engine.pose_analyzer import process_video
from app import app

try:
    print("Testing processing on video_2_8a093ae2.mp4")
    with app.app_context():
        res = process_video(
            input_path=r"c:\Users\91600\Downloads\major project\uploads\video_2_8a093ae2.mp4",
            output_filename="test_output",
            output_dir="processed"
        )
    print("SUCCESS")
    print(res)
except Exception as e:
    print("FAILED")
    traceback.print_exc()
