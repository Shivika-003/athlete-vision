import traceback
from ai_engine.match_analyzer import process_match_video
from app import app

try:
    print("Testing processing on video_4_2cf04cd6.mp4")
    with app.app_context():
        res = process_match_video(
            input_path=r"c:\Users\91600\Downloads\major project\videob.mp4",
            output_filename="test_match_output_videob.mp4",
            output_dir="processed"
        )
    print("SUCCESS")
    print(res)
except Exception as e:
    print("FAILED")
    traceback.print_exc()
