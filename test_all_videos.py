"""Quick test: run 3 different videos through the single-player pipeline."""
import traceback
from ai_engine.pose_analyzer import process_video
from ai_engine.comparison_engine import compare_user_with_reference
from ai_engine.feedback_generator import generate_feedback
from app import app

videos = [
    "uploads/video_3_36170845.mp4",
    "uploads/video_4_2cf04cd6.mp4",
    "uploads/video6_03f3dfbb.mp4",
]

with app.app_context():
    for v in videos:
        print(f"\n=== {v} ===")
        try:
            res = process_video(v, "quicktest.mp4", "processed")
            score = res["final_score"]
            shot = res["shot_type"]
            frames = res["total_frames_analyzed"]
            print(f"  Score: {score}  Shot: {shot}  Frames analyzed: {frames}")
            if frames > 0:
                comp = compare_user_with_reference(res)
                fb = generate_feedback(comp)
                print(f"  Similarity: {comp.get('similarity_score', 0)}")
                print(f"  Feedback items: {len(fb)}")
            else:
                print("  WARNING: No frames analyzed!")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
