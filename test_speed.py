import sys, time
sys.path.insert(0, '.')
start = time.time()
from ai_engine.pose_analyzer import process_video
result = process_video('uploads/video_5_bbfe5947.mp4', 'test_speed.mp4', 'processed')
elapsed = time.time() - start
print(f'SUCCESS in {elapsed:.1f} seconds')
print(f'Shot type: {result.get("shot_type")}')
print(f'Final score: {result.get("final_score")}')
print(f'Contact angles: {result.get("contact_angles")}')
