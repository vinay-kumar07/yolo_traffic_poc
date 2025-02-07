import cv2
import imagehash
from PIL import Image
import os

def hash_frame(frame):
    """Compute perceptual hash (pHash) for a given frame."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)

def count_unique_frames(video_path):
    """Count the number of unique frames in a video using perceptual hashing."""
    cap = cv2.VideoCapture(video_path)
    unique_hashes = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_hash = hash_frame(frame)
        unique_hashes.add(frame_hash)
    
    cap.release()
    return len(unique_hashes)

video_files = [f for f in os.listdir("/home/vinaykumar/vkumar/yolo_traffic_poc/dataset") if f.endswith(".mp4")]
print(f"Total number of videos: {len(video_files)}")

total_unique_frames = 0
for video_file in video_files:
    unique_frame_count = count_unique_frames(f"/home/vinaykumar/vkumar/yolo_traffic_poc/dataset/{video_file}")
    total_unique_frames += unique_frame_count
    print(f"Number of unique frames: {unique_frame_count}")

print(f"Total unique frames: {total_unique_frames}")
