from ultralytics import YOLO
import pickle
import os
import uuid
from datetime import datetime

# Get list of files from dataset folder
dataset_path = "dataset"
video_files = [f for f in os.listdir(dataset_path) if f.endswith(('.mp4', '.avi', '.mov'))]

# Process each video file
for video_file in video_files:
    video_path = os.path.join(dataset_path, video_file)
    print(f"Processing {video_file}...")

    # Load the YOLO model
    model = YOLO("yolo11n.pt")

    # Perform tracking with the model
    results = model.track(video_path)
    print(results)

    # Save results to avoid reprocessing

    # with open(f'{video_file.split(".")[0]}.pkl', 'wb') as f:
    #     pickle.dump(results, f)
