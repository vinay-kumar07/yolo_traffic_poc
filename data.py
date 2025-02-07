from ultralytics import YOLO
import pickle
import os
import uuid
from datetime import datetime
import sys

QUESTION_ID = {
    '2': '23a18951-0b6d-480f-8c5b-a616a8d9a982',
    '1': 'b7b46977-7eb6-46ab-8a49-0994847318e5',
    '0': 'fc84bd7d-834f-4af1-9d5b-ba441997d6d4',
    '5': '4080204b-bcd5-47bd-a690-022c64af4586',
    '7': 'b652a016-80a5-496f-ae20-cd6ac5eb3a05',
}

NAME = {
    '2': 'car',
    '1': 'cyclist',
    '0': 'person',
    '5': 'bus',
    '7': 'truck'
}

COLOR = {
    '2': '#5F9EA0',
    '1': '#228B22',
    '0': '#FF4500',
    '5': '#4682B4',
    '7': '#DAA520'
}

results_path = "results"
result_files = [f for f in os.listdir(results_path) if f.endswith(('.pkl'))]

for result_file in result_files:
    video_name = f"{result_file.split('.')[0]}.mp4"
    video_path = os.path.join("dataset", video_name)

    import subprocess
    import json

    # Get video metadata using ffprobe
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        video_path
    ]
    
    try:
        probe_output = subprocess.check_output(ffprobe_cmd).decode('utf-8')
        video_info = json.loads(probe_output)
        
        # Find video stream
        video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Calculate fps from frame rate fraction
        fps_parts = video_stream['r_frame_rate'].split('/')
        fps = int(fps_parts[0]) / int(fps_parts[1])
        
    except (subprocess.CalledProcessError, KeyError, StopIteration) as e:
        print(f"Error getting video metadata for {video_path}: {str(e)}")
        continue

    with open(os.path.join(results_path, result_file), 'rb') as f:
        results = pickle.load(f)
        answer = {}
        for frame_idx, result in enumerate(results):
            if result.boxes.id is None:
                continue
            box_data = result.boxes.xyxy.tolist()
            question_id = result.boxes.cls.tolist()
            track_id = result.boxes.id.tolist()

            for box, question_id, track_id in zip(box_data, question_id, track_id):
                if str(int(question_id)) not in QUESTION_ID:
                    continue
                x1, y1, x2, y2 = box
                if track_id not in answer:
                    answer[track_id] = {
                        "question_id": QUESTION_ID[str(int(question_id))],
                        "start": 0,
                        "label": NAME[str(int(question_id))],
                        "option_type": "BoundingBox",
                        "color": COLOR[str(int(question_id))],
                        "_id": str(int(track_id)),
                        "attributes": [],
                        "auto_label": False,
                        "isMarkedIncorrect": False,
                        "groupId": "",
                        "image_path": "",
                        "wiai_image_path": "",
                        "wiai_jid": "",
                        "box_geometry": "",
                        "end": 60,
                        "startFrame": 0,
                        "fps": fps,
                        "answer": {},
                        "frames": {},
                        "zIndex": 2,
                        "question_id": QUESTION_ID[str(int(question_id))],
                        "fileId": video_name,
                        "selectedText": "",
                        "height": height,
                        "width": width,
                        "opacity": 1
                    }
                
                answer[track_id]["frames"][str(frame_idx)] = {
                    "answer": {
                        "ymin": y1,
                        "xmin": x1,
                        "ymax": y2,
                        "xmax": x2
                    },
                    "fps": fps,
                    "isManualAnnotation": False,
                    "frame": frame_idx,
                    "timestamp": (frame_idx+1)/fps
                }

        print(answer)

        questions = {}
        for track_id, annotation in answer.items():
            question_id = annotation["question_id"]
            if question_id not in questions:
                questions[question_id] = []    
            questions[question_id].append(annotation)

        data = []
        for question_id, answer in questions.items():
            data.append({
                "question_id": question_id,
                "answer": answer
            })

        for item in data:
            for answer in item["answer"]:
                frame_numbers = [int(frame) for frame in answer["frames"].keys()]
                if frame_numbers:  # Only update if frames exist
                    answer["startFrame"] = min(frame_numbers)
                    answer["start"] = min(frame_numbers)/fps
                    answer["end"] = max(frame_numbers)/fps
                    answer["answer"] = answer["frames"][str(min(frame_numbers))]["answer"]
                    answer["answer"]["rotation"] = 0

        final_data = {
            "data": data,
            "file_id": video_name,
            "is_draft": True,
            "draft_time": 59
        }

        import json
        # print(json.dumps(final_data, indent=4))

        with open(f'{video_name.split(".")[0]}.json', 'w') as f:
            json.dump(final_data, f, indent=4)
