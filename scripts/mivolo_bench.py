import os
import cv2, time, json, pathlib
import numpy as np
from typing import Dict, Union

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from boxmot import BotSort, DeepOcSort, ByteTrack, BoostTrack

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


class MivoloDetector:
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
    ):
        self.yolo = YOLO(weights)
        self.yolo.fuse()

        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.yolo.model = self.yolo.model.half()

        self.detector_names: Dict[int, str] = self.yolo.model.names

        # init yolo.predictor
        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose}

    def predict(self, image: Union[np.ndarray, str]) -> Results:
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return results


# Configuration
YOLO_WEIGHTS   = pathlib.Path('../weights/yolov8x_person_face.pt')
REID_WEIGHTS   = pathlib.Path('../weights/reid/osnet_x1_0_msmt17.pt')
VIDEO          = pathlib.Path('../2min.mp4')
OUT_DIR        = pathlib.Path('../results/mivolo_api'); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Speed optimization settings
SKIP_FRAMES = 3           # Process every 3rd frame (3x faster)
#DETECTION_SIZE = 640      # Resize for detection (smaller = faster)
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5      # Higher confidence = fewer detections = faster
MAX_FRAMES = None         # Set to number to limit total frames (e.g., 300 for testing)

# Auto-detect device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda:0"

TRACKERS = {
    'bytetrack':  (ByteTrack,   dict()),                 # motion-only
    'botsort':    (BotSort,     dict(reid_weights=REID_WEIGHTS, device=device, half=False)),
    'deepocsort': (DeepOcSort,  dict(reid_weights=REID_WEIGHTS, device=device, half=False)),
    'boosttrack': (BoostTrack,  dict(reid_weights=REID_WEIGHTS, device=device, half=False)),
}

print(f"=== MIVOLO-Style Benchmark Configuration ===")
print(f"Device: {device}")
print(f"Half precision: {use_half}")
print(f"Frame skipping: Every {SKIP_FRAMES} frames")
print(f"Using original frame resolution (since YOLO - Ultralytics resize the frame by default)")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"Max frames limit: {MAX_FRAMES if MAX_FRAMES else 'None'}")


# Initialize MIVOLO-style detector
try:
    detector = MivoloDetector(
    weights=str(YOLO_WEIGHTS),
    device=device,
    half=use_half,
    verbose=False,
    conf_thresh=CONF_THRESHOLD,
    iou_thresh=IOU_THRESHOLD
    )
    print(f"✅ Detector initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    exit(1)

for name, (Cls, kw) in TRACKERS.items():
    print(f"\n=== Testing {name.upper()} ===")
    cap = cv2.VideoCapture(str(VIDEO))
    
    try:
        tracker = Cls(**kw)
        print(f"✅ {name} tracker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize {name}: {e}")
        continue
        
    t0, frames, processed_frames, ids = time.time(), 0, 0, set()
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUT_DIR/f'{name}.mp4'), fourcc, fps, (width, height))
    
    # Store last tracking results for interpolation
    last_tracks = []

    while cap.isOpened():
        ok, frame = cap.read()
        frames += 1
        if not ok: break
        if MAX_FRAMES and frames > MAX_FRAMES: break
        
        # Process tracking only on selected frames
        if frames % SKIP_FRAMES == 1:  # Process 1st, 4th, 7th, etc.
            processed_frames += 1
            
            # Resize frame for faster detection
            #detection_frame = cv2.resize(frame, (DETECTION_SIZE, DETECTION_SIZE))

            
            # Get detections using MIVOLO-style detector
            try:
                results = detector.predict(frame)
                
                if len(results.boxes) > 0:
                    # Extract detection data and scale back to original size
                    dets = results.boxes.xyxy.cpu().numpy()
                    #scale_x = width / DETECTION_SIZE
                    #scale_y = height / DETECTION_SIZE
                    #dets[:, [0, 2]] *= scale_x  # Scale x coordinates
                    #dets[:, [1, 3]] *= scale_y  # Scale y coordinates
                    
                    confs = results.boxes.conf.cpu().numpy()
                    clss = results.boxes.cls.cpu().numpy()
                    
                    # Create detections array: [x1, y1, x2, y2, conf, class]
                    detections = np.column_stack((dets, confs, clss))
                else:
                    detections = np.empty((0, 6))
            except Exception as e:
                print(f"Detection error at frame {frames}: {e}")
                detections = np.empty((0, 6))

            # Update tracker
            try:
                tracks = tracker.update(detections, frame)
                last_tracks = tracks  # Store for interpolation
            except Exception as e:
                print(f"Tracking error at frame {frames}: {e}")
                tracks = last_tracks
        else:
            # Use last tracking results for skipped frames
            tracks = last_tracks
        
        # Draw tracks on frame
        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                ids.add(int(track_id))
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                
                # Draw track ID
                cv2.putText(frame, f'ID:{int(track_id)}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        writer.write(frame)
        
        # Print progress every 150 frames
        if frames % 150 == 0:
            elapsed = time.time() - t0
            est_total = elapsed * (cap.get(cv2.CAP_PROP_FRAME_COUNT) / frames) if frames > 0 else 0
            print(f"Frame {frames}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, "
                  f"Processed: {processed_frames}, "
                  f"ETA: {est_total - elapsed:.1f}s")

    cap.release()
    writer.release()
    
    total_time = time.time() - t0
    fps_processing = frames / total_time
    fps_detection = processed_frames / total_time
    
    stats = {
        'tracker': name, 
        'device': device,
        'half_precision': use_half,
        'total_time': round(total_time, 2),
        'fps_overall': round(fps_processing, 2), 
        'fps_detection': round(fps_detection, 2),
        'unique_ids': len(ids), 
        'total_frames': frames,
        'processed_frames': processed_frames,
        'skip_frames': SKIP_FRAMES,
        #'detection_size': DETECTION_SIZE,
        'conf_threshold': CONF_THRESHOLD
    }
    
    print(f"✅ Stats for {name}:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall FPS: {fps_processing:.2f}")
    print(f"  Detection FPS: {fps_detection:.2f}")
    print(f"  Unique IDs: {len(ids)}")
    print(f"  Frames: {frames} (processed: {processed_frames})")
    
    (OUT_DIR/'stats.jsonl').open('a').write(json.dumps(stats) + '\n')

print(f"\n🎉 All tests completed! ===")
print(f"Results saved to: {OUT_DIR}")
print(f"Videos: {list(OUT_DIR.glob('*.mp4'))}")
print(f"Stats: {OUT_DIR/'stats.jsonl'}")

print(f"\n=== Performance Summary ===")
if (OUT_DIR/'stats.jsonl').exists():
    with open(OUT_DIR/'stats.jsonl', 'r') as f:
        all_stats = [json.loads(line) for line in f if line.strip()]
    
    print(f"{'Tracker':<12} {'Device':<6} {'Time(s)':<8} {'FPS':<6} {'IDs':<4} {'Speedup':<8}")
    print("-" * 55)
    for stat in all_stats:
        speedup = f"{SKIP_FRAMES}x" if stat.get('skip_frames', 1) > 1 else "1x"
        device_short = stat.get('device', 'cpu')[:4] 
        print(f"{stat['tracker']:<12} {device_short:<6} {stat['total_time']:<8} {stat['fps_overall']:<6} {stat['unique_ids']:<4} {speedup:<8}") 