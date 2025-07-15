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


# Configuration for quick test
YOLO_WEIGHTS   = pathlib.Path('../weights/yolov8x_person_face.pt')
REID_WEIGHTS   = pathlib.Path('../weights/reid/osnet_x1_0_msmt17.pt')
VIDEO          = pathlib.Path('../2min.mp4')
OUT_DIR        = pathlib.Path('../results/mivolo_quick_test'); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Quick test settings - optimized for accuracy like original MIVOLO
SKIP_FRAMES = 2           # Process every 2nd frame (less skipping for better tracking)
# DETECTION_SIZE = 416    # Removed - use original frame size for better accuracy
CONF_THRESHOLD = 0.4     # Lower confidence like original MIVOLO (was 0.5)
IOU_THRESHOLD = 0.5       # Lower IOU threshold (was 0.7)
MAX_FRAMES = 300          # Only process first 10 seconds

# Auto-detect device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda:0"

# Test only 2 trackers for speed
TRACKERS = {
    'bytetrack':  (ByteTrack,   dict()),                 
    'botsort':    (BotSort,     dict(reid_weights=REID_WEIGHTS, device=device, half=False)),
}

print(f"=== MIVOLO Quick Test Configuration (Accuracy Mode) ===")
print(f"Device: {device}")
print(f"Half precision: {use_half}")
print(f"Frame skipping: Every {SKIP_FRAMES} frames")
print(f"Using original frame resolution (no resizing)")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IOU threshold: {IOU_THRESHOLD}")
print(f"Max frames: {MAX_FRAMES} (about 10 seconds)")
print(f"Testing trackers: {list(TRACKERS.keys())}")

# Initialize MIVOLO-style detector with better accuracy settings
try:
    detector = MivoloDetector(
        weights=str(YOLO_WEIGHTS),
        device=device,
        half=use_half,
        verbose=False,
        conf_thresh=CONF_THRESHOLD,
        iou_thresh=IOU_THRESHOLD
    )
    print(f"✅ MIVOLO Detector initialized successfully on {device}")
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
    writer = cv2.VideoWriter(str(OUT_DIR/f'{name}_test.mp4'), fourcc, fps, (width, height))
    
    last_tracks = []

    while cap.isOpened():
        ok, frame = cap.read()
        frames += 1
        if not ok or frames > MAX_FRAMES: break
        
        # Process tracking only on selected frames
        if frames % SKIP_FRAMES == 1:
            processed_frames += 1
            
            # Use original frame resolution for better accuracy (no resizing)
            # detection_frame = cv2.resize(frame, (DETECTION_SIZE, DETECTION_SIZE))
            
            # Get detections using MIVOLO-style detector
            try:
                results = detector.predict(frame)  # Use original frame, not resized
                
                if len(results.boxes) > 0:
                    # Extract detection data (no scaling needed since using original frame)
                    dets = results.boxes.xyxy.cpu().numpy()
                    # scale_x = width / DETECTION_SIZE  # No scaling needed
                    # scale_y = height / DETECTION_SIZE
                    # dets[:, [0, 2]] *= scale_x
                    # dets[:, [1, 3]] *= scale_y
                    
                    confs = results.boxes.conf.cpu().numpy()
                    clss = results.boxes.cls.cpu().numpy()
                    
                    detections = np.column_stack((dets, confs, clss))
                else:
                    detections = np.empty((0, 6))
            except Exception as e:
                print(f"Detection error: {e}")
                detections = np.empty((0, 6))

            # Update tracker
            try:
                tracks = tracker.update(detections, frame)
                last_tracks = tracks
            except Exception as e:
                print(f"Tracking error: {e}")
                tracks = last_tracks
        else:
            tracks = last_tracks
        
        # Draw tracks on frame
        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                ids.add(int(track_id))
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f'ID:{int(track_id)}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        writer.write(frame)
        
        if frames % 60 == 0:
            detections_count = len([t for t in tracks if len(t) >= 7])
            print(f"Frame {frames}/{MAX_FRAMES}, Processed: {processed_frames}, Active detections: {detections_count}")

    cap.release()
    writer.release()
    
    total_time = time.time() - t0
    print(f"✅ {name}: {total_time:.1f}s, {len(ids)} IDs, {frames} frames")

print(f"\n✅ MIVOLO Quick test completed!")
print(f"If this worked well, run the full benchmark with: python mivolo_bench.py")
print(f"Test videos saved to: {OUT_DIR}") 