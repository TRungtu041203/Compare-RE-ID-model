import os
import cv2, time, json, pathlib
import numpy as np
import random
import gc
import argparse
from typing import Dict, Union, List

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from boxmot import BotSort, DeepOcSort, ByteTrack, BoostTrack

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


# Parse command line arguments
parser = argparse.ArgumentParser(description='Fair MIVOLO Tracking Benchmark')
parser.add_argument('--config', type=str, default='tracker_configs.json', 
                    help='Configuration file to use (default: tracker_configs.json)')
parser.add_argument('--output-dir', type=str, default='../results/bench_ClipReID',
                    help='Output directory for results')
parser.add_argument('--video', type=str, default='../2min.mp4',
                    help='Path to input video file (default: ../2min.mp4)')
args = parser.parse_args()


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
        print(f"Detector classes: {self.detector_names}")

        # init yolo.predictor
        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose}

    def predict(self, image: Union[np.ndarray, str], filter_faces: bool = True) -> Results:
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        
        # Filter out faces if requested (keep only persons)
        if filter_faces and len(results.boxes) > 0:
            # Assuming class 0 = person, class 1 = face (check your model)
            person_mask = results.boxes.cls == 0  # Keep only persons
            if person_mask.any():
                results.boxes = results.boxes[person_mask]
            else:
                # No persons detected, create empty result
                results.boxes = results.boxes[:0]  # Empty but same structure
        
        return results


def clear_gpu_memory():
    """Clear GPU memory to ensure fair comparison"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_system_stats():
    """Get current system stats for monitoring"""
    stats = {}
    if torch.cuda.is_available():
        stats['gpu_memory_used'] = torch.cuda.memory_allocated() / 1e9  # GB
        stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1e9  # GB
    return stats


def load_tracker_configs(config_file: str = "tracker_configs.json"):
    """Load tracker configurations from JSON file"""
    config_path = pathlib.Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def prepare_tracker_params(params: dict, reid_weights_path: pathlib.Path, device: str):
    """Prepare tracker parameters by resolving 'auto' values"""
    prepared_params = params.copy()
    
    # Replace 'auto' values with actual values
    if prepared_params.get('reid_weights') == 'auto':
        prepared_params['reid_weights'] = reid_weights_path
    if prepared_params.get('device') == 'auto':
        prepared_params['device'] = device
    
    return prepared_params


def convert_paths_to_strings(obj):
    """Convert Path objects to strings for JSON serialization"""
    if isinstance(obj, pathlib.Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    else:
        return obj


# Configuration
YOLO_WEIGHTS   = pathlib.Path('../weights/yolov8x_person_face.pt')
REID_WEIGHTS   = pathlib.Path('../weights/reid/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth')
VIDEO          = pathlib.Path(args.video)
OUT_DIR        = pathlib.Path(args.output_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Validate input files
if not YOLO_WEIGHTS.exists():
    print(f"❌ YOLO weights not found: {YOLO_WEIGHTS.absolute()}")
    exit(1)

if not REID_WEIGHTS.exists():
    print(f"❌ ReID weights not found: {REID_WEIGHTS.absolute()}")
    exit(1)

if not VIDEO.exists():
    print(f"❌ Video file not found: {VIDEO.absolute()}")
    exit(1)

# Load configuration from JSON file
try:
    config = load_tracker_configs(args.config)
    test_settings = config['test_settings']
    tracker_configs = config['tracker_configs']
    
    print(f"✅ Loaded configuration from {args.config}")
    print(f"   Found {len(tracker_configs)} tracker configs")
    print(f"   Output directory: {OUT_DIR}")
    print(f"\nAvailable configurations:")
    for key, cfg in tracker_configs.items():
        print(f"  - {cfg['name']}: {cfg['description']}")
        
except Exception as e:
    print(f"❌ Failed to load configuration from {args.config}: {e}")
    print("Using default configurations...")
    
    # Fallback to default configurations
    test_settings = {
        "skip_frames": 5,
        "conf_threshold": 0.3,
        "iou_threshold": 0.5,
        "max_frames": None,
        "filter_faces": True,
        "randomize_order": True
    }
    
    tracker_configs = {
        'bytetrack_default': {
            'class': 'ByteTrack',
            'params': {},
            'name': 'ByteTrack-Default',
            'description': 'Default ByteTrack'
        },
        'botsort_default': {
            'class': 'BotSort',
            'params': {'reid_weights': 'auto', 'device': 'auto', 'half': False},
            'name': 'BotSort-Default',
            'description': 'Default BotSort'
        }
    }

# Extract test settings
SKIP_FRAMES = test_settings['skip_frames']
CONF_THRESHOLD = test_settings['conf_threshold']
IOU_THRESHOLD = test_settings['iou_threshold']
MAX_FRAMES = test_settings['max_frames']
FILTER_FACES = test_settings['filter_faces']
RANDOMIZE_ORDER = test_settings['randomize_order']

# Auto-detect device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda:0"

# Convert config to tracker objects
TRACKER_CLASSES = {
    'ByteTrack': ByteTrack,
    'BotSort': BotSort,
    'DeepOcSort': DeepOcSort,
    'BoostTrack': BoostTrack
}

TRACKERS = {}
for key, cfg in tracker_configs.items():
    tracker_class = TRACKER_CLASSES.get(cfg['class'])
    if tracker_class is None:
        print(f"⚠️  Unknown tracker class: {cfg['class']}, skipping {key}")
        continue
        
    prepared_params = prepare_tracker_params(cfg['params'], REID_WEIGHTS, device)
    
    TRACKERS[key] = {
        'class': tracker_class,
        'params': prepared_params,
        'name': cfg['name'],
        'description': cfg['description']
    }

print(f"\n=== Fair MIVOLO Benchmark Configuration ===")
print(f"Video file: {VIDEO}")
print(f"Device: {device}")
print(f"Half precision: {use_half}")
print(f"Frame skipping: Every {SKIP_FRAMES} frames")
print(f"Original frame resolution")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IOU threshold: {IOU_THRESHOLD}")
print(f"Filter faces: {FILTER_FACES} (only track persons)")
print(f"Max frames limit: {MAX_FRAMES if MAX_FRAMES else 'Full video'}")
print(f"Randomize order: {RANDOMIZE_ORDER}")
print(f"Active tracker configurations: {len(TRACKERS)}")

# Initialize MIVOLO-style detector once
print(f"\n=== Initializing Detector ===")
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

# Randomize tracker order for fair comparison if enabled
tracker_order = list(TRACKERS.keys())
if RANDOMIZE_ORDER:
    random.shuffle(tracker_order)
    print(f"\n=== Randomized Test Order ===")
else:
    print(f"\n=== Test Order (Fixed) ===")

for i, tracker_key in enumerate(tracker_order, 1):
    print(f"{i}. {TRACKERS[tracker_key]['name']}")

# Test each tracker
all_results = []

for test_num, tracker_key in enumerate(tracker_order, 1):
    config = TRACKERS[tracker_key]
    name = config['name']
    Cls = config['class']
    kw = config['params']
    
    print(f"\n=== Test {test_num}/{len(tracker_order)}: {name.upper()} ===")
    print(f"Description: {config['description']}")
    print(f"Parameters: {kw}")
    
    # Clear memory before each test for fairness
    clear_gpu_memory()
    start_stats = get_system_stats()
    print(f"Pre-test GPU memory: {start_stats.get('gpu_memory_used', 0):.2f}GB")
    
    cap = cv2.VideoCapture(str(VIDEO))
    
    try:
        tracker = Cls(**kw)
        print(f"✅ {name} tracker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize {name}: {e}")
        print(f"Parameters that caused error: {kw}")
        continue
        
    t0, frames, processed_frames, ids = time.time(), 0, 0, set()
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use safe filename
    safe_name = tracker_key.replace('_', '-')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUT_DIR/f'{safe_name}.mp4'), fourcc, fps, (width, height))
    
    last_tracks = []

    mot_results_path = OUT_DIR / f'{safe_name}_mot.txt'
    mot_results = []  # List to store all MOT results for this tracker

    while cap.isOpened():
        ok, frame = cap.read()
        frames += 1
        if not ok: break
        if MAX_FRAMES and frames > MAX_FRAMES: break
        
        # Process tracking only on selected frames
        if frames % SKIP_FRAMES == 0:
            processed_frames += 1
            
            # Get detections using MIVOLO-style detector
            try:
                results = detector.predict(frame, filter_faces=FILTER_FACES)
                
                if len(results.boxes) > 0:
                    # Extract detection data (only persons, no faces)
                    dets = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    clss = results.boxes.cls.cpu().numpy()
                    
                    detections = np.column_stack((dets, confs, clss))
                else:
                    detections = np.empty((0, 6))
            except Exception as e:
                print(f"Detection error at frame {frames}: {e}")
                detections = np.empty((0, 6))

            # Update tracker
            try:
                tracks = tracker.update(detections, frame)
                tracks = tracker.update(detections, frame)
                for track in tracks:
                    if len(track) >= 7:
                        x1, y1, x2, y2, track_id, conf, cls = track[:7]
                        bb_left = x1
                        bb_top = y1
                        bb_width = x2 - x1
                        bb_height = y2 - y1
                        # Store as (id, frame, bb_left, bb_top, bb_width, bb_height, conf, cls)
                        mot_results.append([
                            int(track_id), processed_frames, bb_left, bb_top, bb_width, bb_height, conf, int(cls)
                        ])
                last_tracks = tracks
            except Exception as e:
                print(f"Tracking error at frame {frames}: {e}")
                tracks = last_tracks
        else:
            tracks = last_tracks
        
        # Draw tracks on frame (only persons, no faces)
        active_detections = 0
        for track in tracks:
            if len(track) >= 7:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                ids.add(int(track_id))
                active_detections += 1
                
                # Draw bounding box for person only
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f'Person-{int(track_id)}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        writer.write(frame)
        
        # Print progress every 150 frames
        if frames % 150 == 0:
            elapsed = time.time() - t0
            est_total = elapsed * (cap.get(cv2.CAP_PROP_FRAME_COUNT) / frames) if frames > 0 else 0
            current_stats = get_system_stats()
            print(f"Frame {frames}, Active tracks: {active_detections}, "
                  f"GPU mem: {current_stats.get('gpu_memory_used', 0):.2f}GB, "
                  f"ETA: {est_total - elapsed:.1f}s")

    cap.release()
    writer.release()
    # Sort results: first by ID, then by frame
    mot_results.sort(key=lambda x: (x[0], x[1]))  # (id, frame, ...)
    with open(mot_results_path, 'w') as mot_results_file:
        for id, frame, bb_left, bb_top, bb_width, bb_height, conf, cls in mot_results:
            mot_results_file.write(
                f"{frame},{id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{conf:.2f},{cls},1\n"
            )
    print(f"✅ MOT results saved to {mot_results_path}")
    total_time = time.time() - t0
    fps_processing = frames / total_time
    fps_detection = processed_frames / total_time
    
    end_stats = get_system_stats()
    
    result = {
        'tracker': tracker_key,
        'tracker_name': name,
        'description': config['description'],
        'test_order': test_num,
        'device': device,
        'half_precision': use_half,
        'total_time': round(total_time, 2),
        'fps_overall': round(fps_processing, 2), 
        'fps_detection': round(fps_detection, 2),
        'unique_ids': len(ids), 
        'total_frames': frames,
        'processed_frames': processed_frames,
        'skip_frames': SKIP_FRAMES,
        'conf_threshold': CONF_THRESHOLD,
        'filter_faces': FILTER_FACES,
        'pre_test_gpu_memory': start_stats.get('gpu_memory_used', 0),
        'post_test_gpu_memory': end_stats.get('gpu_memory_used', 0),
        'tracker_params': convert_paths_to_strings(kw)
    }
    
    all_results.append(result)
    
    print(f"✅ Stats for {name}:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall FPS: {fps_processing:.2f}")
    print(f"  Detection FPS: {fps_detection:.2f}")
    print(f"  Unique IDs: {len(ids)}")
    print(f"  Frames: {frames} (processed: {processed_frames})")
    print(f"  GPU memory change: {start_stats.get('gpu_memory_used', 0):.2f}GB → {end_stats.get('gpu_memory_used', 0):.2f}GB")
    
    # Save individual result
    (OUT_DIR/'results.jsonl').open('a').write(json.dumps(result) + '\n')
    
    # Clear memory after each test
    del tracker
    clear_gpu_memory()
    print(f"Memory cleared after {name}")

print(f"\n🎉 All tests completed! ===")
print(f"Results saved to: {OUT_DIR}")
print(f"Videos: {list(OUT_DIR.glob('*.mp4'))}")

print(f"\n=== Final Performance Comparison ===")
print(f"{'Tracker':<25} {'Order':<6} {'Time(s)':<8} {'FPS':<6} {'IDs':<4} {'GPU Δ(GB)':<10}")
print("-" * 75)

for result in sorted(all_results, key=lambda x: x['total_time']):
    gpu_delta = result['post_test_gpu_memory'] - result['pre_test_gpu_memory']
    print(f"{result['tracker_name']:<25} {result['test_order']:<6} "
          f"{result['total_time']:<8} {result['fps_overall']:<6} "
          f"{result['unique_ids']:<4} {gpu_delta:+.2f}")

print(f"\n📊 Performance Analysis:")
print(f"✅ Tests run in {'random' if RANDOMIZE_ORDER else 'fixed'} order")
print(f"✅ GPU memory cleared between tests")
print(f"✅ Only person detections tracked (faces filtered out)")
print(f"✅ Multiple parameter configurations tested")
print(f"✅ Configurations loaded from tracker_configs.json")

# Save detailed summary
summary = {
    'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'test_order': [TRACKERS[k]['name'] for k in tracker_order],
    'randomized': RANDOMIZE_ORDER,
    'settings': {
        'skip_frames': SKIP_FRAMES,
        'conf_threshold': CONF_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'filter_faces': FILTER_FACES,
        'device': device,
        'half_precision': use_half,
        'max_frames': MAX_FRAMES
    },
    'tracker_configs': convert_paths_to_strings({k: {
        'params': v['params'],
        'name': v['name'],
        'description': v['description']
    } for k, v in TRACKERS.items()}),
    'results': all_results
}

# Convert all Path objects to strings before JSON serialization
summary_for_json = convert_paths_to_strings(summary)

with open(OUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary_for_json, f, indent=2)

print(f"\n📁 Output files:")
print(f"  📊 summary.json - Complete test results and configuration")
print(f"  📊 results.jsonl - Individual tracker results")
print(f"  🎥 [tracker-name].mp4 - Output videos with person tracking only") 

 
