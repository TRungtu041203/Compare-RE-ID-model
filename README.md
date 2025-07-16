# Fair MIVOLO Tracking Benchmark on 4 algo: BoTSORT, DeepOCSORT, ByteTrack, BoostTrack

## 🎯 **Key Features**

### **1. Fair Comparison**
- ✅ **Randomized test order** - Each run uses different sequence to eliminate order bias
- ✅ **GPU memory clearing** - Memory cleaned between tests for consistent performance  
- ✅ **System monitoring** - Tracks GPU memory usage to detect resource leaks
- ✅ **Isolated tests** - Each tracker runs independently

### **2. Configurable Parameters**
- ✅ **JSON configuration** - Easy parameter tuning without code changes
- ✅ **Multiple parameter sets** - Test different configurations per tracker
- ✅ **Parameter validation** - Clear error messages for invalid settings

### **3. Face Filtering**
- ✅ **Person-only tracking** - Filters out face detections from MIVOLO dual output
- ✅ **Class-based filtering** - Uses YOLO class IDs (0=person, 1=face)
- ✅ **Clean visualization** - Only shows person bounding boxes in output videos

## 📁 **Files**

### **Main Scripts**
- `fair_mivolo_bench.py` - Main benchmark script with all features
- `mivolo_accuracy_test.py` - Maximum accuracy test (processes every frame)
- `mivolo_quick_test.py` - Quick validation test

### **Configuration Files**
- `tracker_configs.json` - Full configuration with all tracker variants
- `test_configs.json` - Smaller config for quick testing (3 trackers, 300 frames)

## 🚀 **Usage**

### **Quick Test (3-5 minutes)**
```bash
python fair_mivolo_bench.py --config test_configs.json
```

### **Full Benchmark (20-40 minutes)**
```bash
python fair_mivolo_bench.py --config tracker_configs.json
```

### **Custom Configuration**
```bash
python fair_mivolo_bench.py --config my_config.json --output-dir ../results/my_test
```

## ⚙️ **Configuration Format**

### **test_settings**
```json
{
  "test_settings": {
    "skip_frames": 3,
    "conf_threshold": 0.4,
    "iou_threshold": 0.5,
    "max_frames": null,
    "filter_faces": true,
    "randomize_order": true
  }
}
```

### **tracker_configs**
```json
{
  "test_settings": {
    "skip_frames": 1,
    "conf_threshold": 0.4,
    "iou_threshold": 0.7,
    "max_frames": null,
    "filter_faces": true,
    "randomize_order": true
  },
  "tracker_configs": {
    "bytetrack_aggressive": {
      "class": "ByteTrack", 
      "name": "ByteTrack-Aggressive",
      "params": {
        "track_thresh": 0.5,
        "min_conf": 0.1,
        "track_buffer": 300,
        "match_thresh": 0.8
      },
      "description": "ByteTrack with aggressive tracking parameters"
    },
    "botsort_strict": {
      "class": "BotSort",
      "name": "BotSort-Strict",
      "params": {
        "reid_weights": "auto", 
        "device": "auto",
        "half": false,
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 300,
        "match_thresh": 0.8,
        "fuse_first_associate": false,
        "with_reid": true
      },
      "description": "BotSort with strict matching thresholds"
    },
    "botsort_loose": {
      "class": "BotSort",
      "name": "BotSort-Loose",
      "params": {
        "reid_weights": "auto",
        "device": "auto", 
        "half": false,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 300,
        "match_thresh": 0.6,
        "proximity_thresh": 0.7,
        "appearance_thresh": 0.15,
        "fuse_first_associate": false,
        "with_reid": true
      },
      "description": "BotSort with loose matching for crowded scenes"
    },
    "deepocsort_strict": {
      "class": "DeepOcSort", 
      "name": "DeepOcSort-Strict",
      "params": {
        "reid_weights": "auto",
        "device": "auto",
        "half": false,
        "det_thresh": 0.5,
        "max_age": 300,
        "inertia": 0.4,
        "cmc_off": true
      },
      "description": "DeepOcSort with strict association"
    },
    "boosttrack_default": {
      "class": "BoostTrack",
      "name": "BoostTrack-Default", 
      "params": {
        "reid_weights": "auto",
        "device": "auto",
        "half": false
      },
      "description": "BoostTrack with default parameters"
    },
    "boosttrack_optimized": {
      "class": "BoostTrack",
      "name": "BoostTrack-Optimized",
      "params": {
        "reid_weights": "auto",
        "device": "auto",
        "half": false,
        "max_age": 300,
        "min_hits": 6,
        "det_thresh": 0.6,
        "min_box_area": 800,
        "iou_threshold": 0.45,
        "use_ecc": false,
        "cmc_method": "None",
        "lambda_iou": 0.55,
        "lambda_mhd": 0.25,
        "lambda_shape": 0.2,
        "dlo_boost_coef": 0.55,
        "use_rich_s": true,
        "use_sb": true,
        "use_vt": true,
        "with_reid": true
      },
      "description": "BoostTrack with optimized parameters for accuracy"
    }
  },
  "parameter_descriptions": {
    "match_thresh": "Threshold for matching tracks between frames (0.0-1.0). Higher = stricter matching",
    "track_thresh": "Confidence threshold for creating new tracks (0.0-1.0)",
    "track_buffer": "Number of frames to keep lost tracks before deletion",
    "proximity_thresh": "Distance threshold for spatial association (0.0-1.0)", 
    "appearance_thresh": "ReID appearance similarity threshold (0.0-1.0)",
    "half": "Use half precision (FP16) for faster inference on GPU",
    "reid_weights": "Path to ReID model weights file",
    "device": "Device to run tracker on (e.g. cuda, cpu)",
    "det_thresh": "Detection confidence threshold for considering detections (0.0-1.0)",
    "max_age": "Maximum number of frames to keep track alive without matching",
    "min_hits": "Minimum number of detections before track is initialized",
    "inertia": "Motion model weight for position prediction (0.0-1.0)",
    "use_ecc": "Whether to use ECC (Enhanced Correlation Coefficient) for motion estimation",
    "cmc_method": "Camera Motion Compensation method (None, ECC, or Sparse)",
    "cmc_off": "Whether to disable Camera Motion Compensation",
    "fuse_first_associate": "Whether to fuse appearance and motion information in first association",
    "with_reid": "Whether to use ReID features for tracking"
  }
} 
```

## 📊 **Output Files**

After running, you'll get:

### **Results Directory**
```
results/fair_mivolo_bench/
├── summary.json              # Complete test results & configuration
├── results.jsonl             # Individual tracker results (one per line)
├── bytetrack-aggressive.mp4  # Output video with person tracking
├── botsort-strict.mp4        # Output video with person tracking
├── botsort-loose.mp4         # Output video with person tracking
├── deepocsort-strict.mp4     # Output video with person tracking
├── boosttrack-default.mp4    # Output video with person tracking
├── boosttrack-optimized.mp4  # Output video with person tracking
└── ...                       # One video per tracker config
```

### **Summary Analysis**
The benchmark automatically provides:
- **Performance ranking** by speed
- **ID consistency** analysis  
- **GPU memory impact** per tracker
- **Test order** to verify fairness
- **Parameter comparison** between configs

## 🔧 **Parameter Tuning Guide**

### **Common Parameters**

| Parameter | Range | Effect |
|-----------|-------|---------|
| `match_thresh` | 0.0-1.0 | Higher = stricter track matching |
| `track_thresh` | 0.0-1.0 | Higher = fewer new tracks created |
| `proximity_thresh` | 0.0-1.0 | Higher = larger distance tolerance |
| `appearance_thresh` | 0.0-1.0 | Higher = more similar appearance required |
| `track_buffer` | 1-100 | Frames to keep lost tracks |

### **Tuning for Different Scenarios**

**Crowded scenes (many people):**
- Lower `match_thresh` (0.6-0.7)
- Higher `proximity_thresh` (0.7-0.8)
- Lower `appearance_thresh` (0.15-0.2)

**Sparse scenes (few people):**
- Higher `match_thresh` (0.8-0.9)
- Lower `proximity_thresh` (0.4-0.5)  
- Higher `appearance_thresh` (0.25-0.3)

**Fast motion:**
- Higher `track_buffer` (30-50)
- Lower `track_thresh` (0.5-0.6)

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"Cannot import tracker"**
   - Check BoxMot installation: `pip list | grep boxmot`
   - Verify tracker names in config match: `ByteTrack`, `BotSort`, etc.

2. **"CUDA out of memory"**
   - Set `"half": false` in tracker params
   - Increase `skip_frames` in test_settings
   - Reduce `max_frames` for testing

3. **"No faces filtered"**
   - Check YOLO model classes: The detector will print class names
   - Verify class 0 = person, class 1 = face in your model

4. **Poor tracking performance**
   - Try different parameter combinations
   - Check if ReID weights are loading correctly
   - Verify device is using GPU (`torch.cuda.is_available()`)

## 📈 **Performance Analysis**

The benchmark provides several metrics for analysis:

### **Speed Metrics**
- `fps_overall`: Video processing speed  
- `fps_detection`: Detection-only speed
- `total_time`: Complete test duration

### **Accuracy Metrics**  
- `unique_ids`: Number of unique person IDs tracked
- ID consistency across frames (visual inspection of videos)

### **Fairness Metrics**
- `test_order`: Sequence number to verify randomization worked
- `gpu_memory_delta`: Memory usage change during test
- `pre_test_gpu_memory`: Starting memory state

## 🎯 **Best Practices**

1. **Run multiple times** - Since order is randomized, run 2-3 times for statistical validity
2. **Monitor resources** - Watch GPU memory and temperature during long tests  
3. **Start small** - Use `test_configs.json` first to verify setup works
4. **Compare similar configs** - Test parameter variations of the same base tracker
5. **Visual validation** - Watch output videos to verify tracking quality matches metrics 
