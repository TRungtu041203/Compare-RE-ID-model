# Fair MIVOLO Tracking Benchmark

This benchmark system addresses the key issues you raised:

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
    "skip_frames": 3,           // Process every Nth frame (1 = every frame)
    "conf_threshold": 0.4,      // YOLO detection confidence
    "iou_threshold": 0.5,       // YOLO NMS IOU threshold  
    "max_frames": null,         // Limit frames (null = full video)
    "filter_faces": true,       // Only track persons, not faces
    "randomize_order": true     // Random test order for fairness
  }
}
```

### **tracker_configs**
```json
{
  "tracker_configs": {
    "bytetrack_default": {
      "class": "ByteTrack",
      "name": "ByteTrack-Default",
      "params": {},
      "description": "ByteTrack with default parameters"
    },
    "botsort_strict": {
      "class": "BotSort", 
      "name": "BotSort-Strict",
      "params": {
        "reid_weights": "auto",     // Will use REID_WEIGHTS path
        "device": "auto",           // Will use detected device
        "half": false,
        "match_thresh": 0.8,        // Stricter matching
        "proximity_thresh": 0.5,    // Distance threshold
        "appearance_thresh": 0.25   // ReID similarity threshold
      },
      "description": "BotSort with strict matching thresholds"
    }
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
├── bytetrack-default.mp4     # Output video with person tracking
├── botsort-strict.mp4        # Output video with person tracking
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
