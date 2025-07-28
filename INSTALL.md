# Multi-Tracker Benchmark Installation Guide


This guide will help you set up the multi-tracker benchmark system on a new computer.

## 📋 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3050 Ti)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space for weights and results

### Software Requirements
- **OS**: Windows 10/11 (tested), Linux (should work)
- **Python**: 3.8-3.10 (tested on 3.10.18)
- **CUDA**: 11.8+ (for GPU acceleration)

## 🚀 Installation Steps

### Step 1: Create Conda Environment

```bash
# Create new conda environment
conda create -n mot310 python=3.10 -y
conda activate mot310
```

### Step 2: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 11.8 support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 3: Install Core Dependencies

```bash
# Install ultralytics (YOLO)
pip install ultralytics

# Install BoxMOT (tracking library)
pip install boxmot

# Install other dependencies
pip install opencv-python numpy pathlib2
```

### Step 4: Fix NumPy Compatibility (if needed)

```bash
# If you encounter NumPy compatibility issues, downgrade:
pip install numpy==1.24.3
```

### Step 5: Create Project Structure

```bash
# Create project directory
mkdir model_bench
cd model_bench

# Create subdirectories
mkdir weights weights/reid scripts results dataset
```

### Step 6: Download Model Weights

#### YOLOv8 Person+Face Detection Model
```bash
# Download YOLOv8X person+face detection weights
# Place in: weights/yolov8x_person_face.pt
```
**Note**: You need to provide the `yolov8x_person_face.pt` file (trained for person+face detection)

#### ReID (Person Re-identification) Weights
Download these ReID model weights and place in `weights/reid/`:

```bash
# Download ReID weights (choose one or more):
# 1. OSNet models (recommended)
weights/reid/osnet_x1_0_msmt17.pt
weights/reid/osnet_x1_0_imagenet.pt  
weights/reid/osnet_x0_5_imagenet.pt

# 2. Alternative ReID models
weights/reid/resnet50_msmt_xent.pt
weights/reid/mobilenetv2_1dot0_msmt.pt
weights/reid/mlfn_msmt_xent.pt
```

**ReID Weight Sources:**
- OSNet: [GitHub - KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- ResNet50: [GitHub - Xiangyu-CAS/MLFN](https://github.com/Xiangyu-CAS/MLFN)

### Step 7: Add Your Video

```bash
# Place your test video in project root
# Example: 2min.mp4
```

### Step 8: Copy Script Files

Copy these files to the `scripts/` directory:

#### Required Files:
- `scripts/fair_mivolo_bench.py` - Main benchmark script
- `scripts/tracker_configs.json` - Tracker configuration file
- `scripts/test_configs.json` - Quick test configuration (optional)

### Step 9: Test Installation

```bash
# Activate environment
conda activate mot310

# Navigate to scripts directory
cd scripts

# Test basic functionality
python -c "
import torch
import cv2
from ultralytics import YOLO
from boxmot import BotSort, DeepOcSort, ByteTrack, BoostTrack
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"
```

## 📁 Final Directory Structure

```
model_bench/
├── INSTALL.md
├── 2min.mp4                          # Your test video
├── weights/
│   ├── yolov8x_person_face.pt       # YOLOv8 person+face detection
│   └── reid/
│       ├── osnet_x1_0_msmt17.pt     # ReID model (primary)
│       ├── osnet_x1_0_imagenet.pt   # ReID model (alternative)
│       └── osnet_x0_5_imagenet.pt   # ReID model (alternative)
├── scripts/
│   ├── fair_mivolo_bench.py         # Main benchmark script
│   ├── tracker_configs.json         # Tracker configurations
│   └── test_configs.json            # Quick test configs (optional)
├── results/                         # Output directory (auto-created)
└── dataset/                         # Dataset directory (optional)
```

## 🏃 Running the Benchmark

### Quick Test (recommended first)
```bash
cd scripts
python fair_mivolo_bench.py --config test_configs.json
```

### Full Benchmark
```bash
cd scripts
python fair_mivolo_bench.py --config tracker_configs.json
```

### Custom Configuration
```bash
cd scripts
python fair_mivolo_bench.py --config your_config.json --output-dir ../results/custom_test
```

## 📊 Expected Output

The benchmark will create:
- **Videos**: `results/fair_mivolo_bench/[tracker-name].mp4`
- **Detailed Results**: `results/fair_mivolo_bench/summary.json`
- **Individual Results**: `results/fair_mivolo_bench/results.jsonl`

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
```

### Import Errors
```bash
# Fix BoxMOT import issues
pip install boxmot --upgrade

# Fix NumPy compatibility
pip install numpy==1.24.3
```

### Video Loading Issues
```bash
# Test video file
python -c "
import cv2
cap = cv2.VideoCapture('2min.mp4')
print(f'Video opened: {cap.isOpened()}')
print(f'Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
cap.release()
"
```

### Memory Issues
```bash
# Reduce batch size in config
# Set "max_frames": 300 in tracker_configs.json for testing
```

## 🎯 Performance Optimization

### For Better Speed:
- Use `skip_frames: 5` (process every 5th frame)
- Set `conf_threshold: 0.6` (higher confidence)
- Set `max_frames: 300` (test on shorter clips)

### For Better Accuracy:
- Use `skip_frames: 1` (process every frame)
- Set `conf_threshold: 0.25` (lower confidence)
- Set `filter_faces: true` (only track persons)

## 📋 Configuration Files

### tracker_configs.json
Contains 8 different tracker configurations:
- ByteTrack (Default, Aggressive)
- BotSort (Default, Strict, Loose)
- DeepOcSort (Default, Strict)
- BoostTrack (Default)

### test_configs.json (Quick Test)
Smaller configuration for initial testing:
- Only 2-3 trackers
- Limited frames
- Faster processing

## 🆘 Support

If you encounter issues:
1. Check Python and PyTorch versions
2. Verify CUDA installation
3. Test with `test_configs.json` first
4. Check video file accessibility
5. Verify weight files are downloaded correctly

## 📝 Notes

- **GPU Recommended**: CPU-only mode will be much slower
- **Memory Usage**: ReID-based trackers use more GPU memory
- **Fair Testing**: Script randomizes test order and clears GPU memory between tests
- **Face Filtering**: Only persons are tracked, faces are filtered out
- **Results**: All configurations and results are saved for analysis

## 🔄 Environment Export/Import

### Export Current Environment
```bash
# Export environment for sharing
conda env export > environment.yml
```

### Import Environment on New Computer
```bash
# Import environment
conda env create -f environment.yml
```

This setup should work on any computer with NVIDIA GPU and CUDA support! 
