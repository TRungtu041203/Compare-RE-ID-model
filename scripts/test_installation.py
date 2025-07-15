#!/usr/bin/env python3
"""
Installation Test Script
Tests if all dependencies are installed correctly and the system is ready for benchmarking.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    try:
        from boxmot import BotSort, DeepOcSort, ByteTrack, BoostTrack
        print(f"✅ BoxMOT trackers imported successfully")
    except ImportError as e:
        print(f"❌ BoxMOT import failed: {e}")
        return False
    
    return True

def test_files():
    """Test if required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        '../weights/yolov8x_person_face.pt',
        '../weights/reid/osnet_x1_0_msmt17.pt',
        '../2min.mp4',
        'tracker_configs.json',
        'fair_mivolo_bench.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_video_loading():
    """Test if video can be loaded"""
    print("\n🔍 Testing video loading...")
    
    try:
        import cv2
        cap = cv2.VideoCapture('../2min.mp4')
        
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Video loaded successfully")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps:.1f}")
            print(f"   Frame count: {frame_count}")
            print(f"   Duration: {frame_count/fps:.1f} seconds")
            
            cap.release()
            return True
        else:
            print(f"❌ Could not open video file")
            return False
            
    except Exception as e:
        print(f"❌ Video loading failed: {e}")
        return False

def test_tracker_initialization():
    """Test if trackers can be initialized"""
    print("\n🔍 Testing tracker initialization...")
    
    try:
        from boxmot import BotSort, DeepOcSort, ByteTrack, BoostTrack
        import torch
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        reid_weights = '../weights/reid/osnet_x1_0_msmt17.pt'
        
        # Test ByteTrack (no ReID)
        try:
            tracker = ByteTrack()
            print("✅ ByteTrack initialized successfully")
        except Exception as e:
            print(f"❌ ByteTrack failed: {e}")
            return False
        
        # Test BotSort (with ReID)
        try:
            tracker = BotSort(reid_weights=reid_weights, device=device, half=False)
            print("✅ BotSort initialized successfully")
        except Exception as e:
            print(f"❌ BotSort failed: {e}")
            return False
        
        # Test DeepOcSort (with ReID)
        try:
            tracker = DeepOcSort(reid_weights=reid_weights, device=device, half=False)
            print("✅ DeepOcSort initialized successfully")
        except Exception as e:
            print(f"❌ DeepOcSort failed: {e}")
            return False
        
        # Test BoostTrack (with ReID)
        try:
            tracker = BoostTrack(reid_weights=reid_weights, device=device, half=False)
            print("✅ BoostTrack initialized successfully")
        except Exception as e:
            print(f"❌ BoostTrack failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tracker initialization failed: {e}")
        return False

def test_yolo_model():
    """Test if YOLO model can be loaded"""
    print("\n🔍 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        model = YOLO('../weights/yolov8x_person_face.pt')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"✅ YOLO model loaded successfully")
        print(f"   Model classes: {model.model.names}")
        print(f"   Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Multi-Tracker Benchmark Installation Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_files),
        ("Video Loading Test", test_video_loading),
        ("YOLO Model Test", test_yolo_model),
        ("Tracker Initialization Test", test_tracker_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for benchmarking.")
        print("\nYou can now run the benchmark with:")
        print("   python fair_mivolo_bench.py --config test_configs.json")
        return True
    else:
        print("❌ Some tests failed. Please check the installation guide.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 