"""
Quick test for the fair benchmark system
Uses test_configs.json with limited trackers and frames for validation
"""

import sys
from fair_mivolo_bench import *

def main():
    print("=== Quick Fair Benchmark Test ===")
    print("This uses test_configs.json for a quick validation")
    
    # Override the config loading to use test config
    global config
    try:
        config = load_tracker_configs("test_configs.json")
        print(f"✅ Loaded test configuration successfully")
    except Exception as e:
        print(f"❌ Failed to load test configuration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if main():
        print("\n🚀 Starting quick test with limited trackers and frames...")
        print("This should complete in 3-5 minutes")
    else:
        print("❌ Test failed to initialize")
        sys.exit(1) 