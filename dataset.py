# inspect_dataset.py - Kaggle Dataset Inspector

import os
from pathlib import Path
import json

def inspect_dataset(dataset_path):
    """Inspect downloaded Kaggle dataset structure"""
    
    print("=" * 60)
    print("📁 DATASET STRUCTURE INSPECTOR")
    print("=" * 60)
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\nDownload it from:")
        print("https://www.kaggle.com/datasets/banderastepan/drone-detection")
        return
    
    # Scan directory structure
    print(f"\n📂 Root: {dataset_path}")
    print("\n📊 Directory Structure:")
    
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}📁 {os.path.basename(root)}/')
        
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}📄 {file}')
        
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')
    
    # Count files by extension
    print("\n📈 File Statistics:")
    
    extensions = {}
    total_files = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
                total_files += 1
    
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count} files")
    
    print(f"\n  Total: {total_files} files")
    
    # Check for common formats
    print("\n🔍 Format Detection:")
    
    has_images = any(ext in extensions for ext in ['.jpg', '.jpeg', '.png'])
    has_labels = any(ext in extensions for ext in ['.txt', '.xml', '.json'])
    has_yaml = '.yaml' in extensions or '.yml' in extensions
    
    if has_images:
        print("  ✅ Images found")
    if has_labels:
        print("  ✅ Labels found")
    if has_yaml:
        print("  ✅ YAML config found")
    
    # Try to detect format
    print("\n🎯 Likely Format:")
    
    if '.txt' in extensions and has_images:
        print("  📋 YOLO format (images + .txt labels)")
    elif '.xml' in extensions:
        print("  📋 Pascal VOC format (.xml)")
    elif '.json' in extensions:
        print("  📋 COCO format (.json)")
    else:
        print("  ⚠️  Unknown format - manual inspection needed")
    
    # Sample a label file
    print("\n📄 Sample Label File:")
    
    for root, dirs, files in os.walk(dataset_path):
        txt_files = [f for f in files if f.endswith('.txt') and f != 'classes.txt']
        if txt_files:
            sample_file = Path(root) / txt_files[0]
            print(f"\n  File: {sample_file.name}")
            with open(sample_file, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f"  Lines: {len(lines)}")
                print(f"  Content preview:")
                for line in lines[:3]:
                    print(f"    {line}")
            break
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Update this path to your downloaded dataset
    dataset_path = "dataset"  # or full path
    
    inspect_dataset(dataset_path)