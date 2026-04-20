# main.py - Military Drone Detection Training with YOLO11

from ultralytics import YOLO
import os
from pathlib import Path
import shutil
import random

def prepare_dataset(source_dir="drone-detection"):
    """
    Prepare Kaggle dataset for YOLO training
    Assumes dataset structure:
    drone-detection/
        ├── images/
        └── labels/
    """
    print("=" * 60)
    print("📊 PREPARING KAGGLE DATASET")
    print("=" * 60)
    
    source_path = Path(source_dir)
    
    # Check if dataset exists
    if not source_path.exists():
        print(f"\n❌ Dataset not found at: {source_path}")
        print("\nPlease download from Kaggle:")
        print("https://www.kaggle.com/datasets/banderastepan/drone-detection")
        print("\nThen extract to: drone-detection/")
        return None
    
    print(f"\n✅ Found dataset at: {source_path}")
    
    # Find images and labels directories
    images_dir = None
    labels_dir = None
    
    # Search for images and labels folders
    for item in source_path.rglob("*"):
        if item.is_dir():
            if "image" in item.name.lower():
                images_dir = item
            elif "label" in item.name.lower():
                labels_dir = item
    
    # If not found, assume they're directly in root
    if images_dir is None:
        possible_images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        if possible_images:
            images_dir = source_path
            print(f"  📁 Images found in root directory")
    
    if labels_dir is None:
        possible_labels = list(source_path.glob("*.txt"))
        if possible_labels:
            labels_dir = source_path
            print(f"  📁 Labels found in root directory")
    
    if not images_dir or not labels_dir:
        print("\n❌ Could not find images or labels directories!")
        print("Please check dataset structure.")
        return None
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"\n📊 Found {len(image_files)} images")
    
    # Verify corresponding labels exist
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
    
    print(f"✅ Verified {len(valid_pairs)} image-label pairs")
    
    # Split dataset: 80% train, 15% val, 5% test
    random.shuffle(valid_pairs)
    
    train_size = int(0.80 * len(valid_pairs))
    val_size = int(0.15 * len(valid_pairs))
    
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:train_size + val_size]
    test_pairs = valid_pairs[train_size + val_size:]
    
    print(f"\n📈 Dataset split:")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"  Valid: {len(val_pairs)} ({len(val_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/len(valid_pairs)*100:.1f}%)")
    
    # Create output structure
    output_dir = Path("military-drones-dataset")
    
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate splits
    print("\n📦 Copying files...")
    
    def copy_split(pairs, split_name):
        for img_file, label_file in pairs:
            # Copy image
            shutil.copy2(img_file, output_dir / split_name / 'images' / img_file.name)
            # Copy label
            shutil.copy2(label_file, output_dir / split_name / 'labels' / label_file.name)
    
    copy_split(train_pairs, 'train')
    copy_split(val_pairs, 'valid')
    copy_split(test_pairs, 'test')
    
    print("✅ Files copied successfully!")
    
    # Create data.yaml
    yaml_content = f"""# Military Drones Dataset Configuration
# Source: Kaggle - banderastepan/drone-detection
# Synthetic dataset with military UAVs

path: {output_dir.absolute()}
train: train/images
val: valid/images
test: test/images

# Classes
nc: 1
names: ['drone']

# Drone types in dataset:
# - Shahed-131, Shahed-136 (Iranian loitering munitions)
# - Lancet (Russian loitering munition)
# - Orlan-10 (Russian reconnaissance UAV)
# - ZALA (Russian reconnaissance UAV)
# - Forpost (Russian MALE UAV)
# - Mohajer (Iranian reconnaissance UAV)
# - Granat series (Russian tactical UAV)
# - Supercam (Russian reconnaissance UAV)
# - Techyon (Commercial/tactical UAV)
# - Mavic 3 (Commercial DJI drone)
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created config: {yaml_path}")
    
    return str(yaml_path)

def train_military_drone_detector(data_yaml):
    """Train YOLO11 on military drones dataset"""
    
    print("\n" + "=" * 60)
    print("🚀 TRAINING YOLO11 - MILITARY DRONE DETECTION")
    print("=" * 60)
    
    # Load YOLO11 nano model
    model = YOLO('yolo11n.pt')
    
    print("\n📊 Model: YOLO11n")
    print("🎯 Task: Military UAV Detection")
    print("⚡ Device: GPU (CUDA) if available")
    
    # Train with optimized parameters for military drones
    results = model.train(
        data=data_yaml,
        epochs=100,              # More epochs for better accuracy
        imgsz=640,              # Matches dataset size
        batch=16,               # Adjust based on VRAM
        device=0,               # GPU
        patience=20,            # Early stopping
        save=True,
        project='results',
        name='drone_model',
        
        # Augmentation optimized for aerial detection
        hsv_h=0.015,           # Minimal hue shift
        hsv_s=0.7,             # Strong saturation for various lighting
        hsv_v=0.5,             # Brightness variance (day/night)
        degrees=10,            # Slight rotation (drones can bank)
        translate=0.2,         # Position variance
        scale=0.7,             # Scale variance (distance)
        flipud=0.0,            # No vertical flip (maintains orientation)
        fliplr=0.5,            # Horizontal flip OK
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.1,             # Mixup for robustness
        
        # Training optimization
        optimizer='auto',       # Automatic optimizer selection
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate
        momentum=0.937,        # SGD momentum
        weight_decay=0.0005,   # L2 regularization
        warmup_epochs=3,       # Warmup epochs
        
        # Advanced options
        amp=True,              # Automatic Mixed Precision
        fraction=1.0,          # Use full dataset
        plots=True,            # Generate plots
        verbose=True,
    )
    
    print("\n✅ Training completed!")
    print(f"📁 Best model: results/drone_model/weights/best.pt")
    
    return results

def validate_model():
    """Validate trained model"""
    
    model_path = "results/drone_model/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Train first.")
        return
    
    print("\n" + "=" * 60)
    print("🔍 VALIDATING MODEL")
    print("=" * 60)
    
    model = YOLO(model_path)
    metrics = model.val()
    
    print("\n📊 Performance Metrics:")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")
    
    return metrics

def export_model():
    """Export model for deployment"""
    
    model_path = "results/drone_model/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return
    
    print("\n" + "=" * 60)
    print("📦 EXPORTING MODEL")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # Export to ONNX for deployment
    model.export(format='onnx')
    print("✅ Exported to ONNX format")
    
    # Optionally export to other formats
    # model.export(format='torchscript')
    # model.export(format='engine')  # TensorRT (NVIDIA only)

def main():
    """Full training pipeline"""
    
    print("=" * 60)
    print("🚁 MILITARY DRONE DETECTION - YOLO11")
    print("=" * 60)
    print("\nDataset: Kaggle - banderastepan/drone-detection")
    print("Target: Shahed, Lancet, Orlan, and other military UAVs")
    print("=" * 60)
    
    # Step 1: Prepare dataset
    data_yaml = prepare_dataset("dataset")
    
    if data_yaml is None:
        print("\n❌ Dataset preparation failed. Exiting.")
        return
    
    # Step 2: Train model
    train_military_drone_detector(data_yaml)
    
    # Step 3: Validate
    validate_model()
    
    # Step 4: Export (optional)
    export_choice = input("\n📦 Export model to ONNX? (y/n): ")
    if export_choice.lower() == 'y':
        export_model()
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n🎯 Next steps:")
    print("  1. Test model: streamlit run app.py")
    print("  2. Check results: results/drone_model/")
    print("  3. View metrics: results/drone_model/results.png")
    print("=" * 60)

if __name__ == "__main__":
    main()