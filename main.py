# main.py - YOLO training pipeline for military object detection
# Classes: Artillery, Missile, Radar, RocketLauncher, Soldier, Tank, Vehicle
#
# Full pipeline:
#   1. Prepare dataset   (calls dataset.py)
#   2. Train YOLO model
#   3. Validate
#   4. Export to ONNX (optional)
#
# Quick start:
#   python main.py                              # uses defaults
#   python main.py --data dataset/data.yaml     # skip dataset prep, use existing
#   python main.py --skip-prep --data dataset/data.yaml
#   python main.py --model yolo11s.pt --epochs 150 --batch 32

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_SOURCE_DIR  = "raw_dataset"     # raw images + labels (input to dataset.py)
DEFAULT_DATASET_DIR = "dataset"         # must match "path:" in data.yaml
DEFAULT_OUTPUT_DIR  = "/home/user/final_project/runs/detect/results"         # training run output
DEFAULT_RUN_NAME    = "detector"        # subfolder inside results/

# Base model to fine-tune. Options (smallest → largest):
#   yolo11n.pt  yolo11s.pt  yolo11m.pt  yolo11l.pt  yolo11x.pt
DEFAULT_MODEL       = "yolo11s.pt"  # s-size recommended for 7-class multi-object task

DEFAULT_EPOCHS      = 100
DEFAULT_IMGSZ       = 640
DEFAULT_BATCH       = 16
DEFAULT_DEVICE      = 0       # 0 = first GPU; "cpu" for CPU-only machines
# ─────────────────────────────────────────────────────────────────────────────


def train(
    data_yaml:   str,
    base_model:  str  = DEFAULT_MODEL,
    epochs:      int  = DEFAULT_EPOCHS,
    imgsz:       int  = DEFAULT_IMGSZ,
    batch:       int  = DEFAULT_BATCH,
    device              = DEFAULT_DEVICE,
    output_dir:  str  = DEFAULT_OUTPUT_DIR,
    run_name:    str  = DEFAULT_RUN_NAME,
) -> YOLO:
    print("\n" + "=" * 60)
    print("🚀  TRAINING")
    print("=" * 60)
    print(f"  Base model : {base_model}")
    print(f"  Data       : {data_yaml}")
    print(f"  Epochs     : {epochs}")
    print(f"  Image size : {imgsz}")
    print(f"  Batch      : {batch}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_dir}/{run_name}")

    model = YOLO(base_model)

    model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = output_dir,
        name    = run_name,

        # Early stopping — quits if val metric doesn't improve for N epochs
        patience = 20,

        # Augmentation — tuned for small object detection (vehicles, people)
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        degrees  = 5,
        translate= 0.1,
        scale    = 0.5,
        flipud   = 0.0,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.05,

        # Optimiser
        optimizer    = "auto",
        lr0          = 0.01,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,

        amp     = True,    # Automatic Mixed Precision — speeds up GPU training
        plots   = True,
        verbose = True,
        save    = True,
    )

    best_pt = Path(output_dir) / run_name / "weights" / "best.pt"
    print(f"\n✅  Training complete.")
    print(f"    Best weights: {best_pt}")
    return YOLO(str(best_pt))


def validate(model: YOLO) -> None:
    print("\n" + "=" * 60)
    print("🔍  VALIDATION")
    print("=" * 60)

    metrics = model.val()

    print("\n📊  Results:")
    print(f"    mAP50      : {metrics.box.map50:.4f}")
    print(f"    mAP50-95   : {metrics.box.map:.4f}")
    print(f"    Precision  : {metrics.box.mp:.4f}")
    print(f"    Recall     : {metrics.box.mr:.4f}")


def export_onnx(model: YOLO, imgsz: int = DEFAULT_IMGSZ) -> None:
    print("\n" + "=" * 60)
    print("📦  ONNX EXPORT")
    print("=" * 60)

    model.export(format="onnx", imgsz=imgsz, half=False, dynamic=False)
    print("✅  Exported to ONNX.")
    print("    Copy the .onnx file to your deployment directory and select")
    print("    'ONNX' as the backend in app.py.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO object detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    g = parser.add_argument_group("Dataset")
    g.add_argument("--source",     default=DEFAULT_SOURCE_DIR,
                   help="Raw dataset folder (images + YOLO labels)")
    g.add_argument("--output-dataset", default=DEFAULT_DATASET_DIR,
                   help="Where to write the prepared dataset")
    g.add_argument("--classes", nargs="+", default=None,
                   help="Class names in index order. Defaults to: "
                        "Artillery Missile Radar RocketLauncher Soldier Tank Vehicle")
    g.add_argument("--skip-prep", action="store_true",
                   help="Skip dataset preparation — use --data instead")
    g.add_argument("--data", default=None,
                   help="Path to an existing data.yaml (implies --skip-prep)")

    # Training
    g = parser.add_argument_group("Training")
    g.add_argument("--model",   default=DEFAULT_MODEL,   help="YOLO base model weights (n/s/m/l/x)")
    g.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    g.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    g.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    g.add_argument("--device",  default=str(DEFAULT_DEVICE),
                   help="Device: 0 (first GPU), 0,1 (multi-GPU), or cpu")
    g.add_argument("--project", default=DEFAULT_OUTPUT_DIR, help="Results parent folder")
    g.add_argument("--name",    default=DEFAULT_RUN_NAME,   help="Run subfolder name")

    # Post-training
    g = parser.add_argument_group("Post-training")
    g.add_argument("--no-validate", action="store_true", help="Skip validation after training")
    g.add_argument("--export-onnx", action="store_true",
                   help="Export best.pt to ONNX after training")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯  OBJECT DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Dataset preparation ───────────────────────────────────────────
    data_yaml = args.data

    if data_yaml and os.path.exists(data_yaml):
        print(f"\n✅  Using existing data.yaml: {data_yaml}")
    else:
        if args.skip_prep:
            print("\n❌  --skip-prep set but no valid --data path provided.")
            return

        # Import here so dataset.py can also be used standalone
        from dataset import prepare_dataset, CLASS_NAMES

        class_names = args.classes if args.classes else CLASS_NAMES

        data_yaml = prepare_dataset(
            source_dir  = args.source,
            output_dir  = args.output_dataset,
            class_names = class_names,
        )

        if data_yaml is None:
            print("\n❌  Dataset preparation failed. Exiting.")
            return

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    # Resolve device: keep as int if numeric, else string ("cpu")
    device = int(args.device) if args.device.isdigit() else args.device

    trained_model = train(
        data_yaml  = data_yaml,
        base_model = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        device     = device,
        output_dir = args.project,
        run_name   = args.name,
    )

    # ── Step 3: Validate ──────────────────────────────────────────────────────
    if not args.no_validate:
        validate(trained_model)

    # ── Step 4: Export ────────────────────────────────────────────────────────
    if args.export_onnx:
        export_onnx(trained_model, imgsz=args.imgsz)
    else:
        ans = input("\n📦  Export to ONNX now? (y/n): ").strip().lower()
        if ans == "y":
            export_onnx(trained_model, imgsz=args.imgsz)

    # ── Done ──────────────────────────────────────────────────────────────────
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("✅  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Trained model : {best_pt}")
    print(f"  Copy best.pt (and best.onnx if exported) to your app directory. The model detects 7 classes: Artillery, Missile, Radar, RocketLauncher, Soldier, Tank, Vehicle.")
    print(f"  then run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
    print("=" * 60)
    print(f"  Copy best.pt (and best.onnx if exported) to your app directory. The model detects 7 classes: Artillery, Missile, Radar, RocketLauncher, Soldier, Tank, Vehicle.")
    print(f"  then run:  streamlit run app.py")
    print("=" * 60)