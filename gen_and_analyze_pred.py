# Import required packages
from pathlib import Path
import shutil
from tlc_ultralytics import YOLO, Settings

# Define paths and constants
WORK_DIR = Path(".")
TEST_DIR = WORK_DIR / "test" / "images"
PRED_DIR = Path("predictions")
IMAGE_SIZE = 640  # Competition requirement

# Get list of test images
test_images = list(TEST_DIR.glob("*.jpg"))

model = YOLO("runs/detect/DEFAULT_run/weights/best.pt")

# ============================================================================
# SAFER FILE MANAGEMENT - Backup instead of delete
# ============================================================================
if PRED_DIR.exists():
    from datetime import datetime

    # Create timestamped backup instead of deleting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"predictions_backup_{timestamp}")

    print("⚠️  Predictions folder exists. Creating backup...")
    print(f"   Moving to: {backup_dir}")

    shutil.move(str(PRED_DIR), str(backup_dir))

    print(f"✅ Previous predictions backed up: {backup_dir}")
    print("   (Delete old backups manually if not needed)")
else:
    print("✅ No existing predictions found")

print("Generating predictions on test set...")
print("=" * 50)
print(f"Test images: {TEST_DIR}")
print(f"Output directory: {PRED_DIR}")
print(f"Test set size: {len(test_images)} images")

# Run inference
print("\nRunning inference...")
test_results = model.predict(
    source=str(TEST_DIR),
    save=True,  # Don't save annotated images (faster, prevents duplication)
    save_txt=True,  # Save YOLO format predictions
    save_conf=True,  # Include confidence scores
    conf=0.25,  # Confidence threshold (adjust as needed)
    imgsz=IMAGE_SIZE,
    project=str(PRED_DIR.parent),
    name=PRED_DIR.name,
    exist_ok=False,  # Don't allow overwriting (ensures clean predictions)
)

print("\n----Predictions generated!")


# Define constants
CLASS_NAMES = ["Carpetweed", "Morning Glory", "Palmer Amaranth"]

# Analyze predictions
PRED_DIR = Path("predictions")  # Must match Cell 21
labels_dir = PRED_DIR / "labels"

if labels_dir.exists():
    print("Test Set Prediction Analysis:")
    print("=" * 50)

    pred_files = list(labels_dir.glob("*.txt"))

    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    images_with_preds = 0
    total_detections = 0

    for pred_file in pred_files:
        if pred_file.stat().st_size > 0:
            images_with_preds += 1
            with open(pred_file, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_detections += 1

    print(f"Total test images: {len(test_images)}")
    print(f"Images with detections: {images_with_preds}")
    print(f"Images with no detections: {len(test_images) - images_with_preds}")
    print(f"Total detections: {total_detections}")

    print("\n Detections by class:")
    for class_id, count in class_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"   {CLASS_NAMES[class_id]:20s}: {count:4d} ({percentage:5.1f}%)")

    print("\n----Analysis complete!")
else:
    print("!!!!No predictions found.")