# ============================================================================
# STEP 8: Generate Kaggle Submission By running this cell of code
# ============================================================================
# Import required packages
from pathlib import Path
import pandas as pd
from IPython.display import display


# Define paths
WORK_DIR = Path(".")  # Current directory
PRED_DIR = Path(
    "predictions"
)  # Prediction directory (change path if you want to convert from a different predictions folder)
TEST_DIR = (
    WORK_DIR / "test" / "images"
)  # Change path if you have the Test images stored Elsewhere


print("=" * 70)
print("GENERATING KAGGLE SUBMISSION")
print("=" * 70)

labels_dir = PRED_DIR / "labels"
output_csv = "result.csv"

# Get all test images (deduplicate by stem to avoid duplicates from case-insensitive file systems)
test_images_dict = {}  # Use dict to automatically deduplicate by image_id (stem)
for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]:
    for img_path in TEST_DIR.glob(ext):
        image_id = img_path.stem  # filename without extension
        if image_id not in test_images_dict:
            test_images_dict[image_id] = img_path

# Convert to sorted list
test_images_list = [
    test_images_dict[img_id] for img_id in sorted(test_images_dict.keys())
]

print(f"\n✓ Found {len(test_images_list)} test images")
print(f"✓ Looking for predictions in: {labels_dir}")

# Create submission data
submission_data = []
images_with_preds = 0
images_without_preds = 0
total_boxes = 0

for img_path in test_images_list:
    image_id = img_path.stem
    pred_file = labels_dir / f"{image_id}.txt"

    # Check if prediction file exists and has content
    if pred_file.exists() and pred_file.stat().st_size > 0:
        prediction_boxes = []

        with open(pred_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()

                    # YOLO saves as: class xc yc w h conf (confidence is LAST!)
                    # Kaggle needs: class conf xc yc w h (confidence is SECOND!)
                    if len(parts) >= 6:
                        # Reorder values: move confidence from position 5 to position 1
                        class_id = parts[0]
                        conf = parts[5]  # Confidence is at the end in YOLO format
                        xc, yc, w, h = parts[1], parts[2], parts[3], parts[4]
                        box_str = f"{class_id} {conf} {xc} {yc} {w} {h}"
                        prediction_boxes.append(box_str)
                        total_boxes += 1

        if prediction_boxes:
            # Join all boxes with spaces
            prediction_string = " ".join(prediction_boxes)
            images_with_preds += 1
        else:
            prediction_string = "no box"
            images_without_preds += 1
    else:
        # No prediction file or empty file
        prediction_string = "no box"
        images_without_preds += 1

    submission_data.append(
        {"image_id": image_id, "prediction_string": prediction_string}
    )

# Create DataFrame with correct column names (lowercase!)
submission_df = pd.DataFrame(submission_data)
submission_df = submission_df[["image_id", "prediction_string"]]

# Save to CSV
submission_df.to_csv(output_csv, index=False)

# Print statistics
print("\n" + "=" * 70)
print("SUBMISSION STATISTICS")
print("=" * 70)
print(f"Total images:               {len(submission_df)}")
print(f"Images with predictions:    {images_with_preds}")
print(f"Images without predictions: {images_without_preds}")
print(f"Total bounding boxes:       {total_boxes}")
if len(submission_df) > 0:
    print(f"Average boxes per image:    {total_boxes / len(submission_df):.2f}")

# Show sample
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)
display(submission_df.head(10))

# Validation
print("\n" + "=" * 70)
print("FORMAT VALIDATION")
print("=" * 70)

# Check format
errors = []
if list(submission_df.columns) != ["image_id", "prediction_string"]:
    errors.append(f"!!! Wrong columns: {list(submission_df.columns)}")
else:
    print("✓ Columns correct: image_id, prediction_string")

if len(submission_df) != len(test_images_list):
    errors.append("!!! Row count mismatch")
else:
    print(f"✓ Row count correct: {len(submission_df)}")

# Validate prediction format (sample first 20)
format_ok = True
for idx in range(min(20, len(submission_df))):
    pred_str = str(submission_df.iloc[idx]["prediction_string"])

    if pred_str == "no box":
        continue

    values = pred_str.split()
    if len(values) % 6 != 0:
        format_ok = False
        break

if format_ok:
    print("✓ All sampled predictions properly formatted (6 values per box)")
else:
    errors.append("!!! Some predictions have wrong format")

if errors:
    print("\n!!! VALIDATION FAILED:")
    for err in errors:
        print(f"  {err}")
else:
    print("\n" + "=" * 70)
print("✅ SUBMISSION READY FOR KAGGLE!")
print("=" * 70)
print(f"\nFile: {output_csv}")
print("\n Upload 'submission.csv' to Kaggle!")
print("\n Tips:")
print("   - Check your score on the public leaderboard")
print("   - You have 3 submissions per day (use them wisely!)")
print("   - Select up to 2 final submissions for judging")