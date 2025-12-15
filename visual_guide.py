# Example images for each weed class

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

print("Finding example images for each weed class...")
print("=" * 70)
# Set up file paths
WORK_DIR = Path(".")  # Current directory
# Set up paths
TRAIN_IMAGES = WORK_DIR / "train" / "images"
TRAIN_LABELS = WORK_DIR / "train" / "labels"
CLASS_NAMES = ["Carpetweed", "Morning Glory", "Palmer Amaranth"]

# Find images containing each class
class_examples = defaultdict(list)

for label_file in TRAIN_LABELS.glob("*.txt"):
    if label_file.stat().st_size > 0:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    image_file = TRAIN_IMAGES / f"{label_file.stem}.jpg"
                    if image_file.exists():
                        class_examples[class_id].append(image_file)

# Select one clear example per class (first occurrence)
examples_to_show = {}
for class_id in range(len(CLASS_NAMES)):
    if class_examples[class_id]:
        examples_to_show[class_id] = class_examples[class_id][0]
        print(
            f"✓ Found example for {CLASS_NAMES[class_id]}: {examples_to_show[class_id].name}"
        )
    else:
        print(f"!!!  No examples found for {CLASS_NAMES[class_id]}")

# Display the examples
if examples_to_show:
    print("\n" + "=" * 70)
    print("Displaying example images with bounding boxes...")
    print("=" * 70)

    fig, axes = plt.subplots(1, len(examples_to_show), figsize=(15, 5))
    if len(examples_to_show) == 1:
        axes = [axes]

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # BGR colors for OpenCV

    for idx, (class_id, image_path) in enumerate(sorted(examples_to_show.items())):
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Read corresponding label
        label_file = TRAIN_LABELS / f"{image_path.stem}.txt"
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls == class_id:  # Only draw boxes for the target class
                        # Convert YOLO format to pixel coordinates
                        x_center, y_center, box_w, box_h = map(float, parts[1:5])
                        x1 = int((x_center - box_w / 2) * w)
                        y1 = int((y_center - box_h / 2) * h)
                        x2 = int((x_center + box_w / 2) * w)
                        y2 = int((y_center + box_h / 2) * h)

                        # Draw bounding box
                        color = colors[class_id]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                        # Add class label
                        label_text = f"{CLASS_NAMES[class_id]}"
                        cv2.putText(
                            img,
                            label_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2,
                        )

        # Display
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"Class {class_id}: {CLASS_NAMES[class_id]}", fontsize=12, fontweight="bold"
        )
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("visual.png")

    print("\n✅ Example images displayed!")
    print("\n Pro Tip: Keep these visual characteristics in mind when:")
    print("   • Analyzing model predictions in the 3LC Dashboard")
    print("   • Identifying mislabeled or missing annotations")
    print("   • Understanding class confusion patterns")

else:
    print("\n⚠️  Could not find example images for visualization")