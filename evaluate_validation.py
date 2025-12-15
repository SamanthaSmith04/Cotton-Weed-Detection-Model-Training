#!/usr/bin/env python3
"""
Evaluate YOLO predictions from CSV against GT CSV using proper mAP@0.5 (VOC-style)
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

# ============================
# CONFIG
# ============================
WORK_DIR = Path(".")
GT_CSV = WORK_DIR / "test/solutions.csv"
PRED_CSV = WORK_DIR / "result.csv"
IOU_THRESHOLD = 0.5
CLASS_NAMES = ["Carpetweed", "Morning Glory", "Palmer Amaranth"]

# ============================
# UTILITIES
# ============================

def load_gt(csv_path):
    """
    Load GT boxes from CSV (image_id, width, height, prediction_string, usage)
    Returns: dict {image_id: [[cls,x,y,w,h], ...]} and dict of image sizes
    """
    gt_dict = {}
    image_sizes = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            image_id = row[0]
            width = float(row[1])
            height = float(row[2])
            image_sizes[image_id] = (width, height)

            pred_str = row[3].strip()
            boxes = []
            if pred_str:
                parts = pred_str.split()
                for i in range(0, len(parts), 5):  # cls x y w h
                    if i + 4 >= len(parts):
                        break
                    cls = int(float(parts[i]))
                    x = float(parts[i+1])
                    y = float(parts[i+2])
                    w = float(parts[i+3])
                    h = float(parts[i+4])
                    boxes.append([cls, x, y, w, h])
            gt_dict[image_id] = boxes
    return gt_dict, image_sizes

def load_pred(csv_path):
    """
    Load predictions CSV (image_id, prediction_string)
    Returns: dict {image_id: [[cls, conf, x, y, w, h], ...]}
    """
    pred_dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            image_id = row[0]
            pred_str = row[1].strip()
            boxes = []
            if pred_str:
                parts = pred_str.split()
                for i in range(0, len(parts), 6):  # cls conf x y w h
                    if i + 5 >= len(parts):
                        break
                    cls = int(float(parts[i]))
                    conf = float(parts[i+1])
                    x = float(parts[i+2])
                    y = float(parts[i+3])
                    w = float(parts[i+4])
                    h = float(parts[i+5])
                    boxes.append([cls, conf, x, y, w, h])
            pred_dict[image_id] = boxes
    return pred_dict

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2
    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1*h1 + w2*h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ============================
# COMPUTE VOC AP
# ============================

def compute_voc_ap(rec, prec):
    """
    Compute VOC 2007 11-point AP
    """
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.0
    return ap

def evaluate_class(gt_dict, pred_dict, class_idx, iou_thresh=0.5):
    """
    Compute AP for a single class
    """
    # Prepare lists
    gt_boxes_per_image = {}
    npos = 0  # total number of GT boxes of this class
    for img_id, boxes in gt_dict.items():
        gt_boxes_per_image[img_id] = [b for b in boxes if b[0]==class_idx]
        npos += len(gt_boxes_per_image[img_id])

    # Collect all predictions for this class
    pred_list = []
    for img_id, boxes in pred_dict.items():
        for b in boxes:
            if b[0]==class_idx:
                pred_list.append([img_id, b[1], b[2], b[3], b[4], b[5]])  # image_id, conf, x, y, w, h

    # Sort by confidence descending
    pred_list.sort(key=lambda x: -x[1])

    # Match predictions
    tp = np.zeros(len(pred_list))
    fp = np.zeros(len(pred_list))
    gt_matched = {img_id: np.zeros(len(boxes)) for img_id, boxes in gt_boxes_per_image.items()}

    for i, pred in enumerate(pred_list):
        img_id, conf, x, y, w, h = pred
        bb = [x, y, w, h]
        gts = gt_boxes_per_image.get(img_id, [])
        ious = np.array([iou(bb, gt[1:]) for gt in gts]) if gts else np.array([])
        if len(ious) > 0:
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            if max_iou >= iou_thresh and gt_matched[img_id][max_iou_idx]==0:
                tp[i] = 1
                gt_matched[img_id][max_iou_idx] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Compute precision/recall
    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    rec = tp_cum / npos if npos>0 else np.zeros_like(tp_cum)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_voc_ap(rec, prec)
    return ap

# ============================
# MAIN
# ============================

GT, image_sizes = load_gt(GT_CSV)
PRED = load_pred(PRED_CSV)

aps = []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    ap = evaluate_class(GT, PRED, cls_idx, IOU_THRESHOLD)
    print(f"AP@0.5 for class '{cls_name}': {ap:.4f}")
    aps.append(ap)

map50 = np.mean(aps)
print("="*50)
print(f"Offline mAP@0.5: {map50:.4f}")
print("="*50)
