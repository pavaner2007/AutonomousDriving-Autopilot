#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Dataset Class Remapping
================================

Utility to validate class ID remapping in YOLO dataset.
Verifies that class IDs have been correctly mapped to consolidated categories.

Author: Justin Mascarenhas
Institution: Rochester Institute of Technology
Course: CMPE 789 - Robot Perception
Date: December 2025
License: MIT
"""

import os
from glob import glob
from collections import Counter

import matplotlib.pyplot as plt

# ---------- CONFIG ---------- #
# UPDATE THIS PATH to your dataset location
dataset_root = "../datasets/yolo_carla_dataset"
splits = ["train", "val", "test"]

# Original CARLA-style IDs
orig_id_to_name = {
    0:  "vehicle",
    1:  "pedestrian",
    2:  "traffic_red_light",
    21: "traffic_red_light",
    3:  "traffic_yellow_light",
    22: "traffic_yellow_light",
    4:  "traffic_green_light",
    23: "traffic_green_light",
    5: "speed_limit_30",
    31: "speed_limit_30",
    6: "speed_limit_60",
    32: "speed_limit_60",
    7: "speed_limit_90",
    33: "speed_limit_90",
}

# Remapped YOLO IDs 0–7
new_id_to_name = {
    0: "vehicle",
    1: "pedestrian",
    2: "traffic_red_light",
    3: "traffic_yellow_light",
    4: "traffic_green_light",
    5: "speed_limit_30",
    6: "speed_limit_60",
    7: "speed_limit_90",
}

# Mapping from original ID -> new ID
old_to_new = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    21: 2,
    22: 3,
    23: 4,
    31: 5,
    32: 6,
    33: 7,
}

STRICT = True  # True = error out if an unmapped ID is seen
# ---------------------------- #


def scan_label_counts(label_dir):
    """Return Counter of class_id -> count for a label directory."""
    counts = Counter()
    if not os.path.isdir(label_dir):
        print(f"[WARN] {label_dir} does not exist")
        return counts

    files = glob(os.path.join(label_dir, "*.txt"))
    for lf in files:
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                counts[cid] += 1
    return counts


def plot_bar(counts, id_to_name, title):
    """Bar chart of class counts."""
    if not counts:
        print(f"[INFO] No labels for {title}")
        return

    ids = sorted(counts.keys())
    vals = [counts[i] for i in ids]
    names = [id_to_name.get(i, str(i)) for i in ids]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(ids)), vals)
    plt.xticks(range(len(ids)), names, rotation=45, ha="right")
    plt.ylabel("instances")
    plt.title(title)
    for i, v in enumerate(vals):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()


def remap_labels_dir(src_dir, dst_dir, old_to_new, strict=True):
    """Remap labels from src_dir -> dst_dir using old_to_new mapping."""
    os.makedirs(dst_dir, exist_ok=True)
    files = glob(os.path.join(src_dir, "*.txt"))
    print(f"[REMAP] {src_dir} -> {dst_dir} ({len(files)} files)")
    unmapped = set()

    for lf in files:
        base = os.path.basename(lf)
        out_path = os.path.join(dst_dir, base)

        with open(lf, "r") as f:
            lines = f.read().strip().splitlines()

        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            try:
                old_cls = int(parts[0])
            except ValueError:
                print(f"[WARN] skip malformed line in {lf}: {line}")
                continue

            if old_cls not in old_to_new:
                unmapped.add(old_cls)
                if strict:
                    raise ValueError(
                        f"Unmapped class ID {old_cls} in {lf}. "
                        f"Add it to old_to_new before remapping."
                    )
                else:
                    print(f"[WARN] {lf}: class {old_cls} not in mapping, skipping box")
                    continue

            new_cls = old_to_new[old_cls]
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))

        with open(out_path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    if unmapped:
        print("[INFO] Unmapped IDs encountered:", sorted(unmapped))
    else:
        print("[INFO] All IDs were mapped successfully.")


# ===== MAIN: for each split, plot orig -> remap -> plot new ===== #
for split in splits:
    print(f"\n========== SPLIT: {split} ==========")
    src_labels = os.path.join(dataset_root, split, "labels_original")
    dst_labels = os.path.join(dataset_root, split, "labels")

    # 1) Plot original distribution
    orig_counts = scan_label_counts(src_labels)
    print("Original counts:", orig_counts)
    plot_bar(orig_counts, orig_id_to_name, f"{split} – ORIGINAL labels")

    # 2) Remap into dst_labels
    remap_labels_dir(src_labels, dst_labels, old_to_new, strict=STRICT)

    # 3) Plot remapped distribution
    new_counts = scan_label_counts(dst_labels)
    print("Remapped counts:", new_counts)
    plot_bar(new_counts, new_id_to_name, f"{split} – REMAPPED labels (0–7)")
