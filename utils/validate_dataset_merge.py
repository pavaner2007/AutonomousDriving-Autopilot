#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Dataset Merge
======================

Utility to validate merged YOLO dataset and visualize class distributions
after combining multiple dataset sources.

Author: Justin Mascarenhas
Institution: Rochester Institute of Technology
Course: CMPE 789 - Robot Perception
Date: December 2025
License: MIT
"""

import os
from glob import glob
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt

# ---------- CONFIG ---------- #
# UPDATE THIS PATH to your dataset location
DATASET_ROOT = "../datasets/yolo_carla_dataset"
SPLITS = ["train", "val", "test"]

# Original CARLA IDs -> names (for plotting original)
original_id_to_name = {
    0:  "vehicle",
    1:  "pedestrian",
    2:  "traffic_light_red",      # or traffic light state 1
    3:  "traffic_light_yellow",   # or traffic light state 2
    4:  "traffic_light_green",    # or traffic light state 3
    5:  "speed_limit",            # speed limit variant 1
    7:  "speed_limit_2",          # speed limit variant 2
}

# New merged IDs -> names
new_id_to_name = {
    0: "vehicle",
    1: "pedestrian",
    2: "traffic_light",     # merged red/yellow/green
    3: "speed_limit",       # merged 30/60/90
}

# Mapping from ORIGINAL ID -> MERGED ID
old_to_new = {
    0: 0,   # vehicle → vehicle
    1: 1,   # pedestrian → pedestrian
    2: 2,   # traffic_light_red → traffic_light
    3: 2,   # traffic_light_yellow → traffic_light
    4: 2,   # traffic_light_green → traffic_light
    5: 3,   # speed_limit_30 → speed_limit
    6: 3,   # speed_limit_60 → speed_limit
    7: 3,   # speed_limit_90 → speed_limit
}

STRICT = True     # Don't error on unmapped IDs (like 6 if it doesn't exist)
OUT_DIR = os.path.join(DATASET_ROOT, "_plots_merge")  # where to save PNGs
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------- #


def scan_label_counts(label_dir, ignore_classes_txt=True):
    """Return Counter of class_id -> count for all files in label_dir."""
    counts = Counter()
    if not os.path.isdir(label_dir):
        print(f"[WARN] {label_dir} does not exist")
        return counts

    files = sorted(glob(os.path.join(label_dir, "*.txt")))
    for lf in files:
        if ignore_classes_txt and os.path.basename(lf) == "classes.txt":
            continue
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


def plot_bar(counts, id_to_name, title, save_path):
    """Bar chart of class counts saved to PNG."""
    if not counts:
        print(f"[INFO] No labels for {title}, skipping plot.")
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
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def remap_labels_dir(src_dir, dst_dir, old_to_new, strict=True):
    """Remap labels from src_dir -> dst_dir using old_to_new mapping."""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob(os.path.join(src_dir, "*.txt")))
    print(f"[REMAP] {src_dir} -> {dst_dir} ({len(files)} files)")
    unmapped = set()

    for lf in files:
        base = os.path.basename(lf)

        # Skip classes.txt – it's not a YOLO label file
        if base == "classes.txt":
            continue

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


if __name__ == "__main__":
    for split in SPLITS:
        print(f"\n========== SPLIT: {split} ==========")

        src_labels = os.path.join(DATASET_ROOT, split, "labels")
        dst_labels = os.path.join(DATASET_ROOT, split, "labels_merged")

        # 1) Original distribution
        orig_counts = scan_label_counts(src_labels)
        print("Original counts:", orig_counts)
        orig_plot_path = os.path.join(OUT_DIR, f"{split}_original_counts.png")
        plot_bar(orig_counts, original_id_to_name, f"{split} – ORIGINAL labels", orig_plot_path)

        # 2) Remap (merge signals)
        remap_labels_dir(src_labels, dst_labels, old_to_new, strict=STRICT)

        # 3) Merged distribution
        merged_counts = scan_label_counts(dst_labels)
        print("Merged counts:", merged_counts)
        merged_plot_path = os.path.join(OUT_DIR, f"{split}_merged_counts.png")
        plot_bar(merged_counts, new_id_to_name, f"{split} – MERGED labels", merged_plot_path)

    print("\nDone. Check plots in:", OUT_DIR)
