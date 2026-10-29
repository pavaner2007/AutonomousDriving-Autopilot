#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify YOLO Dataset Labels
==========================

Utility to verify YOLO label file format and check for invalid entries.
Checks bounding box format, class IDs, and coordinate ranges.

Author: Justin Mascarenhas
Institution: Rochester Institute of Technology
Course: CMPE 789 - Robot Perception
Date: December 2025
License: MIT
"""

import os
from glob import glob

# ---------- CONFIG ---------- #
# UPDATE THIS PATH to your dataset location
DATASET_ROOT = "../datasets/yolo_carla_dataset"

SPLITS = ["train", "val", "test"]
NC = 8  # number of classes after remap (0..7)
# ---------------------------- #


def check_label_file(path, nc):
    """
    Check a single YOLO label file.

    Returns:
      bad_count: number of invalid lines
      reasons: list of (lineno, reason, line_str)
    """
    bad_count = 0
    reasons = []

    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                bad_count += 1
                reasons.append((lineno, "too_few_values", line))
                continue

            # class id
            try:
                cls = int(parts[0])
            except ValueError:
                bad_count += 1
                reasons.append((lineno, "class_not_int", line))
                continue

            if cls < 0 or cls >= nc:
                bad_count += 1
                reasons.append((lineno, "class_out_of_range", line))
                continue

            # coords
            try:
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                bad_count += 1
                reasons.append((lineno, "coords_not_float", line))
                continue

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                bad_count += 1
                reasons.append((lineno, "center_out_of_range", line))
                continue

            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                bad_count += 1
                reasons.append((lineno, "width_height_out_of_range", line))
                continue

            # optional: check extra columns are floats too (if present)
            for extra in parts[5:]:
                try:
                    float(extra)
                except ValueError:
                    bad_count += 1
                    reasons.append((lineno, "extra_field_not_float", line))
                    break

    return bad_count, reasons


def check_split(split, root, nc):
    """
    Check all label files under <root>/<split>/labels.
    Prints per-file issues and returns global stats.
    """
    label_dir = os.path.join(root, split, "labels")
    if not os.path.isdir(label_dir):
        print(f"[SKIP] {split}: {label_dir} does not exist")
        return 0, 0, []

    files = sorted(glob(os.path.join(label_dir, "*.txt")))
    print(f"\n========== SPLIT: {split} ==========")
    print(f"Found {len(files)} label files in {label_dir}")

    total_files = len(files)
    bad_files = 0
    all_bad = []  # list of (file, bad_count, reasons)

    for lf in files:
        bad_count, reasons = check_label_file(lf, nc)
        if bad_count > 0:
            bad_files += 1
            all_bad.append((lf, bad_count, reasons))
            print(f"\n[BAD] {lf}")
            print(f"  bad lines: {bad_count}")
            # print up to first 5 issues for this file
            for lineno, reason, line in reasons[:5]:
                print(f"    line {lineno}: {reason} | {line}")

    print(f"\n[SUMMARY] {split}:")
    print(f"  total label files : {total_files}")
    print(f"  files with issues : {bad_files}")
    return total_files, bad_files, all_bad


if __name__ == "__main__":
    overall_bad = []

    for split in SPLITS:
        _, _, bad_list = check_split(split, DATASET_ROOT, NC)
        overall_bad.extend(bad_list)

    print("\n========== OVERALL SUMMARY ==========")
    print(f"Total files with issues across all splits: {len(overall_bad)}")
    if overall_bad:
        print("\nFiles with issues (name and bad line count):")
        for path, bad_count, _ in overall_bad:
            print(f"  {path}: {bad_count} bad lines")
