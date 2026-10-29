#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find Invalid Classes in YOLO Dataset
=====================================

Utility to find label files containing class IDs not in the allowed list.
Useful for validating dataset integrity before training.

Author: Justin Mascarenhas
Institution: Rochester Institute of Technology
Course: CMPE 789 - Robot Perception
Date: December 2025
License: MIT
"""

import os
from glob import glob
from collections import defaultdict

# ---------- CONFIG ---------- #
# UPDATE THIS PATH to your dataset location
DATASET_ROOT = "../datasets/yolo_carla_dataset"
SPLITS = ["train", "val", "test"]

# Allowed class IDs (modify this list as needed)
ALLOWED_CLASS_IDS = [0, 1, 21, 22, 23, 31, 32, 33]  # 0=vehicle, 1=pedestrian
# ---------------------------- #


def check_file_for_invalid_classes(label_path, allowed_ids):
    """
    Check a single label file for invalid class IDs.
    
    Returns:
        invalid_ids: set of invalid class IDs found in this file
    """
    invalid_ids = set()
    
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    if class_id not in allowed_ids:
                        invalid_ids.add(class_id)
                except ValueError:
                    # Skip malformed lines
                    continue
    except Exception as e:
        print(f"[ERROR] Could not read {label_path}: {e}")
    
    return invalid_ids


def scan_split(split, dataset_root, allowed_ids):
    """
    Scan all label files in a split and find files with invalid class IDs.
    
    Returns:
        files_with_issues: dict of {file_path: set_of_invalid_ids}
    """
    label_dir = os.path.join(dataset_root, split, "labels_original")
    
    if not os.path.isdir(label_dir):
        print(f"[SKIP] {split}: {label_dir} does not exist")
        return {}
    
    files = sorted(glob(os.path.join(label_dir, "*.txt")))
    print(f"\n[{split.upper()}] Scanning {len(files)} label files in {label_dir}...")
    
    files_with_issues = {}
    
    for label_file in files:
        invalid_ids = check_file_for_invalid_classes(label_file, allowed_ids)
        if invalid_ids:
            files_with_issues[label_file] = invalid_ids
    
    return files_with_issues


def main():
    print("=" * 70)
    print("FINDING FILES WITH INVALID CLASS IDs")
    print("=" * 70)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Allowed class IDs: {ALLOWED_CLASS_IDS}")
    print(f"Splits to check: {SPLITS}")
    
    # Track all invalid class IDs found across dataset
    all_invalid_ids = set()
    all_files_with_issues = {}
    
    # Scan each split
    for split in SPLITS:
        files_with_issues = scan_split(split, DATASET_ROOT, ALLOWED_CLASS_IDS)
        
        if files_with_issues:
            print(f"\n  ⚠️  Found {len(files_with_issues)} files with invalid class IDs:")
            for file_path, invalid_ids in sorted(files_with_issues.items()):
                print(f"    {file_path}")
                print(f"      Invalid IDs: {sorted(invalid_ids)}")
                all_invalid_ids.update(invalid_ids)
            all_files_with_issues.update(files_with_issues)
        else:
            print(f"  ✓ All files in {split} have valid class IDs")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files with invalid class IDs: {len(all_files_with_issues)}")
    
    if all_invalid_ids:
        print(f"All invalid class IDs found: {sorted(all_invalid_ids)}")
        print("\nFiles grouped by invalid class ID:")
        
        # Group files by which invalid IDs they contain
        id_to_files = defaultdict(list)
        for file_path, invalid_ids in all_files_with_issues.items():
            for invalid_id in invalid_ids:
                id_to_files[invalid_id].append(file_path)
        
        for invalid_id in sorted(id_to_files.keys()):
            print(f"\n  Class ID {invalid_id}: appears in {len(id_to_files[invalid_id])} files")
            for file_path in id_to_files[invalid_id][:5]:  # Show first 5
                print(f"    {file_path}")
            if len(id_to_files[invalid_id]) > 5:
                print(f"    ... and {len(id_to_files[invalid_id]) - 5} more")
    else:
        print("✓ All files contain only valid class IDs!")


if __name__ == "__main__":
    main()
