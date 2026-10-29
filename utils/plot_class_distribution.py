#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Class Distribution in YOLO Dataset
========================================

Utility to generate bar graphs showing class distribution for each split.
Useful for verifying dataset balance before training.

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
import argparse

# ---------- CONFIG ---------- #
# UPDATE THIS PATH to your dataset location
DATASET_ROOT = "../datasets/yolo_carla_dataset"
SPLITS = ["train", "val", "test"]

# Class names (update as needed)
CLASS_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "traffic_light_red",
    3: "traffic_light_yellow",
    4: "traffic_light_green",
    5: "speed_limit_30",
    6: "speed_limit_60",
    7: "speed_limit_90",
}
# ---------------------------- #


def scan_label_counts(label_dir):
    """
    Count class IDs in all label files in a directory.
    
    Returns:
        Counter object with {class_id: count}
    """
    counts = Counter()
    
    if not os.path.isdir(label_dir):
        print(f"[WARN] {label_dir} does not exist")
        return counts
    
    files = glob(os.path.join(label_dir, "*.txt"))
    
    for label_file in files:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    counts[class_id] += 1
                except ValueError:
                    continue
    
    return counts


def plot_split_distribution(counts, split_name, class_names, save_path=None):
    """
    Create bar chart for a single split.
    """
    if not counts:
        print(f"[INFO] No labels found for {split_name}")
        return
    
    # Sort by class ID
    class_ids = sorted(counts.keys())
    values = [counts[cid] for cid in class_ids]
    labels = [str(cid) for cid in class_ids]  # Just use class ID numbers
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_ids)), values, color='steelblue', alpha=0.8)
    
    # Customize
    plt.xlabel('Class ID', fontsize=12, fontweight='bold')
    plt.ylabel('Instance Count', fontsize=12, fontweight='bold')
    plt.title(f'{split_name.upper()} Split - Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_ids)), labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_distribution(all_counts, class_names, save_path=None):
    """
    Create combined bar chart showing all splits together.
    """
    if not any(all_counts.values()):
        print("[INFO] No data to plot")
        return
    
    # Get all unique class IDs across all splits
    all_class_ids = set()
    for counts in all_counts.values():
        all_class_ids.update(counts.keys())
    class_ids = sorted(all_class_ids)
    
    # Prepare data for grouped bar chart
    splits = sorted(all_counts.keys())
    x = range(len(class_ids))
    width = 0.25
    
    colors = {'train': 'steelblue', 'val': 'coral', 'test': 'mediumseagreen'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot bars for each split
    for i, split in enumerate(splits):
        counts = all_counts[split]
        values = [counts.get(cid, 0) for cid in class_ids]
        offset = (i - len(splits)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, 
                     label=split.upper(), color=colors.get(split, 'gray'), alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val}', ha='center', va='bottom', fontsize=8)
    
    # Customize
    labels = [str(cid) for cid in class_ids]  # Just use class ID numbers
    ax.set_xlabel('Class ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Instance Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Across All Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot class distribution for YOLO dataset')
    parser.add_argument('--dataset', type=str, default=DATASET_ROOT,
                       help='Dataset root directory')
    parser.add_argument('--splits', nargs='+', default=SPLITS,
                       help='Splits to analyze (default: train val test)')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files instead of displaying')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: dataset/_plots_distribution)')
    
    args = parser.parse_args()
    
    # Setup output directory if saving
    if args.save:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(args.dataset, "_plots_distribution")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    print("=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {args.splits}\n")
    
    all_counts = {}
    
    # Scan each split
    for split in args.splits:
        label_dir = os.path.join(args.dataset, split, "labels_merged")
        
        print(f"[{split.upper()}] Scanning {label_dir}...")
        counts = scan_label_counts(label_dir)
        
        if counts:
            all_counts[split] = counts
            print(f"  Found {sum(counts.values())} instances across {len(counts)} classes")
            print(f"  Class distribution: {dict(counts)}")
            
            # Plot individual split
            if args.save:
                save_path = os.path.join(output_dir, f"{split}_distribution.png")
                plot_split_distribution(counts, split, CLASS_NAMES, save_path)
            else:
                plot_split_distribution(counts, split, CLASS_NAMES)
        else:
            print(f"  No labels found")
        
        print()
    
    # Plot combined view
    if all_counts:
        print("[COMBINED] Plotting all splits together...")
        if args.save:
            save_path = os.path.join(output_dir, "combined_distribution.png")
            plot_combined_distribution(all_counts, CLASS_NAMES, save_path)
        else:
            plot_combined_distribution(all_counts, CLASS_NAMES)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_instances = sum(sum(counts.values()) for counts in all_counts.values())
    all_class_ids = set()
    for counts in all_counts.values():
        all_class_ids.update(counts.keys())
    
    print(f"Total instances: {total_instances}")
    print(f"Unique class IDs: {sorted(all_class_ids)}")
    print(f"Total splits analyzed: {len(all_counts)}")
    
    if args.save:
        print(f"\n✓ Plots saved to: {output_dir}")
    else:
        print("\n✓ Done!")


if __name__ == "__main__":
    main()
