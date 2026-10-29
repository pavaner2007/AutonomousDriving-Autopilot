#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Driving Perception & Decision System (Standalone)
============================================================

Runs the multi-modal autonomous driving perception and vehicle control
pipeline on video files, webcam streams, or a built-in simulation feed:
- YOLO11 Object Detection (vehicles, pedestrians, traffic lights)
- HSV-based Traffic Light Color State Analysis (Red/Yellow/Green)
- Lane Boundary Detection & Center Line Navigation
- Adaptive Speed Control, Obstacle Avoidance & Emergency Braking
- High-Tech Tesla-style Autopilot HUD Display

Usage:
    # Run built-in interactive autonomous drive simulation:
    python autonomous_driving_standalone.py

    # Run on a video file:
    python autonomous_driving_standalone.py --video path/to/driving_video.mp4

    # Run on webcam:
    python autonomous_driving_standalone.py --camera 0

Keys:
    'q'     : Quit
    'space' : Pause / Resume
    't'     : Cycle traffic light (Red -> Green -> Yellow)
    'p'     : Trigger pedestrian crossing in simulation
"""

import sys
import os
import time
import math
import argparse
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics is not installed. YOLO object detection will use mock fallback.")


class StandalonePerceptionSystem:
    def __init__(self, yolo_model_path=None, conf_thresh=0.3):
        self.conf_thresh = conf_thresh
        self.use_custom_yolo = False
        
        # Load YOLO model
        if YOLO_AVAILABLE:
            if yolo_model_path and os.path.exists(yolo_model_path):
                print(f"[Init] Loading custom YOLO model: {yolo_model_path}")
                self.yolo_model = YOLO(yolo_model_path)
                self.use_custom_yolo = True
                # CARLA trained classes: 0:vehicle, 1:pedestrian, 2:traffic_light, 3:speed_limit
                self.custom_classes = {0: 'vehicle', 1: 'pedestrian', 2: 'traffic_light', 3: 'speed_limit'}
            else:
                print("[Init] Loading YOLO11n pretrained model...")
                self.yolo_model = YOLO("yolo11n.pt")
                # Standard COCO classes of interest
                self.coco_classes = {
                    0: 'pedestrian',
                    1: 'bicycle',
                    2: 'vehicle',      # car
                    3: 'motorcycle',
                    5: 'vehicle',      # bus
                    7: 'vehicle',      # truck
                    9: 'traffic_light',
                    11: 'stop_sign'
                }
        else:
            self.yolo_model = None

        # Control Parameters matching CARLA project
        self.MAX_STEER_DEGREES = 70.0
        self.STEERING_THRESHOLD = 5.0
        self.base_target_speed = 20.0  # km/h
        self.current_speed = 0.0       # km/h
        self.throttle = 0.0
        self.brake = 0.0
        self.steering_angle = 0.0

        # Safety thresholds
        self.safe_distance_vehicle = 15.0      # meters
        self.safe_distance_pedestrian = 10.0   # meters
        self.safe_distance_traffic_light = 20.0 # meters

        # State tracking
        self.traffic_light_state = "None"
        self.should_stop = False
        self.current_action = "CRUISING"

    # =========================================================================
    # TRAFFIC LIGHT HSV DETECTION & CLASSIFICATION
    # =========================================================================
    def detect_traffic_light_color(self, img_crop):
        """
        Detect traffic light color using HSV color space.
        Tuned for autonomous traffic light color identification.
        """
        if img_crop is None or img_crop.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

        # Red ranges (covers orange-red and violet-red)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        # Yellow/Amber: 16-40
        lower_yellow = np.array([16, 70, 70])
        upper_yellow = np.array([40, 255, 255])

        # Green: 40-95
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([95, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.add(mask_red1, mask_red2)

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)

        threshold = 6
        if red_pixels > threshold and red_pixels >= yellow_pixels and red_pixels >= green_pixels:
            return "Red"
        elif yellow_pixels > threshold and yellow_pixels >= red_pixels and yellow_pixels >= green_pixels:
            return "Yellow"
        elif green_pixels > threshold and green_pixels >= red_pixels and green_pixels >= yellow_pixels:
            return "Green"

        return "Unknown"

    def estimate_distance(self, bbox_height, obj_type):
        """Estimate distance (meters) based on object height in camera view."""
        if bbox_height < 1:
            return float('inf')
        focal_length = 320.0
        real_heights = {
            'vehicle': 1.5,
            'pedestrian': 1.7,
            'traffic_light': 0.7,
            'bicycle': 1.4,
            'motorcycle': 1.4,
            'stop_sign': 0.8
        }
        real_height = real_heights.get(obj_type, 1.5)
        return (real_height * focal_length) / bbox_height

    # =========================================================================
    # YOLO OBJECT DETECTION PIPELINE
    # =========================================================================
    def process_yolo_detections(self, frame, debug_image):
        should_stop = False
        min_vehicle_dist = float('inf')
        min_pedestrian_dist = float('inf')
        closest_traffic_light_color = None
        closest_traffic_light_dist = float('inf')
        detected_objects = []

        if self.yolo_model is None:
            return should_stop, 1.0, debug_image, "None", detected_objects

        h, w = frame.shape[:2]
        results = self.yolo_model(frame, verbose=False, conf=self.conf_thresh)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if self.use_custom_yolo:
                if cls_id not in self.custom_classes:
                    continue
                name = self.custom_classes[cls_id]
            else:
                if cls_id not in self.coco_classes:
                    continue
                name = self.coco_classes[cls_id]

            bbox_height = y2 - y1
            distance = self.estimate_distance(bbox_height, name)
            detected_objects.append((name, conf, distance, (x1, y1, x2, y2)))

            # Pick colors
            if name == 'vehicle':
                color = (255, 120, 0)  # Cyan-blue
                if distance < min_vehicle_dist:
                    min_vehicle_dist = distance
            elif name == 'pedestrian':
                color = (0, 0, 255)    # Red
                if distance < min_pedestrian_dist:
                    min_pedestrian_dist = distance
            elif name == 'traffic_light':
                color = (0, 200, 255)  # Orange/Yellow
                # Crop and analyze color
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                if x2_c > x1_c and y2_c > y1_c:
                    tl_crop = frame[y1_c:y2_c, x1_c:x2_c]
                    tl_color = self.detect_traffic_light_color(tl_crop)
                    if distance < closest_traffic_light_dist:
                        closest_traffic_light_dist = distance
                        closest_traffic_light_color = tl_color
                    
                    # Badge color
                    tc = (0, 0, 255) if tl_color == 'Red' else (0, 255, 255) if tl_color == 'Yellow' else (0, 255, 0) if tl_color == 'Green' else (200, 200, 200)
                    cv2.putText(debug_image, f"TL: {tl_color}", (x1, max(20, y1 - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 2)
            else:
                color = (180, 180, 0)

            # Draw 3D-effect bounding box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            # Label
            label = f"{name} {conf:.2f} [{distance:.1f}m]"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            cv2.rectangle(debug_image, (x1, y1 - 18), (x1 + t_size[0] + 4, y1), color, -1)
            cv2.putText(debug_image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        # Decision Logic: Stop conditions
        tl_state = closest_traffic_light_color if closest_traffic_light_color else "None"
        if closest_traffic_light_color in ["Red", "Yellow"] and closest_traffic_light_dist < self.safe_distance_traffic_light:
            should_stop = True
            self.current_action = f"STOPPED: {closest_traffic_light_color.upper()} LIGHT"
        elif min_pedestrian_dist < self.safe_distance_pedestrian:
            should_stop = True
            self.current_action = "EMERGENCY BRAKE: PEDESTRIAN"
        elif min_vehicle_dist < 4.0:
            should_stop = True
            self.current_action = "EMERGENCY BRAKE: VEHICLE CLOSE"
        else:
            self.current_action = "LANE FOLLOWING / CRUISING"

        # Adaptive speed modifier
        speed_modifier = 1.0
        if min_vehicle_dist < self.safe_distance_vehicle:
            speed_modifier = max(0.2, min_vehicle_dist / self.safe_distance_vehicle)
            if not should_stop:
                self.current_action = "SLOWING: VEHICLE AHEAD"

        return should_stop, speed_modifier, debug_image, tl_state, detected_objects

    # =========================================================================
    # LANE DETECTION & STEERING
    # =========================================================================
    def detect_lanes_and_steering(self, frame, debug_image):
        """
        Fast robust lane boundary detection and target steering angle computation.
        """
        h, w = frame.shape[:2]
        # Region of interest (lower half)
        roi_top = int(h * 0.52)
        roi = frame[roi_top:h, 0:w]

        # Convert to HSV and HLS for road lane marking isolation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)

        # Yellow line mask (center lane)
        lower_yellow = np.array([15, 60, 90])
        upper_yellow = np.array([32, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White line mask (boundary lanes)
        lower_white = np.array([0, 180, 0])
        upper_white = np.array([255, 255, 255])
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        lane_mask = cv2.bitwise_or(mask_yellow, mask_white)

        # Smooth and find lane center
        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

        mid_x = w // 2
        left_points = []
        right_points = []

        # Scan rows from bottom upwards
        scan_rows = np.linspace(h - 1, roi_top + 10, num=15, dtype=int)
        for y in scan_rows:
            row_idx = y - roi_top
            if row_idx < 0 or row_idx >= lane_mask.shape[0]:
                continue
            row = lane_mask[row_idx, :]
            active_indices = np.where(row > 0)[0]
            if len(active_indices) > 0:
                left_candidates = active_indices[active_indices < mid_x]
                right_candidates = active_indices[active_indices >= mid_x]
                if len(left_candidates) > 0:
                    left_points.append((int(np.mean(left_candidates)), y))
                if len(right_candidates) > 0:
                    right_points.append((int(np.mean(right_candidates)), y))

        # Default lanes if not enough detected
        if len(left_points) < 3:
            left_points = [(int(w * 0.22), h), (int(w * 0.42), roi_top)]
        if len(right_points) < 3:
            right_points = [(int(w * 0.78), h), (int(w * 0.58), roi_top)]

        # Overlay lane corridor
        overlay = debug_image.copy()
        pts_left = np.array(sorted(left_points, key=lambda p: p[1]))
        pts_right = np.array(sorted(right_points, key=lambda p: p[1], reverse=True))

        corridor_pts = np.vstack([pts_left, pts_right])
        cv2.fillPoly(overlay, [corridor_pts], (0, 220, 100))
        cv2.addWeighted(overlay, 0.25, debug_image, 0.75, 0, debug_image)

        # Draw left and right lane curves
        for i in range(len(left_points) - 1):
            cv2.line(debug_image, left_points[i], left_points[i + 1], (0, 255, 255), 3)
        for i in range(len(right_points) - 1):
            cv2.line(debug_image, right_points[i], right_points[i + 1], (255, 255, 255), 3)

        # Center line and steering angle
        bottom_left_x = sorted(left_points, key=lambda p: p[1])[-1][0]
        bottom_right_x = sorted(right_points, key=lambda p: p[1])[-1][0]
        lane_bottom_center = (bottom_left_x + bottom_right_x) // 2

        top_left_x = sorted(left_points, key=lambda p: p[1])[0][0]
        top_right_x = sorted(right_points, key=lambda p: p[1])[0][0]
        lane_top_center = (top_left_x + top_right_x) // 2

        # Draw target centerline
        cv2.line(debug_image, (lane_bottom_center, h), (lane_top_center, roi_top), (0, 165, 255), 2)
        cv2.circle(debug_image, (lane_top_center, roi_top), 6, (0, 165, 255), -1)

        # Lateral deviation in pixels
        lateral_offset_px = lane_bottom_center - mid_x
        # Angle relative to vertical
        dx = lane_top_center - lane_bottom_center
        dy = h - roi_top
        steer_angle_deg = math.degrees(math.atan2(dx, dy))

        # Combine angle and lateral offset for stable steering
        target_steering = steer_angle_deg + (lateral_offset_px / 12.0)
        target_steering = np.clip(target_steering, -self.MAX_STEER_DEGREES, self.MAX_STEER_DEGREES)

        return target_steering, debug_image

    # =========================================================================
    # VEHICLE DYNAMICS & CONTROL
    # =========================================================================
    def update_vehicle_controls(self, should_stop, speed_modifier, target_steering):
        # Steering smoothing
        self.steering_angle = 0.7 * self.steering_angle + 0.3 * target_steering

        # Turn speed reduction (reduce speed by up to 30% in sharp turns)
        turn_factor = 1.0 - min(0.3, abs(self.steering_angle) / 70.0 * 0.3)
        effective_target = self.base_target_speed * speed_modifier * turn_factor

        if should_stop:
            self.throttle = 0.0
            self.brake = 1.0
            self.current_speed = max(0.0, self.current_speed - 1.8)  # Fast braking
        else:
            self.brake = 0.0
            if self.current_speed < effective_target:
                self.throttle = min(1.0, self.throttle + 0.08)
                self.current_speed = min(effective_target, self.current_speed + 0.6)
            else:
                self.throttle = max(0.2, self.throttle - 0.05)
                self.current_speed = max(effective_target, self.current_speed - 0.4)

    # =========================================================================
    # TESLA-STYLE AUTOPILOT HUD DASHBOARD
    # =========================================================================
    def draw_hud(self, image, fps):
        h, w = image.shape[:2]

        # Top status bar
        status_bar = image.copy()
        cv2.rectangle(status_bar, (0, 0), (w, 55), (15, 15, 20), -1)
        cv2.addWeighted(status_bar, 0.85, image, 0.15, 0, image)

        # Autopilot logo & Status
        cv2.circle(image, (30, 27), 12, (0, 215, 255), 2)
        cv2.putText(image, "AUTOPILOT", (50, 33), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        # Action badge color
        action_col = (0, 255, 120)  # Green default
        if "STOPPED" in self.current_action or "BRAKE" in self.current_action:
            action_col = (0, 80, 255)  # Bright Red/Orange
        elif "SLOWING" in self.current_action:
            action_col = (0, 220, 255)  # Yellow
        cv2.putText(image, f"STATE: {self.current_action}", (240, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_col, 2)

        # FPS counter
        cv2.putText(image, f"FPS: {fps:.1f}", (w - 110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Left Telemetry Panel
        panel_w = 260
        panel_h = 240
        panel_x = 20
        panel_y = 70
        panel = image.copy()
        cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), -1)
        cv2.addWeighted(panel, 0.82, image, 0.18, 0, image)
        cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 70, 85), 1)

        # Digital Speedometer
        cv2.putText(image, f"{int(self.current_speed)}", (panel_x + 20, panel_y + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 2)
        cv2.putText(image, "KM/H", (panel_x + 115, panel_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 160, 180), 1)
        cv2.putText(image, f"LIMIT: {int(self.base_target_speed)}", (panel_x + 115, panel_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Steering Indicator
        steer_text = f"STEER: {self.steering_angle:+.1f}°"
        arrow = "◀" if self.steering_angle < -1.5 else ("▶" if self.steering_angle > 1.5 else "▲")
        cv2.putText(image, f"{steer_text} {arrow}", (panel_x + 20, panel_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Steering bar
        bar_cx = panel_x + panel_w // 2
        bar_y = panel_y + 120
        cv2.line(image, (bar_cx - 80, bar_y), (bar_cx + 80, bar_y), (70, 80, 95), 2)
        cv2.line(image, (bar_cx, bar_y - 5), (bar_cx, bar_y + 5), (200, 200, 200), 2)
        steer_px = int(np.clip(self.steering_angle / 70.0 * 80, -80, 80))
        cv2.circle(image, (bar_cx + steer_px, bar_y), 5, (0, 215, 255), -1)

        # Throttle & Brake Gauges
        th_w = int(self.throttle * 140)
        br_w = int(self.brake * 140)

        cv2.putText(image, "THROTTLE", (panel_x + 20, panel_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.rectangle(image, (panel_x + 100, panel_y + 144), (panel_x + 100 + 140, panel_y + 156), (40, 50, 60), -1)
        cv2.rectangle(image, (panel_x + 100, panel_y + 144), (panel_x + 100 + th_w, panel_y + 156), (0, 220, 100), -1)

        cv2.putText(image, "BRAKE", (panel_x + 20, panel_y + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.rectangle(image, (panel_x + 100, panel_y + 174), (panel_x + 100 + 140, panel_y + 186), (40, 50, 60), -1)
        cv2.rectangle(image, (panel_x + 100, panel_y + 174), (panel_x + 100 + br_w, panel_y + 186), (0, 60, 255), -1)

        # Traffic light badge in panel
        tl_col = (100, 100, 100)
        if self.traffic_light_state == "Red":
            tl_col = (0, 0, 255)
        elif self.traffic_light_state == "Yellow":
            tl_col = (0, 255, 255)
        elif self.traffic_light_state == "Green":
            tl_col = (0, 255, 0)

        cv2.putText(image, f"LIGHT: {self.traffic_light_state}", (panel_x + 20, panel_y + 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.circle(image, (panel_x + 180, panel_y + 215), 9, tl_col, -1)

        # Bottom help banner
        cv2.putText(image, "[Q] Quit  |  [Space] Pause  |  [T] Cycle Light  |  [P] Pedestrian Event",
                    (w // 2 - 240, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 210, 230), 1, cv2.LINE_AA)

        return image


# =============================================================================
# SYNTHETIC CARLA-LIKE DRIVING SCENE SIMULATOR (Self-Contained)
# =============================================================================
class DrivingSceneSimulator:
    """
    Simulates dynamic autonomous vehicle camera frames with road curves,
    lane markings, surrounding traffic, pedestrians, and cycling traffic lights.
    """
    def __init__(self, width=960, height=540):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.curve = 0.0

        # Traffic light simulation state
        self.tl_cycle = ["Green", "Yellow", "Red"]
        self.tl_index = 0
        self.tl_timer = 0

        # Obstacle simulation
        self.car_dist = 35.0
        self.pedestrian_dist = 60.0
        self.pedestrian_active = False

    def cycle_traffic_light(self):
        self.tl_index = (self.tl_index + 1) % len(self.tl_cycle)

    def trigger_pedestrian(self):
        self.pedestrian_active = True
        self.pedestrian_dist = 22.0

    def generate_frame(self, speed_kmh, steer_deg):
        self.frame_count += 1
        w, h = self.width, self.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Horizon and ground
        horizon = int(h * 0.45)
        # Sky gradient
        frame[0:horizon, :] = [210, 180, 130]  # Light sky
        # Road ground
        frame[horizon:h, :] = [65, 68, 72]     # Dark asphalt

        # Road perspective parameters
        speed_factor = speed_kmh / 20.0
        self.curve = math.sin(self.frame_count * 0.02) * 120.0

        vp_x = int(w // 2 + self.curve)
        vp_y = horizon

        road_bottom_left = int(w * 0.1)
        road_bottom_right = int(w * 0.9)

        # Draw road polygon
        road_poly = np.array([
            (road_bottom_left, h),
            (vp_x - 30, vp_y),
            (vp_x + 30, vp_y),
            (road_bottom_right, h)
        ], np.int32)
        cv2.fillPoly(frame, [road_poly], (50, 52, 55))

        # Lane markings animation (moving towards viewer)
        offset = (self.frame_count * int(speed_factor * 8)) % 80
        num_segments = 10
        for i in range(num_segments):
            progress1 = ((i * 80 + offset) / (num_segments * 80.0)) ** 2
            progress2 = (((i + 0.5) * 80 + offset) / (num_segments * 80.0)) ** 2

            if progress1 > 1.0 or progress2 > 1.0:
                continue

            y1 = int(vp_y + (h - vp_y) * progress1)
            y2 = int(vp_y + (h - vp_y) * progress2)

            # Center yellow dashed line
            x1 = int(vp_x + (w // 2 - vp_x) * progress1)
            x2 = int(vp_x + (w // 2 - vp_x) * progress2)
            cv2.line(frame, (x1, y1), (x2, y2), (20, 200, 240), max(2, int(6 * progress2)))

            # Left solid-like markings
            lx1 = int((vp_x - 30) + (road_bottom_left - (vp_x - 30)) * progress1)
            lx2 = int((vp_x - 30) + (road_bottom_left - (vp_x - 30)) * progress2)
            cv2.line(frame, (lx1, y1), (lx2, y2), (230, 230, 230), max(2, int(5 * progress2)))

            # Right solid-like markings
            rx1 = int((vp_x + 30) + (road_bottom_right - (vp_x + 30)) * progress1)
            rx2 = int((vp_x + 30) + (road_bottom_right - (vp_x + 30)) * progress2)
            cv2.line(frame, (rx1, y1), (rx2, y2), (230, 230, 230), max(2, int(5 * progress2)))

        # Update Traffic Light cycle
        self.tl_timer += 1
        if self.tl_timer > 140:
            self.tl_timer = 0
            self.tl_index = (self.tl_index + 1) % len(self.tl_cycle)
        current_light = self.tl_cycle[self.tl_index]

        # Draw simulated Traffic Light on gantry (right side)
        tl_x, tl_y = int(w * 0.75), int(h * 0.22)
        cv2.rectangle(frame, (tl_x - 14, tl_y - 38), (tl_x + 14, tl_y + 38), (25, 25, 25), -1)
        cv2.rectangle(frame, (tl_x - 14, tl_y - 38), (tl_x + 14, tl_y + 38), (80, 80, 80), 2)
        # Pole
        cv2.line(frame, (tl_x, tl_y + 38), (tl_x, int(h * 0.55)), (70, 70, 70), 4)

        # Light bulbs
        # Red
        r_col = (0, 0, 255) if current_light == "Red" else (0, 0, 60)
        cv2.circle(frame, (tl_x, tl_y - 22), 8, r_col, -1)
        # Yellow
        y_col = (0, 230, 240) if current_light == "Yellow" else (0, 60, 60)
        cv2.circle(frame, (tl_x, tl_y), 8, y_col, -1)
        # Green
        g_col = (0, 255, 0) if current_light == "Green" else (0, 60, 0)
        cv2.circle(frame, (tl_x, tl_y + 22), 8, g_col, -1)

        # Draw simulated Leading Vehicle ahead
        if current_light == "Red":
            # Vehicle slows down ahead
            self.car_dist = max(11.0, self.car_dist - 0.2)
        elif speed_kmh > 0:
            self.car_dist = np.clip(self.car_dist + (18.0 - speed_kmh) * 0.05, 12.0, 45.0)

        car_scale = 1.0 / (self.car_dist / 10.0)
        car_w = int(120 * car_scale)
        car_h = int(80 * car_scale)
        car_cx = int(vp_x + (w // 2 - vp_x) * (1.0 - 0.2 * self.car_dist / 30.0))
        car_cy = int(horizon + (h - horizon) * (0.3 + 0.4 * (1.0 - self.car_dist / 50.0)))

        # Car body
        cv2.rectangle(frame, (car_cx - car_w // 2, car_cy - car_h), (car_cx + car_w // 2, car_cy), (180, 50, 40), -1)
        # Car roof
        roof_w = int(car_w * 0.7)
        cv2.rectangle(frame, (car_cx - roof_w // 2, car_cy - car_h - int(car_h * 0.6)),
                      (car_cx + roof_w // 2, car_cy - car_h), (120, 30, 20), -1)
        # Tail lights
        tail_col = (0, 0, 255) if current_light == "Red" or speed_kmh == 0 else (0, 0, 160)
        cv2.circle(frame, (car_cx - car_w // 2 + 8, car_cy - int(car_h * 0.3)), max(2, int(6 * car_scale)), tail_col, -1)
        cv2.circle(frame, (car_cx + car_w // 2 - 8, car_cy - int(car_h * 0.3)), max(2, int(6 * car_scale)), tail_col, -1)

        # Draw Pedestrian crossing event
        if self.pedestrian_active:
            self.pedestrian_dist -= 0.3
            ped_scale = 1.0 / (self.pedestrian_dist / 8.0)
            ped_h = int(90 * ped_scale)
            ped_w = int(28 * ped_scale)
            ped_x = int(w * 0.45 + (15 - self.pedestrian_dist) * 20)
            ped_y = int(horizon + (h - horizon) * 0.55)

            # Person silhouette
            cv2.circle(frame, (ped_x, ped_y - ped_h + 10), int(10 * ped_scale), (30, 30, 220), -1)
            cv2.rectangle(frame, (ped_x - ped_w // 2, ped_y - ped_h + 20), (ped_x + ped_w // 2, ped_y), (40, 40, 180), -1)

            if self.pedestrian_dist < 4.0:
                self.pedestrian_active = False

        return frame


# =============================================================================
# MAIN RUN LOOP
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving System - Standalone Runner")
    parser.add_argument("--video", type=str, default=None, help="Path to input driving video file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index for live webcam feed")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold (default: 0.3)")
    args = parser.parse_args()

    print("=" * 65)
    print("   AUTONOMOUS DRIVING PERCEPTION & AUTOPILOT SYSTEM  ")
    print("   Author: Pavan (pavaner2007)                       ")
    print("=" * 65)

    system = StandalonePerceptionSystem(yolo_model_path=args.weights, conf_thresh=args.conf)

    # Determine input source
    cap = None
    simulator = None

    if args.video:
        if not os.path.exists(args.video):
            print(f"[Error] Video file not found: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        print(f"[Source] Playing video: {args.video}")
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        print(f"[Source] Streaming from camera device: {args.camera}")
    else:
        print("[Source] Running Built-in Dynamic Autonomous Driving Simulation")
        print("         (Simulates live road, curves, vehicles, pedestrians & traffic lights)")
        simulator = DrivingSceneSimulator(width=960, height=540)

    print("\nControls:")
    print("  [Q]     : Quit")
    print("  [Space] : Pause / Resume")
    print("  [T]     : Cycle Traffic Light (Green -> Yellow -> Red)")
    print("  [P]     : Trigger Pedestrian Crossing event")
    print("-" * 65)

    window_name = "Autonomous Driving Autopilot HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    paused = False
    prev_time = time.time()

    while True:
        loop_start = time.time()

        if not paused:
            # 1. Fetch next frame
            if simulator:
                raw_frame = simulator.generate_frame(system.current_speed, system.steering_angle)
            else:
                ret, raw_frame = cap.read()
                if not ret:
                    print("[Stream] Video ended, looping back to start...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                    if not ret:
                        break

            debug_image = raw_frame.copy()

            # 2. Perception: YOLO Object Detection
            should_stop, speed_modifier, debug_image, tl_state, objects = system.process_yolo_detections(raw_frame, debug_image)
            system.traffic_light_state = tl_state

            # 3. Perception: Lane Detection & Steering
            target_steer, debug_image = system.detect_lanes_and_steering(raw_frame, debug_image)

            # 4. Vehicle Control Update
            system.update_vehicle_controls(should_stop, speed_modifier, target_steer)

            # 5. Render Autopilot HUD
            now = time.time()
            fps = 1.0 / max(1e-4, (now - prev_time))
            prev_time = now

            hud_frame = system.draw_hud(debug_image, fps)
            cv2.imshow(window_name, hud_frame)

        # Handle user keyboard inputs
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # Space
            paused = not paused
        elif key == ord('t') or key == ord('T'):
            if simulator:
                simulator.cycle_traffic_light()
                print(">> [Event] Traffic Light cycled manually!")
        elif key == ord('p') or key == ord('P'):
            if simulator:
                simulator.trigger_pedestrian()
                print(">> [Event] Pedestrian crossing triggered!")

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("\n[Exit] Autonomous Driving System stopped gracefully.")


if __name__ == "__main__":
    main()

