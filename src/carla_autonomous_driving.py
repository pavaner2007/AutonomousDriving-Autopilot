#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA Autonomous Driving System
===============================

A complete autonomous driving pipeline for the CARLA 0.9.15 simulator featuring:
- YOLO11 object detection (vehicles, pedestrians, traffic lights, speed signs)
- Ultra Fast Lane Detection (UFLD) for lane boundary identification
- Lane-following with waypoint-based fallback navigation
- HSV-based traffic light color recognition
- Real-time decision making and vehicle control

Author: Pavan (pavaner2007)
Date: 2026
License: MIT

Usage:
    python carla_autonomous_driving.py --vehicles 30 --pedestrians 10

Requirements:
    - CARLA 0.9.15 simulator running
    - PyTorch with CUDA (optional, for GPU acceleration)
    - Ultralytics YOLO
    - OpenCV

Features:
    - Waypoint-based route following
    - YOLO for vehicle/pedestrian/traffic_light detection
    - Speed capped at 20 km/h
    - HSV traffic light color detection
    - Steering threshold: 5° (lane following vs waypoint)
    - Camera: 1640x590, FOV=150°
"""

import glob
import os
import sys
import argparse

# ==============================================================================
# CARLA Python API Setup
# UPDATE THE PATH BELOW to point to your CARLA installation directory.
# Example: '/opt/carla-simulator' or 'C:/CARLA_0.9.15'
# The .egg file is located in: <CARLA_ROOT>/PythonAPI/carla/dist/
# ==============================================================================
CARLA_PATH = '../<path_to_carla>/CARLA_0.9.15'  # <-- UPDATE THIS PATH

try:
    sys.path.append(glob.glob(f'{CARLA_PATH}/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("ERROR: CARLA Python API not found. Please update CARLA_PATH in this script.")
    print(f"Current CARLA_PATH: {CARLA_PATH}")
    sys.exit(1)

import carla
import cv2
import numpy as np
import math
import random
import torch
from ultralytics import YOLO
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType


class FullNavigationSystemV4:
    """
    Waypoint-based autonomous driving with YOLO for speed control.
    """
    
    def __init__(self, num_vehicles=0, num_pedestrians=0):
        # CARLA setup
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Actor references
        self.ego_vehicle = None
        self.camera = None
        self.rgb_image = None
        self.initial_spawn_point = None
        
        # Traffic generation
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        
        # Check CUDA availability
        self.use_gpu = torch.cuda.is_available()
        if not self.use_gpu:
            print("⚠ CUDA not available, using CPU")
        
        # ==============================================================================
        # Model Paths - Relative to project root (run from src/ directory)
        # ==============================================================================
        YOLO_MODEL_PATH = '../models/yolo_carla/carla_yolo11n_one_map2/weights/best.pt'
        UFLD_MODEL_PATH = '../models/ufld_carla/checkpoint_best.pth'
        
        # YOLO for object detection - Custom trained model
        # Classes: 0:vehicle, 1:pedestrian, 2:traffic_light, 3:speed_limit
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.yolo_classes = {0: 'vehicle', 1: 'pedestrian', 2: 'traffic_light', 3: 'speed_limit'}
        print("✓ YOLO custom model loaded")
        
        # UFLD for lane detection - Fine-tuned on CARLA
        try:
            self.lane_detector = UltrafastLaneDetector(
                UFLD_MODEL_PATH, 
                ModelType.TUSIMPLE, 
                use_gpu=self.use_gpu
            )
            print("✓ UFLD fine-tuned model loaded" + (" (GPU)" if self.use_gpu else " (CPU)"))
        except RuntimeError as e:
            print(f"⚠ GPU error, falling back to CPU: {e}")
            self.lane_detector = UltrafastLaneDetector(
                UFLD_MODEL_PATH, 
                ModelType.TUSIMPLE, 
                use_gpu=False
            )
            print("✓ UFLD fine-tuned model loaded (CPU fallback)")
        
        # Control parameters
        self.MAX_STEER_DEGREES = 70
        self.STEERING_THRESHOLD = 5.0  # Go straight if lane angle > 5°
        self.base_target_speed = 20  # km/h - fixed speed cap
        
        # Safe distances for YOLO detections
        self.safe_distance_vehicle = 15.0  # meters
        self.safe_distance_pedestrian = 10.0  # meters
        self.safe_distance_traffic_light = 20.0  # meters
        
        # Traffic light state
        self.traffic_light_state = "Unknown"
        self.should_stop = False
        
        # Setup environment
        self.setup_environment()

    def setup_environment(self):
        """Setup simulation environment with synchronous mode"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.3  # Slower simulation
        self.world.apply_settings(settings)
        
        # Setup traffic manager for synchronous mode
        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        
        # Clear existing traffic
        self.clear_traffic()
        
        # Setup ego vehicle and sensors
        self.ego_vehicle = self.setup_ego_vehicle()
        self.setup_sensors()
        
        # Generate traffic if requested
        if self.num_vehicles > 0 or self.num_pedestrians > 0:
            self.spawn_traffic()
        
        # Set spectator (main CARLA camera) to follow ego vehicle
        self.set_spectator_follow()
        
        print("✓ Environment setup complete")

    def set_spectator_follow(self):
        """Set the spectator camera to follow the ego vehicle from behind"""
        if self.ego_vehicle is None:
            return
        
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        
        # Position spectator behind and above the vehicle
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=30),  # 30m above
            carla.Rotation(pitch=-90)  # Look straight down
        ))

    def update_spectator(self):
        """Update spectator to follow ego vehicle (call each frame)"""
        if self.ego_vehicle is None:
            return
        
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        
        # Get forward vector and position spectator behind the car
        fwd = transform.get_forward_vector()
        
        # Position: 8m behind, 4m above the car
        spectator_location = transform.location + carla.Location(
            x=-8 * fwd.x,
            y=-8 * fwd.y,
            z=4
        )
        
        spectator.set_transform(carla.Transform(
            spectator_location,
            carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
        ))

    def clear_traffic(self):
        """Remove all NPC vehicles and walkers from the world"""
        actors = self.world.get_actors()
        
        vehicles = actors.filter('vehicle.*')
        for vehicle in vehicles:
            vehicle.destroy()
        
        walkers = actors.filter('walker.*')
        for walker in walkers:
            walker.destroy()
        
        controllers = actors.filter('controller.*')
        for controller in controllers:
            controller.destroy()
        
        print(f"✓ Cleared {len(vehicles)} vehicles and {len(walkers)} walkers")

    def setup_ego_vehicle(self):
        """Spawn the ego vehicle"""
        ego_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        
        spawn_point = random.choice(spawn_points)
        spawn_idx = spawn_points.index(spawn_point)
        print(f"Selected spawn point: #{spawn_idx} at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)
        if ego_vehicle:
            ego_vehicle.set_autopilot(False)
            print("✓ Ego vehicle spawned successfully")
            self.initial_spawn_point = spawn_point
        else:
            print("✗ Failed to spawn ego vehicle")
        return ego_vehicle

    def setup_sensors(self):
        """Setup RGB camera sensor - MUST MATCH UFLD training data"""
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        # Match UFLD training data capture resolution
        cam_bp.set_attribute('image_size_x', '1640')
        cam_bp.set_attribute('image_size_y', '590')
        cam_bp.set_attribute('fov', '150')  # Match zigzag/training FOV
        
        # Camera transform - same as data generator
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.ego_vehicle)
        self.camera.listen(self.rgb_callback)
        print("✓ Camera sensor setup (1640x590, FOV=150°)")

    def spawn_traffic(self):
        """Spawn traffic vehicles and pedestrians"""
        print(f"Spawning traffic: {self.num_vehicles} vehicles, {self.num_pedestrians} pedestrians...")
        
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(True)
        
        # Spawn vehicles
        if self.num_vehicles > 0:
            blueprints = self.blueprint_library.filter('vehicle.*')
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            
            selected_points = spawn_points[:min(self.num_vehicles, len(spawn_points)//4)]
            
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor
            
            batch = []
            for transform in selected_points:
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                blueprint.set_attribute('role_name', 'autopilot')
                
                batch.append(SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            
            for response in self.client.apply_batch_sync(batch, True):
                if not response.error:
                    self.vehicles_list.append(response.actor_id)
            
            print(f"✓ Spawned {len(self.vehicles_list)} vehicles")
        
        # Spawn pedestrians
        if self.num_pedestrians > 0:
            blueprintsWalkers = self.blueprint_library.filter('walker.pedestrian.*')
            
            spawn_points = []
            for i in range(self.num_pedestrians):
                loc = self.world.get_random_location_from_navigation()
                if loc:
                    spawn_points.append(carla.Transform(location=loc))
            
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                if walker_bp.has_attribute('speed'):
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_speed.append(0.0)
                batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
            
            results = self.client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i, result in enumerate(results):
                if not result.error:
                    self.walkers_list.append({"id": result.actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            
            # Spawn controllers
            batch = []
            walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
            for walker in self.walkers_list:
                batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker["id"]))
            
            results = self.client.apply_batch_sync(batch, True)
            for i, result in enumerate(results):
                if not result.error:
                    self.walkers_list[i]["con"] = result.actor_id
            
            for walker in self.walkers_list:
                self.all_id.append(walker.get("con"))
                self.all_id.append(walker["id"])
            
            self.world.tick()
            
            # Start walkers
            all_actors = self.world.get_actors([x for x in self.all_id if x])
            for i in range(0, len(all_actors), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            
            print(f"✓ Spawned {len(self.walkers_list)} pedestrians")

    def rgb_callback(self, image):
        """Callback for RGB camera images"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.rgb_image = array[:, :, :3]

    # ============================================================================
    # YOLO OBJECT DETECTION (from step2_yolo_carla_final.py)
    # ============================================================================
    
    def detect_traffic_light_color(self, img_crop):
        """
        Detect traffic light color using HSV color space.
        Tuned for CARLA's traffic light colors.
        """
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        
        # CARLA traffic lights have specific color ranges
        # Red in CARLA: includes orange-red tones (hue 0-15 and 160-180)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 70, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Yellow/Amber: 16-40
        lower_yellow = np.array([16, 70, 70])
        upper_yellow = np.array([40, 255, 255])
        
        # Green: 40-90 (CARLA green is more cyan-ish)
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([90, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.add(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Count pixels
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Debug: print pixel counts
        total = red_pixels + yellow_pixels + green_pixels
        if total > 0:
            print(f"  TL HSV: R={red_pixels}, Y={yellow_pixels}, G={green_pixels}")
        
        threshold = 5
        
        # Determine color based on highest pixel count
        if red_pixels > threshold and red_pixels >= yellow_pixels and red_pixels >= green_pixels:
            return "Red"
        elif yellow_pixels > threshold and yellow_pixels >= red_pixels and yellow_pixels >= green_pixels:
            return "Yellow"
        elif green_pixels > threshold and green_pixels >= red_pixels and green_pixels >= yellow_pixels:
            return "Green"
        
        return "Unknown"

    def estimate_distance(self, bbox_height, obj_class):
        """Estimate distance to object based on bbox height"""
        if bbox_height < 1:
            return float('inf')
        
        # Approximate focal length for 1640x590 @ 150° FOV
        focal_length = 300
        
        # Real heights by class
        if obj_class == 0:  # vehicle
            real_height = 1.5
        elif obj_class == 1:  # pedestrian
            real_height = 1.7
        elif obj_class == 2:  # traffic_light
            real_height = 0.4
        else:  # speed_limit
            real_height = 0.5
        
        distance = (real_height * focal_length) / bbox_height
        return distance

    def process_yolo_detections(self, rgb_image, debug_image):
        """
        Run YOLO detection and process results for speed control.
        
        Returns:
            should_stop: bool - should we stop for red light or obstacle
            target_speed_modifier: float - speed modifier (0.0-1.0)
            debug_image: annotated image
        """
        # Run YOLO with confidence threshold 0.3
        results = self.yolo_model(rgb_image, verbose=False, conf=0.3)
        
        should_stop = False
        min_vehicle_dist = float('inf')
        min_pedestrian_dist = float('inf')
        closest_traffic_light_color = None
        closest_traffic_light_dist = float('inf')
        
        h, w = rgb_image.shape[:2]
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls not in self.yolo_classes:
                continue
            
            name = self.yolo_classes[cls]
            bbox_height = y2 - y1
            distance = self.estimate_distance(bbox_height, cls)
            
            # Colors for different classes
            if cls == 0:  # vehicle
                color = (255, 0, 0)  # Blue
                if distance < min_vehicle_dist:
                    min_vehicle_dist = distance
            elif cls == 1:  # pedestrian
                color = (0, 0, 255)  # Red
                if distance < min_pedestrian_dist:
                    min_pedestrian_dist = distance
            elif cls == 2:  # traffic_light
                color = (0, 165, 255)  # Orange
                
                # Crop and detect color
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                
                if x2_c > x1_c and y2_c > y1_c:
                    tl_crop = rgb_image[y1_c:y2_c, x1_c:x2_c]
                    tl_color = self.detect_traffic_light_color(tl_crop)
                    
                    if distance < closest_traffic_light_dist:
                        closest_traffic_light_dist = distance
                        closest_traffic_light_color = tl_color
                    
                    # Draw color on image
                    cv2.putText(debug_image, f"{tl_color}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:  # speed_limit
                color = (255, 255, 0)  # Cyan - ignored for now
            
            # Draw bounding box with thick border
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 4)
            
            # Draw label with background
            label = f"{name} {conf:.2f} {distance:.1f}m"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(debug_image, (x1, y1 - 28), (x1 + t_size[0] + 4, y1), color, -1)
            cv2.putText(debug_image, label, (x1 + 2, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Determine if we should stop
        # Stop for red/yellow/unknown traffic lights within range (only green means go)
        if closest_traffic_light_color is not None and closest_traffic_light_dist < self.safe_distance_traffic_light:
            self.traffic_light_state = closest_traffic_light_color
            if closest_traffic_light_color != "Green":
                # Stop for Red, Yellow, or Unknown
                should_stop = True
        
        # Stop for pedestrians that are too close
        if min_pedestrian_dist < self.safe_distance_pedestrian:
            should_stop = True
        
        # Calculate speed modifier based on closest vehicle
        speed_modifier = 1.0
        if min_vehicle_dist < self.safe_distance_vehicle:
            # Slow down proportionally
            speed_modifier = max(0.3, min_vehicle_dist / self.safe_distance_vehicle)
        
        return should_stop, speed_modifier, debug_image

    # ============================================================================
    # UFLD LANE DETECTION (from step3_ufld_v2_worksZigZag.py)
    # ============================================================================
    
    def get_lane_angle(self, lane_points):
        """Calculate the angle of a lane relative to vertical (forward direction)."""
        if len(lane_points) < 2:
            return None
        
        pts = np.array(lane_points)
        bottom_idx = np.argmax(pts[:, 1])
        top_idx = np.argmin(pts[:, 1])
        
        if bottom_idx == top_idx:
            return None
            
        dx = pts[top_idx, 0] - pts[bottom_idx, 0]
        dy = pts[bottom_idx, 1] - pts[top_idx, 1]
        
        if dy < 40:  # Scaled from 20 (800x288) to 40 (1640x590)
            return None
            
        angle = math.degrees(math.atan2(dx, dy))
        return angle

    def get_ego_lane_points(self, lanes_points):
        """Extract ego lane center points with intersection filtering."""
        if not lanes_points or len(lanes_points) < 2:
            return None
        
        IMG_CENTER_X = 820  # Half of 1640
        MAX_LANE_ANGLE = 40
        MAX_ANGLE_DIFF = 25
        MIN_LANE_WIDTH = 200   # Scaled from 100 (800px) to 200 (1640px)
        MAX_LANE_WIDTH = 1600  # Scaled from 800 (800px) to 1600 (1640px)
        
        valid_lanes = []
        for i, lane in enumerate(lanes_points):
            if lane and len(lane) > 0:
                pts = np.array(lane)
                angle = self.get_lane_angle(pts)
                
                if angle is not None and abs(angle) < MAX_LANE_ANGLE:
                    avg_x = np.mean(pts[:, 0])
                    bottom_idx = np.argmax(pts[:, 1])
                    bottom_x = pts[bottom_idx, 0]
                    bottom_y = pts[bottom_idx, 1]
                    valid_lanes.append({
                        'idx': i, 'points': pts, 'angle': angle,
                        'avg_x': avg_x, 'bottom_x': bottom_x, 'bottom_y': bottom_y
                    })
        
        if len(valid_lanes) < 2:
            return None
        
        valid_lanes.sort(key=lambda l: l['avg_x'])
        
        best_pair = None
        best_score = float('inf')
        
        for i in range(len(valid_lanes)):
            for j in range(i + 1, len(valid_lanes)):
                left_lane = valid_lanes[i]
                right_lane = valid_lanes[j]
                
                angle_diff = abs(left_lane['angle'] - right_lane['angle'])
                if angle_diff > MAX_ANGLE_DIFF:
                    continue
                
                lane_width = abs(right_lane['bottom_x'] - left_lane['bottom_x'])
                if lane_width < MIN_LANE_WIDTH or lane_width > MAX_LANE_WIDTH:
                    continue
                
                straddles_center = left_lane['avg_x'] < IMG_CENTER_X < right_lane['avg_x']
                
                center_dist = abs(left_lane['avg_x'] - IMG_CENTER_X) + abs(right_lane['avg_x'] - IMG_CENTER_X)
                score = angle_diff * 3 + center_dist / 50
                if not straddles_center:
                    score += 100
                
                if score < best_score:
                    best_score = score
                    best_pair = (left_lane, right_lane)
        
        if best_pair is None:
            return None
        
        left_center = best_pair[0]['points']
        right_center = best_pair[1]['points']
        
        center_points = []
        min_len = min(len(left_center), len(right_center))
        
        for i in range(min_len):
            center_x = (left_center[i][0] + right_center[i][0]) / 2
            center_y = (left_center[i][1] + right_center[i][1]) / 2
            center_points.append([center_x, center_y])
        
        return np.array(center_points)

    def get_steering_from_lane_center(self, ego_lane, img_width=1640, img_height=590):
        """Calculate steering angle from the lane center line."""
        if ego_lane is None or len(ego_lane) < 2:
            return None, 0.0
        
        IMG_CENTER_X = img_width / 2
        
        bottom_idx = np.argmax(ego_lane[:, 1])
        bottom_point = ego_lane[bottom_idx]
        
        sorted_by_y = ego_lane[np.argsort(ego_lane[:, 1])[::-1]]
        
        if len(sorted_by_y) < 3:
            return None, 0.0
        
        lookahead_idx = len(sorted_by_y) // 3
        lookahead_point = sorted_by_y[lookahead_idx]
        
        lateral_offset = bottom_point[0] - IMG_CENTER_X
        
        dx = lookahead_point[0] - bottom_point[0]
        dy = bottom_point[1] - lookahead_point[1]
        
        if dy < 20:  # Scaled from 10 (800x288) to 20 (1640x590)
            return None, 0.0
        
        lane_angle = math.degrees(math.atan2(dx, dy))
        
        offset_correction = (lateral_offset / IMG_CENTER_X) * 15
        
        steering_angle = lane_angle + offset_correction
        
        confidence = min(1.0, len(ego_lane) / 40.0)  # Scaled from 20 to 40
        
        return steering_angle, confidence

    def smooth_lane(self, points, degree=2):
        """Fit polynomial for smoother visualization"""
        if len(points) < 4:
            return points
        pts = np.array(points)
        try:
            coeffs = np.polyfit(pts[:, 1], pts[:, 0], degree)
            y_vals = pts[:, 1]
            x_smooth = np.polyval(coeffs, y_vals)
            return np.column_stack([x_smooth, y_vals]).astype(int)
        except:
            return pts

    def scale_lane_points(self, points, from_size=(800, 288), to_size=(1640, 590)):
        """
        Scale lane points from model output resolution to camera resolution.
        UFLD model outputs at 800x288, camera captures at 1640x590.
        """
        if len(points) == 0:
            return points
        
        pts = np.array(points, dtype=float)
        scale_x = to_size[0] / from_size[0]  # 1640/800 = 2.05
        scale_y = to_size[1] / from_size[1]  # 590/288 = 2.0486
        
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        
        return pts

    def process_ufld_lanes(self, rgb_image, debug_image):
        """
        Process UFLD lane detection.
        Returns center_line for steering calculation (in 1640x590 coordinates).
        """
        # Detect lanes - returns 800x288 visualization and lane points in 800x288 coords
        lane_visualization = self.lane_detector.detect_lanes(rgb_image)
        lanes_points = self.lane_detector.lanes_points
        
        center_line = None
        
        # Model output resolution and camera resolution
        MODEL_W, MODEL_H = 800, 288
        CAM_W, CAM_H = 1640, 590
        
        if lanes_points is not None:
            valid_lane_points = []
            for i, lane in enumerate(lanes_points):
                if lane and len(lane) > 0:
                    # Scale points from model coords (800x288) to camera coords (1640x590)
                    points = self.scale_lane_points(lane, (MODEL_W, MODEL_H), (CAM_W, CAM_H))
                    if points.size > 0:
                        valid_lane_points.append(points)
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                        color = colors[i % len(colors)]
                        
                        smooth_pts = self.smooth_lane(points).astype(int)
                        cv2.polylines(debug_image, [smooth_pts], False, color, 3, cv2.LINE_AA)
            
            # Draw yellow center line between the first two valid detected lanes
            if len(valid_lane_points) >= 2:
                lane1 = valid_lane_points[0]
                lane2 = valid_lane_points[1]
                
                center_points = []
                for pt1 in lane1:
                    y_diffs = np.abs(lane2[:, 1] - pt1[1])
                    closest_idx = np.argmin(y_diffs)
                    if y_diffs[closest_idx] < 40:  # Increased threshold for scaled coords
                        pt2 = lane2[closest_idx]
                        center_x = (pt1[0] + pt2[0]) / 2
                        center_y = (pt1[1] + pt2[1]) / 2
                        center_points.append([center_x, center_y])
                
                if len(center_points) > 1:
                    center_pts = np.array(center_points).astype(int)
                    center_pts = center_pts[np.argsort(center_pts[:, 1])]
                    smooth_center = self.smooth_lane(center_pts)
                    cv2.polylines(debug_image, [smooth_center], False, (0, 255, 255), 4, cv2.LINE_AA)
                    center_line = np.array(center_points)
        
        return center_line, debug_image

    # ============================================================================
    # WAYPOINT NAVIGATION (from step3_ufld_v2_worksZigZag.py)
    # ============================================================================
    
    def get_waypoints(self, spawn_point):
        """Get a list of waypoints from the map"""
        current_waypoint = self.world.get_map().get_waypoint(spawn_point.location)
        
        waypoints = []
        waypoints.append(current_waypoint)
        
        start_location = current_waypoint.transform.location
        
        while len(waypoints) < 50:
            next_waypoints = current_waypoint.next(2.0)
            if not next_waypoints:
                break
                
            current_waypoint = next_waypoints[0]
            waypoints.append(current_waypoint)
            
            if (len(waypoints) > 10 and 
                current_waypoint.transform.location.distance(start_location) < 5.0):
                print("Route loops back to start")
                break
        
        print(f"Generated {len(waypoints)} waypoints")
        return waypoints

    def get_angle(self, vehicle, waypoint):
        """Calculate angle between car's heading and waypoint"""
        vehicle_transform = vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_fwd = vehicle_transform.get_forward_vector()

        wp_loc = waypoint.transform.location
        
        direction = np.array([
            wp_loc.x - vehicle_loc.x,
            wp_loc.y - vehicle_loc.y
        ])
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        fwd = np.array([vehicle_fwd.x, vehicle_fwd.y])
        
        dot_product = np.dot(fwd, direction)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle = np.degrees(angle)
        
        cross_product = np.cross([vehicle_fwd.x, vehicle_fwd.y], direction)
        if cross_product < 0:
            angle = -angle
            
        return angle

    def maintain_speed(self, current_speed, steering_angle_abs, speed_modifier=1.0):
        """
        Maintain speed with turn-aware adjustment.
        Speed capped at 20 km/h, reduced during turns.
        """
        TURN_THRESHOLD = 10.0
        
        if steering_angle_abs > TURN_THRESHOLD:
            target_speed = self.base_target_speed * 0.7
        else:
            target_speed = self.base_target_speed
        
        # Apply YOLO-based speed modifier
        target_speed *= speed_modifier
        
        diff = target_speed - current_speed
    
        if abs(diff) < 1.0:
            throttle = 0.5
            brake = 0.0
        elif diff > 0:
            throttle = min(0.75, 0.5 + diff * 0.1)
            brake = 0.0
        else:
            if abs(diff) > 5.0:
                throttle = 0.0
                brake = min(0.5, abs(diff) * 0.05)
            else:
                throttle = max(0.25, 0.5 + diff * 0.1)
                brake = 0.0
        
        return throttle, brake

    def apply_control(self, throttle, steer, brake):
        """Apply vehicle control"""
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        )
        self.ego_vehicle.apply_control(control)

    # ============================================================================
    # MAIN CONTROL LOOP
    # ============================================================================
    
    def follow_waypoints(self):
        """Main control loop: Follow waypoints with UFLD lane detection and YOLO speed control"""
        waypoints = self.get_waypoints(self.initial_spawn_point)
        curr_wp = 1
        
        cv2.namedWindow('CARLA View', cv2.WINDOW_AUTOSIZE)
        print(f"Starting navigation with {len(waypoints)} waypoints")
        print("Using UFLD for steering, YOLO for speed control")
        print(f"Speed cap: {self.base_target_speed} km/h")
        print(f"Steering threshold: {self.STEERING_THRESHOLD}° (go straight if lane angle > threshold)")
        
        while True:
            self.world.tick()
            
            # Update spectator camera to follow ego vehicle
            self.update_spectator()
            
            rgb_data = self.rgb_image
            if rgb_data is None:
                continue
            
            # Create debug visualization
            debug_image = rgb_data.copy()
            
            # === YOLO DETECTION ===
            should_stop, speed_modifier, debug_image = self.process_yolo_detections(rgb_data, debug_image)
            
            # === UFLD LANE DETECTION ===
            center_line, debug_image = self.process_ufld_lanes(rgb_data, debug_image)

            # Get current vehicle state
            current_transform = self.ego_vehicle.get_transform()
            current_location = current_transform.location
            velocity = self.ego_vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Update waypoint if reached
            if curr_wp < len(waypoints):
                if current_location.distance(waypoints[curr_wp].transform.location) < 2.0:
                    curr_wp += 1
                    print(f"Reached waypoint {curr_wp}/{len(waypoints)}")
            
            # Check if route completed
            if curr_wp >= len(waypoints):
                print("="*60)
                print("Route completed successfully!")
                print("="*60)
                break
            
            # === STEERING CALCULATION ===
            # Use lane center steering, fallback to waypoint if steer > 5 degrees
            lane_steer, lane_confidence = self.get_steering_from_lane_center(center_line)
            waypoint_angle = self.get_angle(self.ego_vehicle, waypoints[curr_wp])
            
            if lane_steer is not None and lane_confidence > 0.3 and abs(lane_steer) <= self.STEERING_THRESHOLD:
                # Use lane-based steering (following yellow center line)
                predicted_angle = lane_steer
                steering_source = "LANE"
            else:
                # Fallback to waypoint steering (large turns or no lane detected)
                predicted_angle = waypoint_angle
                steering_source = "WAYPOINT"
            
            steer_input = predicted_angle
            
            # Apply steering limits
            if predicted_angle < -self.MAX_STEER_DEGREES:
                steer_input = -self.MAX_STEER_DEGREES
            elif predicted_angle > self.MAX_STEER_DEGREES:
                steer_input = self.MAX_STEER_DEGREES
            
            steer_input = steer_input / 75  # Normalize to [-1, 1]
            
            # === SPEED CONTROL ===
            if should_stop:
                throttle = 0.0
                brake = 1.0  # Full brake for stopping
            else:
                throttle, brake = self.maintain_speed(speed, abs(predicted_angle), speed_modifier)
            
            # Apply control
            self.apply_control(throttle, steer_input, brake)
            
            # === VISUALIZATION ===
            # Resize to 800x500 for display
            display_image = cv2.resize(debug_image, (800, 500))
            
            cv2.imshow('CARLA View', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        print("\nCleaning up...")
        self.cleanup()

    def cleanup(self):
        """Destroy all actors and cleanup resources"""
        try:
            # Stop the vehicle
            if self.ego_vehicle:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = 1.0
                self.ego_vehicle.apply_control(control)
            
            # Destroy camera sensor
            if self.camera:
                print("Destroying camera...")
                self.camera.stop()
                self.camera.destroy()
                self.camera = None
            
            # Destroy spawned traffic
            print("Destroying traffic...")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
            
            for i in range(0, len(self.all_id), 2):
                self.client.apply_batch([carla.command.DestroyActor(self.all_id[i])])
                self.client.apply_batch([carla.command.DestroyActor(self.all_id[i + 1])] if i + 1 < len(self.all_id) else [])
            
            # Destroy ego vehicle
            if self.ego_vehicle:
                print("Destroying ego vehicle...")
                self.ego_vehicle.destroy()
                self.ego_vehicle = None
            
            # Disable synchronous mode
            print("Disabling synchronous mode...")
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            cv2.destroyAllWindows()
            
            print("Cleanup complete. CARLA server still running.")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='Full Autonomous Driving System v4')
    parser.add_argument('--vehicles', '--vehicle', type=int, default=0, 
                        help='Number of NPC vehicles to spawn (default: 0)')
    parser.add_argument('--pedestrians', '--pedestrian', type=int, default=0,
                        help='Number of pedestrians to spawn (default: 0)')
    args = parser.parse_args()
    
    print("="*60)
    print("Full Autonomous Driving System v4")
    print("Waypoint navigation + UFLD lanes + YOLO detection")
    print("="*60)
    print("Controls:")
    print("  Press 'q' to quit")
    print(f"Traffic: {args.vehicles} vehicles, {args.pedestrians} pedestrians")
    print("="*60)
    
    # Create system with traffic
    navigation_system = FullNavigationSystemV4(num_vehicles=args.vehicles, num_pedestrians=args.pedestrians)
    navigation_system.follow_waypoints()


if __name__ == "__main__":
    main()
