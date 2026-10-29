import cv2
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

# Original TuSimple row anchors (for pre-trained model)
tusimple_row_anchor_original = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]

# Custom row anchors matching training (bottom 60% of 288 height image)
# np.linspace(int(288*0.4), 288-1, 56) = linspace(115, 287, 56)
tusimple_row_anchor = [int(x) for x in np.linspace(115, 287, 56)]

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 800  # Match training resolution
		self.img_h = 288  # Match training resolution
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Determine backbone based on model configuration
		# TuSimple uses ResNet18, CULane uses ResNet34
		backbone = '18' if cfg.griding_num == 100 else '34'
		
		# Load the model architecture - MUST include size parameter to match training!
		net = parsingNet(pretrained=False, backbone=backbone, 
						size=(cfg.img_h, cfg.img_w),  # Critical: must match training
						cls_dim=(cfg.griding_num+1, cfg.cls_num_per_lane, 4),
						use_aux=False)


		# Load the weights from the downloaded model
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				checkpoint = torch.load(model_path, map_location='mps', weights_only=False)
			else:
				net = net.cuda()
				checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
		else:
			checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
		
		# Handle different checkpoint formats
		if 'model_state_dict' in checkpoint:
			state_dict = checkpoint['model_state_dict']  # Custom trained model format
		elif 'model' in checkpoint:
			state_dict = checkpoint['model']  # Original UFLD format
		elif 'state_dict' in checkpoint:
			state_dict = checkpoint['state_dict']  # Alternative format
		else:
			state_dict = checkpoint  # Direct state dict

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transform operation to normalize the input images
		# MUST match training: ToPILImage -> ToTensor -> Normalize
		img_transforms = transforms.Compose([
			transforms.ToPILImage(),  # Convert numpy to PIL (training does this)
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def prepare_input(self, img):
		# Transform the image for inference - EXACTLY like training notebook
		# Training: cv2.resize -> ToPILImage -> ToTensor -> Normalize
		
		# 1. Convert BGR to RGB (cv2 loads as BGR)
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		# 2. Resize using cv2 (same as training dataset)
		img_resized = cv2.resize(img_rgb, (self.cfg.img_w, self.cfg.img_h))  # (800, 288)
		
		# 3. Apply transforms (ToTensor + Normalize)
		input_img = self.img_transform(img_resized)
		input_tensor = input_img[None, ...]

		if self.use_gpu:
			if not torch.backends.mps.is_built():
				input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.model(input_tensor)

		return output

	@staticmethod
	def process_output(output, cfg):		
		"""Process network output - matches training notebook exactly"""
		# output shape: (1, 101, 56, 4) -> (griding_num+1, cls_num_per_lane, num_lanes)
		out = output[0].data.cpu().numpy()  # (101, 56, 4)
		
		lanes_points = []
		lanes_detected = []
		
		# Process each lane (same loop order as notebook)
		for lane_idx in range(out.shape[2]):  # 4 lanes
			lane_points = []
			valid_points = 0
			
			for row_idx in range(cfg.cls_num_per_lane):  # 56 row anchors
				probs = out[:, row_idx, lane_idx]  # (101,) - access exactly like notebook
				x_pred = np.argmax(probs)
				
				# Check if valid lane point (not "no lane" class) - no confidence threshold like notebook
				if x_pred < cfg.griding_num:
					# Convert grid to pixel coordinates - exactly like notebook
					x_pixel = int(x_pred * cfg.img_w / cfg.griding_num)
					y_pixel = cfg.row_anchor[row_idx]
					lane_points.append([x_pixel, y_pixel])
					valid_points += 1
			
			lanes_detected.append(valid_points > 2)
			lanes_points.append(lane_points)
		
		return lanes_points, lanes_detected

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

		# Draw a mask for the current lane
		if(lanes_detected[1] and lanes_detected[2]):
			lane_segment_img = visualization_img.copy()
			
			cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

		if(draw_points):
			for lane_num,lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

		return visualization_img


	







