# Copyright (c) ...
# Create by ... <...@...>

import os
import cv2
import yaml
import copy
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
from imantics import Polygons, Mask
from tqdm import tqdm

from enum import Enum
import sys
import warnings

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def mask_to_poly(mask):
    h, w = mask.shape[-2:]

    polygons = Mask(mask.reshape(h, w, 1)).polygons()
    

    #use the polygon with most points
    l = len(polygons.points[0])
    point_out = polygons.points[0]
    for point in polygons.points:
        if len(point) > l:
            l = len(point)
            point_out = point
    
    return point_out

def poly_to_bbox(poly):
    x0 = np.inf
    y0 = np.inf
    x1 = -1
    y1 = -1
    for point in poly:
        x0 = min(x0, point[0])
        y0 = min(y0, point[1])
        x1 = max(x1, point[0])
        y1 = max(y1, point[1])
    return [x0, y0, x1, y1]

def shape_points_to_bbox(qpoints):
    x0 = np.inf
    y0 = np.inf
    x1 = -1
    y1 = -1
    for point in qpoints:
        x0 = min(x0, point.x())
        y0 = min(y0, point.y())
        x1 = max(x1, point.x())
        y1 = max(y1, point.y())
    return [x0, y0, x1, y1]

def bbox_to_shape_points(bbox):
    x0, y0, x1, y1 = bbox
    return [QPointF(x0, y0), QPointF(x1, y0), QPointF(x1, y1), QPointF(x0, y1)]

VIT_B_PATH= "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
#VIT_L_PATH = "models/SAM/sam_vit_l_0b3195.pth"
VIT_L_PATH = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
#VIT_H_PATH = "models/SAM/sam_vit_h_4b8939.pth"
VIT_H_PATH = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

class VIT_STATE(Enum):
    OFF = 0
    VIT_B = 1
    VIT_L = 2
    VIT_H = 3


class SAM(object):
    def __init__(self):
        self.sam_model = None
        self.sam_predictor = None
        self.image_path = None
        self.image = None
        self.state = None
        self.current_state = None
        self.new_image = False
        self.new_model = False
        self.default_device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
        self.device = None

        #pre download models
        torch.hub.load_state_dict_from_url(VIT_H_PATH)
        torch.hub.load_state_dict_from_url(VIT_L_PATH)
        torch.hub.load_state_dict_from_url(VIT_B_PATH)

    def change_sam_model(self, state, device = None):
        #request to change model
        self.state = state
        if device is None:
            device = self.default_device
        self.device = device
        self.new_model = True

    def set_sam_model(self, force=False):
        #set model
        self.new_model = False
        if not force and self.current_state == self.state:#no change of state, return
            return
        self.current_state = self.state
        del self.sam_predictor
        del self.sam_model
        if self.state == VIT_STATE.OFF:
            self.sam_model = None
            self.sam_predictor = None
            return
        try:
            if self.state == VIT_STATE.VIT_B:
                #init sam model
                self.sam_model = sam_model_registry["vit_b"](checkpoint=VIT_B_PATH)
            elif self.state == VIT_STATE.VIT_L:
                #init sam model
                self.sam_model = sam_model_registry["vit_l"](checkpoint=VIT_L_PATH)
            elif self.state == VIT_STATE.VIT_H:
                #init sam model
                self.sam_model = sam_model_registry["vit_h"](checkpoint=VIT_H_PATH)
            self.sam_model.eval()
            self.sam_model.to(self.device)
            self.sam_predictor = SamPredictor(self.sam_model)
            self.new_image = True #force image reload
        except torch.cuda.OutOfMemoryError:#out of memory, switch to cpu(may not want to use cpu, just anti crash)
            warnings.warn("GPU out of memory, switching to cpu!!!")
            self.change_sam_model(self.state, "cpu")
            self.set_sam_model(force=True)
        except Exception as e:
            warnings.warn("Error loading model: {}".format(e))
            self.sam_model = None
            self.sam_predictor = None

    def load_image_path(self, path):
        #request to load image
        self.image_path = path
        self.new_image = True
    
    def set_image(self):
        #set image
        self.image = cv2.imread(self.image_path)
        self.new_image = False
        if self.sam_predictor:
            try:
                self.sam_predictor.set_image(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            except torch.cuda.OutOfMemoryError:#out of memory, switch to cpu(may not want to use cpu, just anti crash)
                warnings.warn("GPU out of memory, switching to cpu!!!")
                self.change_sam_model(self.state, "cpu")
                self.set_sam_model(force=True)
                self.set_image()
            except Exception as e:
                warnings.warn("Error setting image: {}".format(e))
                self.image = None
                self.change_sam_model(VIT_STATE.OFF)
                self.set_sam_model(force=True)
    
    def predict(self, shape_points):
        if self.new_model:#model has changed, reload
            self.set_sam_model(force=True)
        if self.sam_predictor:
            if self.new_image:#image has changed, reload
                self.set_image()
            try:
                input_boxes = torch.tensor(np.array(shape_points_to_bbox(shape_points))[None], device="cpu")
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, self.image.shape[:2]).to(self.sam_predictor.device)
                masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            except torch.cuda.OutOfMemoryError:#out of memory, switch to cpu(may not want to use cpu, just anti crash)
                warnings.warn("GPU out of memory, switching to cpu!!!")
                self.device = "cpu"
                self.change_sam_model(self.state, "cpu")
                self.set_sam_model(force=True)
                self.set_image()
                return self.predict(shape_points)
            except Exception as e:
                warnings.warn("Error predicting: {}".format(e))
                if self.device == "cpu":
                    print("Trying again with cpu")
                    self.device = "cpu"
                    self.change_sam_model(self.state, "cpu")
                    self.set_sam_model(force=True)
                    self.set_image()
                    return self.predict(shape_points)
                return shape_points
            polys = [mask_to_poly(mask.cpu().numpy()) for mask in masks]#convert mask to polygon
            bboxes = [poly_to_bbox(poly) for poly in polys]#convert polygon to bbox
            bboxes = [[max(bbox[0]-1, 0), max(bbox[1]-1, 0), min(bbox[2]+1, self.image.shape[1]-1), min(bbox[3]+1, self.image.shape[0]-1)] for bbox in bboxes]#fix accuracy issue
            pointss = [bbox_to_shape_points(bbox) for bbox in bboxes]#convert bbox to points
            return pointss[0]
        return shape_points