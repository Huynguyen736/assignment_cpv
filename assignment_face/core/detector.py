from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from assignment_face.core.preprocess import to_grayscale


class HaarCascade:
    def __init__(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        cascade = root.find('cascade')
        
        self.win_w = int(cascade.find('width').text)
        self.win_h = int(cascade.find('height').text)
        self.win_area = self.win_w * self.win_h
        
        stages_node = cascade.find('stages')
        self.stages = []
        for stage in stages_node.findall('_'):
            stage_threshold = float(stage.find('stageThreshold').text)
            weaks = []
            for weak in stage.find('weakClassifiers').findall('_'):
                internal = weak.find('internalNodes').text.strip().split()
                feature_idx = int(internal[2])
                threshold = float(internal[3])
                
                leafs = weak.find('leafValues').text.strip().split()
                left_val = float(leafs[0])
                right_val = float(leafs[1])
                
                weaks.append((feature_idx, threshold, left_val, right_val))
            self.stages.append((stage_threshold, weaks))
            
        features_node = cascade.find('features')
        self.features = []
        for feat in features_node.findall('_'):
            rects = []
            for rect in feat.find('rects').findall('_'):
                r = list(map(float, rect.text.strip().split()))
                rects.append(r)
            self.features.append(rects)

    def compute_integral_images(self, img):
        img_float = img.astype(np.float64)
        integral = img_float
        sq_integral = img_float ** 2
        return integral, sq_integral

    def get_region_sum(self, img_array, x, y, w, h):
        x_start = int(x)
        y_start = int(y)
        x_end = int(x + w)
        y_end = int(y + h)
        sum_val = 0.0
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                sum_val += img_array[i, j]    
        return sum_val

    def detect_multi_scale(self, img, scale_factor=1.2, step=4):
        h_img, w_img = img.shape
        detections = []
        scale = 1.0
        while True:
            resized_w = int(w_img / scale)
            resized_h = int(h_img / scale)
            
            if resized_w < self.win_w or resized_h < self.win_h:
                break
                
            resized_img = cv2.resize(img, (resized_w, resized_h))
            integral, sq_integral = self.compute_integral_images(resized_img)
            
            for y in range(0, resized_h - self.win_h, step):
                for x in range(0, resized_w - self.win_w, step):
                    win_sum = self.get_region_sum(integral, x, y, self.win_w, self.win_h)
                    win_sq_sum = self.get_region_sum(sq_integral, x, y, self.win_w, self.win_h)
                    mean = win_sum / self.win_area
                    variance = (win_sq_sum / self.win_area) - (mean ** 2)
                    std_dev = np.sqrt(variance) if variance > 0 else 1.0             
                    passed_all_stages = True
                    for stage_threshold, weaks in self.stages:
                        stage_sum = 0.0
                        for feature_idx, weak_thresh, left_val, right_val in weaks:
                            rects = self.features[feature_idx]
                            feat_val = 0.0
                            for rx, ry, rw, rh, weight in rects:
                                r_sum = self.get_region_sum(integral, x + rx, y + ry, rw, rh)
                                feat_val += r_sum * weight
                                
                            feat_val = feat_val / self.win_area
                            if feat_val < weak_thresh * std_dev:
                                stage_sum += left_val
                            else:
                                stage_sum += right_val
                                
                        if stage_sum < stage_threshold:
                            passed_all_stages = False
                            break
                            
                    if passed_all_stages:
                        orig_x = int(x * scale)
                        orig_y = int(y * scale)
                        orig_w = int(self.win_w * scale)
                        orig_h = int(self.win_h * scale)
                        detections.append([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h])
                        
            scale *= scale_factor
            
        return detections


class FaceDetector:
    def __init__(self, cascade_path, face_size):
        self.cascade = HaarCascade(str(cascade_path))
        self.face_size = face_size

    def detect(self, frame):
        gray = to_grayscale(frame)
        
        raw_detections = self.cascade.detect_multi_scale(gray, scale_factor=1.1, step=4)
        
        raw_boxes = []
        for det in raw_detections:
            x1, y1, x2, y2 = det
            w = x2 - x1
            h = y2 - y1
            if w >= 60 and h >= 60:
                raw_boxes.append([x1, y1, w, h])
                
        boxes = []
        if len(raw_boxes) > 0:
            grouped_boxes, weights = cv2.groupRectangles(raw_boxes, groupThreshold=1, eps=0.2)
            for bbox in grouped_boxes:
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                boxes.append((x, y, w, h))

        faces = []
        for x, y, w, h in boxes:
            face_crop = gray[y : y + h, x : x + w]
            face_img = cv2.resize(face_crop, self.face_size)
            faces.append(face_img)
            
        return faces, boxes