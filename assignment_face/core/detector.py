from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from assignment_face.core.preprocess import to_grayscale

REFERENCE_FACE_LANDMARKS = np.array(
    [
        [0.32, 0.36],  # left eye
        [0.68, 0.36],  # right eye
        [0.50, 0.55],  # nose
        [0.38, 0.75],  # left mouth corner
        [0.62, 0.75],  # right mouth corner
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AlignedFace:
    image: np.ndarray
    transform_matrix: np.ndarray

    def transform_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        points = np.asarray(landmarks, dtype=np.float32)
        homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float32)])
        return homogeneous @ self.transform_matrix.T


def expand_bounding_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    expansion: float = 0.4,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    image_h, image_w = image_shape[:2]
    pad_x = int(round(w * expansion / 2.0))
    pad_y = int(round(h * expansion / 2.0))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_w, x + w + pad_x)
    y2 = min(image_h, y + h + pad_y)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def align_face_with_landmarks(
    face_crop: np.ndarray,
    landmarks: np.ndarray,
    output_size: tuple[int, int],
) -> AlignedFace:
    output_w, output_h = output_size
    source_landmarks = np.asarray(landmarks, dtype=np.float32)
    target_landmarks = REFERENCE_FACE_LANDMARKS * np.array([output_w, output_h], dtype=np.float32)
    transform_matrix, _ = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks, method=cv2.LMEDS)
    if transform_matrix is None:
        transform_matrix = cv2.getAffineTransform(source_landmarks[:3], target_landmarks[:3])

    aligned = cv2.warpAffine(
        face_crop,
        transform_matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return AlignedFace(image=aligned, transform_matrix=transform_matrix.astype(np.float32))


def _cascade_path(name: str) -> str:
    return str(Path(cv2.data.haarcascades) / name)


def _load_cv2_cascade(name: str) -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(_cascade_path(name))


def _largest_box(boxes: np.ndarray | tuple, x_offset: int = 0, y_offset: int = 0) -> tuple[int, int, int, int] | None:
    if boxes is None or len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
    return (int(x) + x_offset, int(y) + y_offset, int(w), int(h))


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _detect_eye_landmarks(
    gray_crop: np.ndarray,
    eye_cascade: cv2.CascadeClassifier,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    crop_h, crop_w = gray_crop.shape[:2]
    upper_face = gray_crop[: max(1, int(crop_h * 0.65)), :]
    eyes = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))
    eye_boxes = sorted(eyes, key=lambda box: int(box[2]) * int(box[3]), reverse=True)

    candidates: list[tuple[float, float]] = []
    for box in eye_boxes:
        center_x, center_y = _box_center(tuple(int(value) for value in box))
        if 0.12 * crop_w <= center_x <= 0.88 * crop_w:
            candidates.append((center_x, center_y))

    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            horizontal_gap = abs(second[0] - first[0])
            vertical_gap = abs(second[1] - first[1])
            if horizontal_gap >= crop_w * 0.18 and vertical_gap <= crop_h * 0.16:
                left_eye, right_eye = sorted([first, second], key=lambda point: point[0])
                return left_eye, right_eye
    return None


def _detect_mouth_landmarks(
    gray_crop: np.ndarray,
    smile_cascade: cv2.CascadeClassifier,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    crop_h = gray_crop.shape[0]
    lower_start = int(crop_h * 0.48)
    lower_face = gray_crop[lower_start:, :]
    smiles = smile_cascade.detectMultiScale(lower_face, scaleFactor=1.2, minNeighbors=12, minSize=(18, 8))
    mouth_box = _largest_box(smiles, y_offset=lower_start)
    if mouth_box is None:
        return None

    x, y, w, h = mouth_box
    mouth_y = y + h * 0.58
    return (x + w * 0.15, mouth_y), (x + w * 0.85, mouth_y)


def detect_five_landmarks(
    gray_crop: np.ndarray,
    eye_cascade: cv2.CascadeClassifier,
    smile_cascade: cv2.CascadeClassifier,
) -> np.ndarray | None:
    if eye_cascade.empty() or smile_cascade.empty():
        return None

    eyes = _detect_eye_landmarks(gray_crop, eye_cascade)
    if eyes is None:
        return None

    crop_h, crop_w = gray_crop.shape[:2]
    left_eye, right_eye = eyes
    mouth = _detect_mouth_landmarks(gray_crop, smile_cascade)
    if mouth is None:
        mouth_y = max(left_eye[1], right_eye[1]) + crop_h * 0.36
        mouth_half_width = abs(right_eye[0] - left_eye[0]) * 0.34
        mouth_center_x = (left_eye[0] + right_eye[0]) / 2.0
        mouth = (
            (mouth_center_x - mouth_half_width, mouth_y),
            (mouth_center_x + mouth_half_width, mouth_y),
        )

    mouth_left, mouth_right = mouth
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
    nose = (eye_center_x, eye_center_y + (mouth_center_y - eye_center_y) * 0.52)

    landmarks = np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32)
    landmarks[:, 0] = np.clip(landmarks[:, 0], 0, crop_w - 1)
    landmarks[:, 1] = np.clip(landmarks[:, 1], 0, crop_h - 1)
    return landmarks


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
        self.eye_cascade = _load_cv2_cascade("haarcascade_eye_tree_eyeglasses.xml")
        self.smile_cascade = _load_cv2_cascade("haarcascade_smile.xml")

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
            expanded_x, expanded_y, expanded_w, expanded_h = expand_bounding_box((x, y, w, h), gray.shape, expansion=0.4)
            face_crop = gray[expanded_y : expanded_y + expanded_h, expanded_x : expanded_x + expanded_w]
            landmarks = detect_five_landmarks(face_crop, self.eye_cascade, self.smile_cascade)
            if landmarks is not None:
                face_img = align_face_with_landmarks(face_crop, landmarks, output_size=self.face_size).image
            else:
                face_img = cv2.resize(face_crop, self.face_size)
            faces.append(face_img)
            
        expanded_boxes = [expand_bounding_box(box, gray.shape, expansion=0.4) for box in boxes]
        return faces, expanded_boxes
