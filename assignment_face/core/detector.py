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

FACE_CROP_EXPANSION = 0.15
BOX_DISPLAY_EXPANSION = 0.4


class AlignedFace:
    def __init__(self, image, transform_matrix):
        self.image = image
        self.transform_matrix = transform_matrix

    def transform_landmarks(self, landmarks):
        points = np.asarray(landmarks, dtype=np.float32)
        homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float32)])
        return np.dot(homogeneous, self.transform_matrix.T)


def expand_bounding_box(
    box,
    image_shape,
    expansion=0.4,
):
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
    face_crop,
    landmarks,
    output_size,
):
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


def _cascade_path(name):
    return str(Path(__file__).parent.parent / "models" / name)


def _load_cascade(name):
    return cv2.CascadeClassifier(_cascade_path(name))


def _largest_box(boxes, x_offset=0, y_offset=0):
    if boxes is None or len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
    return (int(x) + x_offset, int(y) + y_offset, int(w), int(h))


def _box_center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _detect_eye_landmarks(
    gray_crop,
    eye_cascade,
):
    crop_h, crop_w = gray_crop.shape[:2]
    upper_face = gray_crop[: max(1, int(crop_h * 0.65)), :]
    raw_eyes = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1)
    eyes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw_eyes]
    eye_boxes = sorted(eyes, key=lambda box: int(box[2]) * int(box[3]), reverse=True)

    candidates = []
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
    gray_crop,
    smile_cascade,
):
    crop_h = gray_crop.shape[0]
    lower_start = int(crop_h * 0.48)
    lower_face = gray_crop[lower_start:, :]
    raw_smiles = smile_cascade.detectMultiScale(lower_face, scaleFactor=1.2)
    smiles = [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw_smiles]
    mouth_box = _largest_box(smiles, y_offset=lower_start)
    if mouth_box is None:
        return None

    x, y, w, h = mouth_box
    mouth_y = y + h * 0.58
    return (x + w * 0.15, mouth_y), (x + w * 0.85, mouth_y)


def detect_five_landmarks(gray_crop, eye_cascade, smile_cascade):
    if eye_cascade is None or smile_cascade is None:
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


class FaceDetector:
    def __init__(self, cascade_path, face_size):
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        self.face_size = face_size
        self.eye_cascade = _load_cascade("haarcascade_eye_tree_eyeglasses.xml")
        self.smile_cascade = _load_cascade("haarcascade_smile.xml")

    def detect(self, frame):
        gray = to_grayscale(frame)
        raw_boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1)
        boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw_boxes]

        faces = []
        for x, y, w, h in boxes:
            expanded_x, expanded_y, expanded_w, expanded_h = expand_bounding_box(
                (x, y, w, h),
                gray.shape,
                expansion=FACE_CROP_EXPANSION,
            )
            face_crop = gray[expanded_y : expanded_y + expanded_h, expanded_x : expanded_x + expanded_w]
            landmarks = detect_five_landmarks(face_crop, self.eye_cascade, self.smile_cascade)
            if landmarks is not None:
                face_img = align_face_with_landmarks(face_crop, landmarks, output_size=self.face_size).image
            else:
                face_img = cv2.resize(face_crop, self.face_size)
            faces.append(face_img)
            
        expanded_boxes = [expand_bounding_box(box, gray.shape, expansion=BOX_DISPLAY_EXPANSION) for box in boxes]
        return faces, expanded_boxes
