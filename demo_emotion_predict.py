"""Realtime emotion prediction with the EfficientNet-B0 + landmark fusion model.

Run for example:
python demo_emotion_fusion.py --checkpoint models/fusion_model_partial.pt
"""

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

from Train_model.train_fusion import (
    DEVICE,
    NUM_LANDMARKS,
    FusionModel,
    align_image,
    align_landmarks,
    build_transforms,
    compute_roll_angles_and_centers,
    normalize_landmarks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict emotion from webcam in real time.")
    parser.add_argument(
        "--checkpoint",
        default="models/fusion_model_partial.pt",
        help="Path to the .pt checkpoint saved by train_fusion.py (default: models/fusion_model_partial.pt)",
    )
    parser.add_argument(
        "--labels", required=False,
        help="Comma-separated class names in exactly the same numeric order as the training CSV, e.g. angry,happy,neutral,sad,surprise",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--ema", type=float, default=0.7, help="Prediction smoothing from 0 to <1 (default: 0.7)")
    return parser.parse_args()


def load_model(checkpoint_path, labels):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    num_classes = int(checkpoint["num_classes"])
    if len(labels) != num_classes:
        raise ValueError(
            f"Checkpoint has {num_classes} classes but --labels contains {len(labels)} names. "
            "Supply one name per label, in the training CSV's label order."
        )
    if checkpoint.get("cnn_architecture", "efficientnet_b0") != "efficientnet_b0":
        raise ValueError("This realtime script is for an EfficientNet-B0 fusion checkpoint.")

    model = FusionModel(
        num_classes=num_classes,
        pretrained_cnn=False,  # weights are completely overwritten by the checkpoint; do not download ImageNet weights
        freeze_mode="none",
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scaler_mean"], checkpoint["scaler_scale"], int(checkpoint["img_size"])


def crop_face(frame_rgb, landmarks):
    """Crop a padded square around the face and make landmarks relative to that crop."""
    height, width = frame_rgb.shape[:2]
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    left, right = max(0, int(np.floor(xs.min()))), min(width, int(np.ceil(xs.max())))
    top, bottom = max(0, int(np.floor(ys.min()))), min(height, int(np.ceil(ys.max())))
    side = max(right - left, bottom - top)
    pad = int(side * 0.35)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    x0, x1 = max(0, center_x - side // 2 - pad), min(width, center_x + side // 2 + pad)
    y0, y1 = max(0, center_y - side // 2 - pad), min(height, center_y + side // 2 + pad)

    crop = frame_rgb[y0:y1, x0:x1]
    crop_landmarks = landmarks.copy()
    crop_landmarks[:, 0] -= x0
    crop_landmarks[:, 1] -= y0
    return crop, crop_landmarks, (x0, y0, x1, y1)


def prepare_inputs(frame_bgr, face_landmarks, scaler_mean, scaler_scale, transform):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    landmarks = np.asarray(
        [[point.x * width, point.y * height, point.z * width] for point in face_landmarks.landmark],
        dtype=np.float32,
    )
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected {NUM_LANDMARKS} face landmarks, received {landmarks.shape[0]}.")

    face_rgb, landmarks, box = crop_face(frame_rgb, landmarks)
    raw = landmarks[None, ...]
    angles, centers = compute_roll_angles_and_centers(raw)
    aligned_landmarks = align_landmarks(raw, angles, centers)
    normalized = normalize_landmarks(aligned_landmarks).reshape(1, -1)
    standardized = (normalized - scaler_mean) / scaler_scale

    aligned_image = align_image(Image.fromarray(face_rgb), float(angles[0]), centers[0])
    image_tensor = transform(aligned_image).unsqueeze(0)
    landmark_tensor = torch.from_numpy(standardized.astype(np.float32))
    return image_tensor, landmark_tensor, box


def main():
    args = parse_args()
    if not 0.0 <= args.ema < 1.0:
        raise ValueError("--ema must be in the range [0, 1).")
    
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()]
    else:
        labels = ["angry", "happy", "neutral", "sad", "surprise"]
        
    model, scaler_mean, scaler_scale, img_size = load_model(args.checkpoint, labels)
    _, transform = build_transforms(img_size)

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not capture.isOpened() and os.name == "nt":
        # Fallback for Windows if CAP_DSHOW fails to open the camera device
        capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1 if another camera is selected.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # produces 478 landmarks, including iris points
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )
    smoothed_probs = None
    print("Webcam started. Press Q or Esc to exit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_face_landmarks:
                try:
                    image, landmark, (x0, y0, x1, y1) = prepare_inputs(
                        frame, result.multi_face_landmarks[0], scaler_mean, scaler_scale, transform
                    )
                    with torch.inference_mode():
                        probs = torch.softmax(model(image.to(DEVICE), landmark.to(DEVICE)), dim=1)[0].cpu().numpy()
                    smoothed_probs = probs if smoothed_probs is None else args.ema * smoothed_probs + (1 - args.ema) * probs
                    class_id = int(np.argmax(smoothed_probs))
                    text = f"{labels[class_id]}: {smoothed_probs[class_id] * 100:.1f}%"
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 0), 2)
                    cv2.putText(frame, text, (x0, max(28, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2, cv2.LINE_AA)
                except Exception as error:
                    print(f"[Prediction Error]: {error}")
                    cv2.putText(frame, f"Prediction error: {error}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                smoothed_probs = None
                cv2.putText(frame, "No face detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 190, 255), 2, cv2.LINE_AA)

            cv2.imshow("Fusion emotion prediction | Q or Esc: exit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        face_mesh.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
