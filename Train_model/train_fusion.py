"""
Chạy:
    python train_fusion.py --csv "../Output/face_landmarks.csv" --dataset_dir "../data/emotion" --out ../models/fusion_model.pt

Nếu muốn đóng băng phần lớn CNN backbone lúc đầu (khuyên dùng nếu đang overfit nhanh):
    python train_fusion.py --csv "../Output/face_landmarks.csv" --dataset_dir "../data/emotion" --freeze_backbone partial --out ../models/fusion_model_partial.pt
"""

import argparse
import os
import time
import wandb
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
NUM_LANDMARKS = 478
NUM_CLASSES = 5
INPUT_SIZE = NUM_LANDMARKS * 3  # 1434

BATCH_SIZE = 128
MAX_EPOCHS = 100              # cao, vì early stopping sẽ tự dừng sớm hơn
PATIENCE = 4                 # dừng nếu val macro-F1 không tăng sau 4 epoch
LR_PATIENCE = 3
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Landmark normalization (dịch tâm về mũi + chia theo khoảng cách 2 mắt)
# ----------------------------------------------------------------------------
def normalize_landmarks(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    nose = X[:, 1:2, :]
    X = X - nose

    left_eye = X[:, 33, :]
    right_eye = X[:, 263, :]
    eye_distance = np.linalg.norm(left_eye - right_eye, axis=1, keepdims=True)
    eye_distance = np.maximum(eye_distance, 1e-6)

    X = X / eye_distance[:, :, None]
    return X


# ----------------------------------------------------------------------------
# Face alignment (roll correction)
#
# Ảnh trong dataset đã crop sát mặt nhưng CHƯA xoay thẳng (mặt có thể nghiêng
# nhiều so với ±10° mà RandomRotation augment cover được). Nếu không align,
# CNN và Landmark MLP phải tự học cách "bỏ qua" độ nghiêng đầu (roll) thay vì
# tập trung vào biểu cảm — tốn data vô ích và dễ generalize kém.
#
# Cách làm: dùng chính landmark mắt trái/phải (đã có sẵn, pixel thật) để tính
# góc roll rồi xoay landmark VÀ ảnh cùng 1 phép xoay (cùng center, cùng angle)
# để đường nối 2 mắt về nằm ngang. Công thức xoay landmark bên dưới khớp 1-1
# với cv2.getRotationMatrix2D(center, angle, 1.0) — verify bằng tay để đảm bảo
# ảnh (rotate trong Dataset bằng cv2) và landmark (rotate ở đây bằng numpy,
# cho nhanh vì vector hoá được trên toàn bộ tập) luôn khớp pha với nhau.
# ----------------------------------------------------------------------------
def compute_roll_angles_and_centers(X: np.ndarray):
    """X: (N, 478, 3) landmark pixel gốc, CHƯA normalize.
    Trả về góc roll (độ) và tâm xoay (trung điểm 2 mắt) cho từng sample."""
    left_eye = X[:, 33, :2]
    right_eye = X[:, 263, :2]
    dx = right_eye[:, 0] - left_eye[:, 0]
    dy = right_eye[:, 1] - left_eye[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))
    centers = (left_eye + right_eye) / 2.0
    return angles, centers


def align_landmarks(X: np.ndarray, angles: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Xoay toạ độ (x, y) của từng sample quanh tâm 2 mắt của chính nó để
    đường nối 2 mắt nằm ngang (roll = 0). z giữ nguyên (chỉ xoay trong mặt
    phẳng ảnh)."""
    X = X.copy()
    theta = np.radians(angles)
    alpha = np.cos(theta)
    beta = np.sin(theta)
    cx, cy = centers[:, 0], centers[:, 1]

    x, y = X[:, :, 0], X[:, :, 1]
    x_new = alpha[:, None] * x + beta[:, None] * y + ((1 - alpha) * cx - beta * cy)[:, None]
    y_new = -beta[:, None] * x + alpha[:, None] * y + (beta * cx + (1 - alpha) * cy)[:, None]
    # ## Dễ đọc (làm vậy để tăng tốc độ lên)
    # x0 = x - cx
    # y0 = y - cy
    # x_new = alpha * x0 + beta * y0 + cx
    # y_new = -beta * x0 + alpha * y0 + cy
    
    X[:, :, 0] = x_new
    X[:, :, 1] = y_new
    return X


def align_image(image: Image.Image, angle: float, center: np.ndarray) -> Image.Image:
    """Xoay ảnh quanh `center` (trung điểm 2 mắt, pixel gốc) đúng `angle` độ —
    cùng phép xoay đã áp cho landmark ở align_landmarks, để ảnh và landmark
    không bị lệch pha. Vì ảnh đã crop sát mặt, mở rộng canvas trước khi xoay
    (pad bằng BORDER_REFLECT thay vì đen) để không bị cắt mất cằm/trán/tai."""
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    pad_h = (diag - h) // 2 + 1
    pad_w = (diag - w) // 2 + 1
    padded = cv2.copyMakeBorder(image_np, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REFLECT)

    new_center = (float(center[0]) + pad_w, float(center[1]) + pad_h)
    M = cv2.getRotationMatrix2D(new_center, float(angle), 1.0)
    rotated = cv2.warpAffine(
        padded, M, (padded.shape[1], padded.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    return Image.fromarray(rotated)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, image_paths, landmarks, labels, angles, centers, dataset_dir, transform=None):
        self.image_paths = image_paths
        self.landmarks = landmarks
        self.labels = labels
        self.angles = angles
        self.centers = centers
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        # Xoay ảnh cùng góc/tâm đã dùng để xoay landmark (align_landmarks) —
        # đảm bảo ảnh đưa vào CNN và landmark đưa vào MLP khớp pha nhau.
        image = align_image(image, self.angles[idx], self.centers[idx])

        if self.transform:
            image = self.transform(image)

        landmarks = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, landmarks, label


def build_transforms(img_size: int = IMG_SIZE):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # khi fine-tune toàn bộ EfficientNet-B0.
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class LandmarkMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)


class CNNBranch(nn.Module):
    """
    EfficientNet-B0 pretrained -> vector 1280D

    freeze_mode:
      - "none":    fine-tune toàn bộ EfficientNet
      - "partial": chỉ fine-tune block cuối (features[-1])
      - "full":    đóng băng toàn bộ backbone
    """

    def __init__(self, pretrained: bool = True, freeze_mode: str = "none"):
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.efficientnet_b0(weights=weights)

        # Bỏ classifier, giữ lại feature extractor
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.freeze_mode = freeze_mode

        if freeze_mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif freeze_mode == "partial":
            # Đóng băng toàn bộ
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Chỉ mở block cuối
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class FusionModel(nn.Module):
    def __init__(self, num_classes: int, pretrained_cnn: bool = True, freeze_mode: str = "none"):
        super().__init__()
        self.freeze_mode = freeze_mode
        self.cnn = CNNBranch(pretrained=pretrained_cnn, freeze_mode=freeze_mode)
        self.landmark = LandmarkMLP()

        # EfficientNet-B0 trả về vector 1280D sau khi thay classifier bằng Identity.
        fusion_in = 1280 + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def train(self, mode: bool = True):
        """Override để giữ BatchNorm của phần backbone bị đóng băng ở eval mode.

        Chỉ set requires_grad=False KHÔNG đủ để "đóng băng" thật sự — nếu
        module vẫn ở train mode, các lớp BatchNorm bên trong vẫn cập nhật
        running_mean/running_var theo batch hiện tại dù weight không nhận
        gradient. Với freeze_mode="full" hoặc "partial", các phần bị đóng
        băng cần ở eval() để thống kê BN không trôi theo data mới."""
        super().train(mode)
        if mode and self.freeze_mode in ("partial", "full"):
            self.cnn.backbone.eval()
            if self.freeze_mode == "partial":
                # Block cuối đang được fine-tune thật, nên BN của riêng nó vẫn
                # phải ở train mode để học thống kê phù hợp với data mới.
                self.cnn.backbone.features[-1].train()
        return self

    def forward(self, image, landmark_feat):
        cnn_feat = self.cnn(image)
        lm_feat = self.landmark(landmark_feat)
        fused = torch.cat([cnn_feat, lm_feat], dim=1)
        return self.classifier(fused)


# ----------------------------------------------------------------------------
# Train / eval loop
# ----------------------------------------------------------------------------
def _print_progress(stage, epoch, total_epochs, batch_idx, total_batches, start_time):
    """In tiến độ định kỳ mà không cần thêm dependency như tqdm."""
    elapsed = time.perf_counter() - start_time
    batches_per_second = batch_idx / max(elapsed, 1e-6)
    remaining_seconds = (total_batches - batch_idx) / max(batches_per_second, 1e-6)
    progress = 100.0 * batch_idx / total_batches
    print(
        f"  {stage} | Epoch {epoch:02d}/{total_epochs} | "
        f"{batch_idx:3d}/{total_batches} ({progress:5.1f}%) | "
        f"{batches_per_second:.2f} batch/s | ETA {remaining_seconds / 60:.1f} min",
        flush=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_batches = len(loader)
    progress_every = max(1, total_batches // 20)  # khoảng 20 cập nhật mỗi epoch
    start_time = time.perf_counter()

    for batch_idx, (images, landmarks, labels) in enumerate(loader, start=1):
        images, landmarks, labels = images.to(DEVICE), landmarks.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, landmarks)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

        if batch_idx % progress_every == 0 or batch_idx == total_batches:
            _print_progress("Train", epoch, total_epochs, batch_idx, total_batches, start_time)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, epoch, total_epochs):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    total_batches = len(loader)
    progress_every = max(1, total_batches // 10)  # khoảng 10 cập nhật validation
    start_time = time.perf_counter()

    for batch_idx, (images, landmarks, labels) in enumerate(loader, start=1):
        images, landmarks, labels = images.to(DEVICE), landmarks.to(DEVICE), labels.to(DEVICE)

        logits = model(images, landmarks)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        if batch_idx % progress_every == 0 or batch_idx == total_batches:
            _print_progress("Validation", epoch, total_epochs, batch_idx, total_batches, start_time)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / total_samples, total_correct / total_samples, macro_f1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(csv_path, dataset_dir, out_path, freeze_mode, pretrained_cnn, epochs, patience,
         weight_decay, label_smoothing):
    wandb.init(
        project="emotion-fusion",       # tên project, tự đặt
        name=f"efficientnet-b0_{freeze_mode}",   # tên run, giúp phân biệt các lần chạy
        config={
            "freeze_mode": freeze_mode,
            "epochs": epochs,
            "patience": patience,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "loss_type": "CrossEntropyLoss",
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
    })
    print(f"Device: {DEVICE}")

    # --- Đọc CSV landmark ---
    df = pd.read_csv(csv_path)
    image_paths = df["filepath"].values
    X_landmarks = df.drop(columns=["filepath", "label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    X_landmarks = X_landmarks.reshape(-1, NUM_LANDMARKS, 3)
    print(f"Tổng số sample: {len(df)} | Landmark shape: {X_landmarks.shape}")

    # --- Face alignment: tính góc roll từ landmark pixel gốc, xoay landmark
    # về đường-nối-2-mắt-nằm-ngang TRƯỚC khi normalize (translate + scale).
    # Ảnh sẽ được xoay cùng góc/tâm này khi load trong Dataset. ---
    roll_angles, eye_centers = compute_roll_angles_and_centers(X_landmarks)
    X_landmarks_aligned = align_landmarks(X_landmarks, roll_angles, eye_centers)

    # --- Normalize landmark ---
    X_scale = normalize_landmarks(X_landmarks_aligned)

    # --- Split train/val/test 70/15/15 (split luôn angles/centers theo cùng index) ---
    (paths_train, paths_temp, X_train, X_temp, y_train, y_temp,
     angles_train, angles_temp, centers_train, centers_temp) = train_test_split(
        image_paths, X_scale, y, roll_angles, eye_centers,
        test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    (paths_val, paths_test, X_val, X_test, y_val, y_test,
     angles_val, angles_test, centers_val, centers_test) = train_test_split(
        paths_temp, X_temp, y_temp, angles_temp, centers_temp,
        test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {len(paths_train)} | Val: {len(paths_val)} | Test: {len(paths_test)}")

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # --- StandardScaler (fit CHỈ trên train) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # --- Dataset & DataLoader ---
    train_transform, val_transform = build_transforms(IMG_SIZE)

    train_dataset = EmotionDataset(paths_train, X_train, y_train, angles_train, centers_train, dataset_dir, train_transform)
    val_dataset = EmotionDataset(paths_val, X_val, y_val, angles_val, centers_val, dataset_dir, val_transform)
    test_dataset = EmotionDataset(paths_test, X_test, y_test, angles_test, centers_test, dataset_dir, val_transform)

    num_workers = 4 if os.name == "nt" else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # --- Model + class weight + optimizer ---
    class_counts = np.bincount(y_train)
    print("Class counts:", class_counts)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print("Class weights:", class_weights)

    model = FusionModel(
        num_classes=NUM_CLASSES,
        pretrained_cnn=pretrained_cnn,
        freeze_mode=freeze_mode,
    ).to(DEVICE)
    print(f"CNN: EfficientNet-B0 | pretrained = {pretrained_cnn} | freeze_mode = '{freeze_mode}'")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing).to(DEVICE)
    print(f"Loss: CrossEntropyLoss(label_smoothing={label_smoothing})")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=weight_decay,
    )
    print(f"weight_decay = {weight_decay}")
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=LR_PATIENCE)

    def save_best_checkpoint(state_dict, epoch, val_f1):
        """Lưu ngay checkpoint tốt nhất; ghi qua file tạm để tránh file dở dang."""
        checkpoint = {
            "model_state_dict": state_dict,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "num_classes": NUM_CLASSES,
            "num_landmarks": NUM_LANDMARKS,
            "img_size": IMG_SIZE,
            "cnn_architecture": "efficientnet_b0",
            "cnn_feature_dim": 1280,
            "pretrained_cnn": pretrained_cnn,
            "freeze_mode": freeze_mode,
            "best_epoch": epoch,
            "best_val_macro_f1": val_f1,
        }
        temp_path = f"{out_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, out_path)
        print(f">> Đã lưu checkpoint tốt nhất (epoch {epoch}, val macro-F1={val_f1:.4f}): {out_path}", flush=True)

    # --- Training loop với early stopping theo val macro-F1 ---
    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, epochs
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, epoch, epochs)
        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": current_lr,
        })
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            save_best_checkpoint(best_state, epoch, best_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n>> Val macro-F1 không cải thiện sau {patience} epoch. Dừng ở epoch {epoch}.")
                break

    # --- Load lại best model trước khi test ---
    model.load_state_dict(best_state)
    print(f"\nBest val macro-F1: {best_f1:.4f}")

    # --- Evaluation cuối cùng trên test set ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, landmarks, labels in test_loader:
            logits = model(images.to(DEVICE), landmarks.to(DEVICE))
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=all_preds, y_true=all_labels,
                class_names=[str(i) for i in range(NUM_CLASSES)],
            )
        })
    wandb.summary["best_val_macro_f1"] = best_f1
    wandb.finish()

    print("\n=== Classification report (test set) ===")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    print("=== Confusion matrix (test set) ===")
    print(confusion_matrix(all_labels, all_preds))

    print(f"\nCheckpoint tốt nhất đã được lưu ở: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Đường dẫn CSV landmark (có cột filepath)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Thư mục gốc chứa ảnh (fane_data)")
    parser.add_argument("--out", type=str, default="fusion_model.pt")
    parser.add_argument("--freeze_backbone", type=str, default="none", choices=["none", "partial", "full"],
                         help="none: fine-tune toàn bộ EfficientNet-B0 | partial: chỉ fine-tune block cuối | "
                              "full: đóng băng toàn bộ backbone, CNN chỉ làm feature extractor cố định")
    parser.add_argument("--no_pretrained_cnn", action="store_false", dest="pretrained_cnn",
                         help="Không tải trọng số ImageNet cho EfficientNet-B0 (mặc định có dùng pretrained weights)")
    parser.set_defaults(pretrained_cnn=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                         help="Tăng lên nếu đang overfit (mặc định cũ là 1e-4, đã bump lên 5e-4)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                         help="0 để tắt. >0 giúp model bớt tự tin thái quá, thường giảm overfit")
    args = parser.parse_args()
    main(args.csv, args.dataset_dir, args.out, args.freeze_backbone, args.pretrained_cnn, args.epochs, args.patience,
         args.weight_decay, args.label_smoothing)
