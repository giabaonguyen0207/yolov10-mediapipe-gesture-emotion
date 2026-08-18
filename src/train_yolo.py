import wandb
from ultralytics import YOLO

DATA = "config/data.yaml"
MODEL = "models/YOLOv10n_gestures.pt"

wandb.init(
    project="my-awesome-project",
    entity="huynhnguyenhuyhoang01092007-national-technology",
    name="YOLOv10n-gestures"
)

def log_metrics(trainer):
    m = trainer.metrics

    p = float(m["metrics/precision(B)"])
    r = float(m["metrics/recall(B)"])

    wandb.log({
        "epoch": trainer.epoch + 1,
        "precision": p,
        "recall": r,
        "F1": 2 * p * r / (p + r) if p + r else 0,
        "mAP50": float(m["metrics/mAP50(B)"]),
        "mAP50-95": float(m["metrics/mAP50-95(B)"])
    })

model = YOLO(MODEL)
model.add_callback("on_fit_epoch_end", log_metrics)

model.train(
    data=DATA,
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda",
    patience=10,
    workers=0,
    save=True,
    exist_ok=True,
    resume=False
)

wandb.finish()