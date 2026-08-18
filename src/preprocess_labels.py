import random
import shutil
from pathlib import Path
import yaml
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SRC = Path("data/raw/roboflow_raw")
DST = Path("data/processed/Preprocessed_DTS")


NAME_MAP = {
    "Call-Me": "call",
    "Fingers-Crossed": None,
    "Fingers-Spread": "palm",
    "Hand-Raised": "stop",
    "Heart": "hand_heart",
    "Italian-Gesture": None,
    "Left-Fist": "fist",
    "Live-Long-Prosper": None,
    "Love-You": None,
    "Middle": "middle_finger",
    "Okay": "ok",
    "Oncoming-Punch": "fist",
    "Peace": "peace",
    "Pinched-Fingers": None,
    "Point-Down": "point",
    "Point-Left": "point",
    "Point-Right": "point",
    "Pointing-Up": "one",
    "Praying-Hands": "holy",
    "Raised-Fist": "fist",
    "Right-Fist": "fist",
    "Rock": "rock",
    "Thumbs-Down": "dislike",
    "Thumbs-Up": "like",
}


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def build_class_mapping(class_names):
    old_to_new = {}
    name_to_id = {}
    new_names = []

    for old_id, old_name in enumerate(class_names):
        new_name = NAME_MAP.get(old_name)

        if new_name is None:
            continue

        if new_name not in name_to_id:
            name_to_id[new_name] = len(new_names)
            new_names.append(new_name)

        old_to_new[old_id] = name_to_id[new_name]

    return old_to_new, new_names


def collect_data(src, old_to_new):
    data = []
    dropped_boxes = 0
    kept_boxes = 0

    for split in ["train", "valid", "test"]:
        image_dir = src / split / "images"
        label_dir = src / split / "labels"

        if not image_dir.exists():
            continue

        for image in image_dir.glob("*.*"):
            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                continue

            new_lines = []

            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue

                fields = line.split()
                old_id = int(fields[0])

                if old_id not in old_to_new:
                    dropped_boxes += 1
                    continue

                fields[0] = str(old_to_new[old_id])
                new_lines.append(" ".join(fields))
                kept_boxes += 1

            if not new_lines:
                continue

            data.append((image, new_lines))

    return data, kept_boxes, dropped_boxes


def reduce_classes(data, class_names):
    for class_name in ["fist", "point"]:
        class_id = class_names.index(class_name)

        target = [
            x for x in data
            if class_id in {
                int(line.split()[0])
                for line in x[1]
            }
        ]

        random.shuffle(target)

        keep = len(target) // 3
        remove = target[keep:]

        data = [
            x for x in data
            if x not in remove
        ]

        print(f"{class_name}: {len(target)} -> {keep}")

    random.shuffle(data)

    return data


def split_data(data, class_names):
    y = np.zeros((len(data), len(class_names)), dtype=int)

    for i, (_, labels) in enumerate(data):
        for line in labels:
            y[i, int(line.split()[0])] = 1

    sss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )

    train_idx, temp_idx = next(sss.split(data, y))
    temp_data = [data[i] for i in temp_idx]

    sss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=42
    )

    val_idx, test_idx = next(sss.split(temp_data, y[temp_idx]))

    return {
        "train": [data[i] for i in train_idx],
        "valid": [temp_data[i] for i in val_idx],
        "test": [temp_data[i] for i in test_idx]
    }


def write_dataset(dst, splits, class_names):
    for split, items in splits.items():

        image_dir = dst / split / "images"
        label_dir = dst / split / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image, labels in items:

            shutil.copy2(
                image,
                image_dir / image.name
            )

            (label_dir / f"{image.stem}.txt").write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8"
            )

        print(f"[{split}] {len(items)} images")

    config = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names
    }

    (dst / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8"
    )


def main():

    config = yaml.safe_load(
        (SRC / "data.yaml").read_text(encoding="utf-8")
    )

    old_to_new, class_names = build_class_mapping(config["names"])

    print(f"Keeping {len(class_names)} classes:")
    print(class_names)

    data, kept_boxes, dropped_boxes = collect_data(
        SRC,
        old_to_new
    )

    print(f"\nBefore reduction:")
    print(f"Images: {len(data)}")
    print(f"Boxes kept: {kept_boxes}")
    print(f"Boxes dropped: {dropped_boxes}")

    data = reduce_classes(data, class_names)

    print(f"\nAfter reduction:")
    print(f"Images: {len(data)}")

    splits = split_data(data,class_names)

    write_dataset(DST, splits, class_names)

    print(f"\nDone! Dataset saved to {DST}")


if __name__ == "__main__":
    main()