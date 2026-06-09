import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.pair_utils import (
    append_jsonl,
    ensure_runtime_dirs,
    group_rows_by_pair,
    load_image,
    load_jsonl,
)


def enhance_image(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(1.1)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(1.2)
    return pil_image


def score_crop(crop, model, processor, device):
    if crop.size == 0:
        return 0.0

    prompts = config.POSITIVE_PROMPTS + config.NEGATIVE_PROMPTS
    inputs = processor(
        text=prompts,
        images=enhance_image(crop),
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    positive_count = len(config.POSITIVE_PROMPTS)
    positive_score = torch.sum(probs[0, :positive_count]).item()
    negative_score = torch.sum(probs[0, positive_count:]).item()
    if negative_score > positive_score:
        return 0.0
    return positive_score


def draw_row(image, row):
    x1, y1, x2, y2 = map(int, row["bbox"])
    label = f'{row["status"]}:{row["clip_score"]:.2f}'
    canvas = image.copy()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        canvas,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    ensure_runtime_dirs()

    source_json = (
        config.RAW_DETECTION_JSONL
        if config.RAW_DETECTION_JSONL.exists()
        else config.RAW_SEGMENTATION_JSONL
    )
    if source_json == config.RAW_DETECTION_JSONL:
        target_json = config.VERIFIED_DETECTION_JSONL
        target_dir = config.VERIFIED_MASKS_DIR
    else:
        target_json = config.VERIFIED_SEGMENTATION_JSONL
        target_dir = config.VERIFIED_SEG_MASKS_DIR

    rows = load_jsonl(source_json)
    if not rows:
        raise SystemExit(f"No rows found in {source_json}")

    target_json.unlink(missing_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(
        config.CLIP_MODEL_NAME,
        local_files_only=True,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        config.CLIP_MODEL_NAME,
        local_files_only=True,
    )
    grouped = group_rows_by_pair(rows)
    stats = Counter()

    for pair_id, items in tqdm(grouped.items(), desc="cleaner"):
        new_image = load_image(Path(items[0]["new_image"]))
        annotated = new_image.copy()
        kept = []

        for row in items:
            x1, y1, x2, y2 = row["bbox"]
            crop = new_image[
                max(0, int(y1) - config.CLIP_PADDING): min(new_image.shape[0], int(y2) + config.CLIP_PADDING),
                max(0, int(x1) - config.CLIP_PADDING): min(new_image.shape[1], int(x2) + config.CLIP_PADDING),
            ]
            score = score_crop(crop, model, processor, device)
            if score >= config.CLIP_CONF_THRESHOLD:
                row["clip_score"] = round(score, 4)
                kept.append(row)
                annotated = draw_row(annotated, row)
                append_jsonl(target_json, row)
                stats["kept"] += 1
            else:
                stats["dropped"] += 1

        if kept:
            cv2.imwrite(str(target_dir / f"pair_{pair_id}.jpg"), annotated)
            stats["files_kept"] += 1
        else:
            stats["files_dropped"] += 1

    print(dict(stats))
    print(f"source={source_json}")
    print(f"target={target_json}")


if __name__ == "__main__":
    main()

