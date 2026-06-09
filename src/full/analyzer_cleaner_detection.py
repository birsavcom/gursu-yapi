import gc
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src import config
from src.pair_utils import (
    append_jsonl,
    bbox_distance,
    build_pairs,
    compute_visual_change,
    ensure_runtime_dirs,
    get_device,
    image_size,
    load_image,
    optional_geo_center,
)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)


def enhance_image(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(1.1)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(1.2)
    return pil_image


def clip_score(crop, model, processor, device):
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


def draw_boxes(image: np.ndarray, boxes, color, label: str):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def build_review_canvas(pair_id: str, old_image: np.ndarray, new_image: np.ndarray):
    header_height = 48
    footer_height = 64
    height = max(old_image.shape[0], new_image.shape[0]) + header_height + footer_height
    width = old_image.shape[1] + new_image.shape[1]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[header_height:header_height + old_image.shape[0], :old_image.shape[1]] = old_image
    canvas[header_height:header_height + new_image.shape[0], old_image.shape[1]:old_image.shape[1] + new_image.shape[1]] = new_image

    cv2.putText(canvas, f"PAIR {pair_id} | OLD", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"PAIR {pair_id} | NEW", (old_image.shape[1] + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)
    return canvas


def log_error(pair_id: str, exc: Exception):
    message = f"pair={pair_id} error={type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
    with config.DETECTION_ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message)


def process_pair(pair, yolo_model, clip_model, clip_processor, device, clip_device):
    old_image = new_image = None
    old_result = new_result = None
    try:
        old_image = load_image(pair.old_path)
        new_image = load_image(pair.new_path)

        old_result = yolo_model.predict(
            old_image,
            conf=config.YOLO_CONF_THRESHOLD,
            device=device,
            verbose=False,
            agnostic_nms=True,
        )[0]
        new_result = yolo_model.predict(
            new_image,
            conf=config.YOLO_CONF_THRESHOLD,
            device=device,
            verbose=False,
            agnostic_nms=True,
        )[0]

        old_boxes = old_result.boxes.xyxy.cpu().numpy() if old_result.boxes else []
        new_boxes = new_result.boxes.xyxy.cpu().numpy() if new_result.boxes else []
        new_confs = new_result.boxes.conf.cpu().numpy() if new_result.boxes else []

        old_annotated = old_image.copy()
        new_annotated = new_image.copy()
        draw_boxes(old_annotated, old_boxes, GREEN, "building")
        draw_boxes(new_annotated, new_boxes, GREEN, "building")

        kept = 0
        for index, box in enumerate(new_boxes):
            confidence = float(new_confs[index])
            matched = None
            best_distance = float("inf")
            for old_box in old_boxes:
                distance = bbox_distance(box, old_box)
                if distance < min(best_distance, config.DIST_THRESHOLD):
                    best_distance = distance
                    matched = old_box

            status = "new_building"
            area_change_ratio = 1.0
            visual_change = 0.0
            if matched is not None:
                old_area = max(1.0, float((matched[2] - matched[0]) * (matched[3] - matched[1])))
                new_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
                area_change_ratio = abs(new_area - old_area) / max(old_area, new_area)
                visual_change = compute_visual_change(old_image, new_image, matched, box)
                if area_change_ratio < config.AREA_CHANGE_RATIO and visual_change < config.VISUAL_CHANGE_THRESHOLD:
                    continue
                status = "changed_building"

            x1, y1, x2, y2 = map(int, box[:4])
            crop = new_image[
                max(0, y1 - config.CLIP_PADDING): min(new_image.shape[0], y2 + config.CLIP_PADDING),
                max(0, x1 - config.CLIP_PADDING): min(new_image.shape[1], x2 + config.CLIP_PADDING),
            ]
            score = clip_score(crop, clip_model, clip_processor, clip_device)
            if score < config.CLIP_CONF_THRESHOLD:
                continue

            kept += 1
            cv2.rectangle(new_annotated, (x1, y1), (x2, y2), RED, 3)
            cv2.putText(
                new_annotated,
                f"{status}:{score:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED,
                2,
                cv2.LINE_AA,
            )
            center_x = float((box[0] + box[2]) / 2.0)
            center_y = float((box[1] + box[3]) / 2.0)
            width, height = image_size(pair.new_path)
            append_jsonl(
                config.VERIFIED_DETECTION_JSONL,
                {
                    "pair_id": pair.pair_id,
                    "status": status,
                    "confidence": round(confidence, 4),
                    "clip_score": round(score, 4),
                    "area_change_ratio": round(area_change_ratio, 4),
                    "visual_change_score": round(visual_change, 4),
                    "centroid_px": {"x": round(center_x, 2), "y": round(center_y, 2)},
                    "image_size": {"width": width, "height": height},
                    "old_image": str(pair.old_path),
                    "new_image": str(pair.new_path),
                    "bbox": [round(float(v), 2) for v in box[:4]],
                    "geo_center": optional_geo_center(pair.new_path, center_x, center_y),
                },
            )

        review = build_review_canvas(pair.pair_id, old_annotated, new_annotated)
        footer_lines = [
            f"old_boxes={len(old_boxes)} | new_boxes={len(new_boxes)} | highlighted={kept}",
            "green=all detected buildings | red=present in new year but absent/changed vs old year",
        ]
        y = review.shape[0] - 20
        for line in reversed(footer_lines):
            cv2.putText(review, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 2, cv2.LINE_AA)
            y -= 26
        cv2.imwrite(str(config.ALL_PAIRS_DIR / f"pair_{pair.pair_id}.jpg"), review)

        if kept:
            cv2.imwrite(str(config.VERIFIED_MASKS_DIR / f"pair_{pair.pair_id}.jpg"), review)
            return "detected"
        return "clean"
    finally:
        del old_result, new_result, old_image, new_image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(
    model_path: Optional[str] = None,
    workers: Optional[int] = None,
    pair_id: Optional[str] = None,
    max_pairs: Optional[int] = None,
):
    ensure_runtime_dirs()
    config.VERIFIED_DETECTION_JSONL.unlink(missing_ok=True)
    config.DETECTION_ERROR_LOG.unlink(missing_ok=True)
    device = get_device()
    yolo_weights = Path(model_path) if model_path else config.DETECTION_MODEL_PATH
    yolo_model = YOLO(str(yolo_weights)).to(device)

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(
        config.CLIP_MODEL_NAME,
        local_files_only=True,
    ).to(clip_device)
    clip_processor = CLIPProcessor.from_pretrained(
        config.CLIP_MODEL_NAME,
        local_files_only=True,
    )

    pairs = build_pairs(pair_id=pair_id, max_pairs=max_pairs)
    if not pairs:
        raise SystemExit("No matching image pairs found.")

    stats = Counter()
    for pair in tqdm(pairs, total=len(pairs), desc="detect-full"):
        try:
            stats[process_pair(pair, yolo_model, clip_model, clip_processor, device, clip_device)] += 1
        except Exception as exc:
            stats["error"] += 1
            log_error(pair.pair_id, exc)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"device={device}")
    print(dict(stats))


if __name__ == "__main__":
    main()

