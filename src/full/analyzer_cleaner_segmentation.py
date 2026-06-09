import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from shapely.geometry import Polygon
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src import config
from src.pair_utils import (
    align_old_image_to_new,
    append_pair_id,
    append_jsonl,
    build_pairs,
    clear_generated_files,
    compute_visual_change,
    ensure_runtime_dirs,
    export_detected_points,
    get_device,
    infer_dataset_year,
    image_size,
    load_pair_ids,
    load_image,
    normalize_polygon,
    optional_geo_center,
    polygon_match_metrics,
    polygon_to_bbox,
    resize_to_match,
)


def record_exception(exc: Exception):
    config.DETECTION_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with config.DETECTION_ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(traceback.format_exc())
        handle.write("\n")


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


def build_review_canvas(
    pair_id: str,
    old_image: np.ndarray,
    new_image: np.ndarray,
    annotated_image: np.ndarray,
    kept: int,
    alignment_used: bool,
) -> np.ndarray:
    display_old = resize_to_match(old_image, new_image)
    display_annotated = resize_to_match(annotated_image, new_image)
    images = [display_old, new_image, display_annotated]
    separator_width = 2
    image_height = max(image.shape[0] for image in images)
    image_width = sum(image.shape[1] for image in images) + (
        separator_width * (len(images) - 1)
    )
    canvas = np.full((image_height, image_width, 3), 0, dtype=np.uint8)

    offset_x = 0
    for idx, image in enumerate(images):
        canvas[
            0:image.shape[0],
            offset_x:offset_x + image.shape[1],
        ] = image
        offset_x += image.shape[1]
        if idx < len(images) - 1:
            offset_x += separator_width

    return canvas


def prepare_outputs(resume: bool):
    if resume:
        return

    config.VERIFIED_SEGMENTATION_JSONL.unlink(missing_ok=True)
    config.SEGMENTATION_PROCESSED_PAIRS_TXT.unlink(missing_ok=True)
    config.VERIFIED_SEGMENTATION_POINTS_TXT.unlink(missing_ok=True)
    config.VERIFIED_SEGMENTATION_POINTS_CSV.unlink(missing_ok=True)
    config.VERIFIED_SEGMENTATION_POINTS_GEOJSON.unlink(missing_ok=True)
    clear_generated_files(config.SEGMENTATION_ALL_PAIRS_DIR, "pair_*.jpg")
    clear_generated_files(config.VERIFIED_SEG_MASKS_DIR, "pair_*.jpg")


def filter_pairs_for_resume(pairs, resume: bool):
    if not resume:
        return pairs, 0

    finished_ids = load_pair_ids(config.SEGMENTATION_PROCESSED_PAIRS_TXT)
    remaining_pairs = [pair for pair in pairs if pair.pair_id not in finished_ids]
    skipped = len(pairs) - len(remaining_pairs)
    return remaining_pairs, skipped


def process_pair(
    pair,
    yolo_model,
    clip_model,
    clip_processor,
    device,
    clip_device,
    jsonl_lock,
    checkpoint_lock,
):
    old_year = infer_dataset_year(pair.old_path)
    new_year = infer_dataset_year(pair.new_path)
    adjacent_year_pair = (
        old_year is not None
        and new_year is not None
        and abs(new_year - old_year) <= 1
    )
    long_gap_pair = (
        old_year is not None
        and new_year is not None
        and abs(new_year - old_year) > 1
    )
    old_buffer_px = (
        config.ADJACENT_YEAR_BUFFER_SIZE_PX
        if adjacent_year_pair
        else config.LONG_GAP_BUFFER_SIZE_PX
        if long_gap_pair
        else config.BUFFER_SIZE_PX
    )
    overlap_min = (
        config.ADJACENT_YEAR_CHANGE_OVERLAP_MIN
        if adjacent_year_pair
        else config.LONG_GAP_CHANGE_OVERLAP_MIN
        if long_gap_pair
        else config.CHANGE_OVERLAP_MIN
    )
    buffered_overlap_min = (
        config.ADJACENT_YEAR_BUFFERED_OVERLAP_MIN
        if adjacent_year_pair
        else config.LONG_GAP_BUFFERED_OVERLAP_MIN
        if long_gap_pair
        else config.BUFFERED_OVERLAP_MIN
    )
    distance_limit_scale = (
        config.ADJACENT_YEAR_DISTANCE_LIMIT_SCALE
        if adjacent_year_pair
        else config.LONG_GAP_DISTANCE_LIMIT_SCALE
        if long_gap_pair
        else 1.0
    )
    visual_threshold = (
        config.ADJACENT_YEAR_VISUAL_THRESHOLD
        if adjacent_year_pair
        else config.LONG_GAP_VISUAL_THRESHOLD
        if long_gap_pair
        else config.NEW_BUILDING_VISUAL_THRESHOLD
    )
    clip_threshold = (
        config.ADJACENT_YEAR_CLIP_CONF_THRESHOLD
        if adjacent_year_pair
        else config.LONG_GAP_CLIP_CONF_THRESHOLD
        if long_gap_pair
        else config.CLIP_CONF_THRESHOLD
    )
    old_conf_threshold = (
        config.ADJACENT_YEAR_OLD_YOLO_CONF_THRESHOLD
        if adjacent_year_pair
        else config.LONG_GAP_OLD_YOLO_CONF_THRESHOLD
        if long_gap_pair
        else config.YOLO_CONF_THRESHOLD
    )
    area_change_limit = (
        config.LONG_GAP_AREA_CHANGE_RATIO
        if long_gap_pair
        else config.AREA_CHANGE_RATIO
    )

    old_raw = load_image(pair.old_path)
    new_image = load_image(pair.new_path)
    old_image, alignment_used = align_old_image_to_new(old_raw, new_image)

    old_result = yolo_model.predict(
        old_image,
        conf=old_conf_threshold,
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

    old_polygons = []
    if old_result.masks:
        for coords in old_result.masks.xy:
            if len(coords) < 3:
                continue
            polygon = normalize_polygon(Polygon(coords).buffer(old_buffer_px))
            if config.MIN_AREA <= polygon.area <= config.MAX_AREA:
                old_polygons.append(polygon)

    annotated = new_image.copy()
    kept = 0
    rows = []
    confs = new_result.boxes.conf.cpu().numpy() if new_result.boxes else []

    if new_result.masks:
        for index, coords in enumerate(new_result.masks.xy):
            if len(coords) < 3:
                continue
            polygon = normalize_polygon(Polygon(coords))
            if not (config.MIN_AREA <= polygon.area <= config.MAX_AREA):
                continue

            matched_polygon = None
            best_metrics = None
            best_score = -1.0
            for old_polygon in old_polygons:
                metrics = polygon_match_metrics(polygon, old_polygon)
                distance_ratio = metrics["centroid_distance"] / max(
                    1.0,
                    metrics["distance_limit"],
                )
                score = max(
                    metrics["direct_overlap"],
                    metrics["buffered_overlap"],
                ) - (0.15 * distance_ratio)
                if score > best_score:
                    best_score = score
                    matched_polygon = old_polygon
                    best_metrics = metrics

            area_change_ratio = 1.0
            visual_change = 0.0
            direct_overlap = 0.0
            buffered_overlap = 0.0
            centroid_distance = 0.0
            if matched_polygon is not None and best_metrics is not None:
                direct_overlap = best_metrics["direct_overlap"]
                buffered_overlap = best_metrics["buffered_overlap"]
                centroid_distance = best_metrics["centroid_distance"]
                area_change_ratio = best_metrics["area_change_ratio"]
                distance_limit = best_metrics["distance_limit"] * distance_limit_scale
                likely_same_building = (
                    direct_overlap >= overlap_min
                    or (
                        buffered_overlap >= buffered_overlap_min
                        and centroid_distance <= distance_limit
                        and area_change_ratio <= area_change_limit
                    )
                    or (
                        centroid_distance <= (distance_limit * 0.65)
                        and buffered_overlap >= (buffered_overlap_min * 0.7)
                        and area_change_ratio <= area_change_limit
                    )
                )
            else:
                likely_same_building = False

            if likely_same_building:
                continue

            x1, y1, x2, y2 = polygon_to_bbox(polygon)
            old_bbox = (x1, y1, x2, y2)
            if matched_polygon is not None:
                old_bbox = polygon_to_bbox(matched_polygon)
            visual_change = compute_visual_change(
                old_image,
                new_image,
                old_bbox,
                (x1, y1, x2, y2),
            )
            if visual_change < visual_threshold:
                continue
            crop = new_image[
                max(0, y1 - config.CLIP_PADDING): min(new_image.shape[0], y2 + config.CLIP_PADDING),
                max(0, x1 - config.CLIP_PADDING): min(new_image.shape[1], x2 + config.CLIP_PADDING),
            ]
            score = clip_score(crop, clip_model, clip_processor, clip_device)
            if score < clip_threshold:
                continue

            kept += 1
            center_x = float(polygon.centroid.x)
            center_y = float(polygon.centroid.y)
            point_x = int(round(center_x))
            point_y = int(round(center_y))
            cv2.circle(annotated, (point_x, point_y), 5, (0, 0, 255), -1)
            width, height = image_size(pair.new_path)
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "status": "new_building",
                    "confidence": round(float(confs[index]) if len(confs) > index else 0.0, 4),
                    "clip_score": round(score, 4),
                    "overlap_ratio": round(direct_overlap, 4),
                    "buffered_overlap_ratio": round(buffered_overlap, 4),
                    "centroid_distance_px": round(centroid_distance, 2),
                    "area_change_ratio": round(area_change_ratio, 4),
                    "visual_change_score": round(visual_change, 4),
                    "alignment_used": alignment_used,
                    "centroid_px": {"x": round(center_x, 2), "y": round(center_y, 2)},
                    "image_size": {"width": width, "height": height},
                    "old_image": str(pair.old_path),
                    "new_image": str(pair.new_path),
                    "bbox": [x1, y1, x2, y2],
                    "polygon": np.asarray(coords).round(2).tolist(),
                    "geo_center": optional_geo_center(pair.new_path, center_x, center_y),
                }
            )

    if kept:
        review = build_review_canvas(
            pair.pair_id,
            old_raw,
            new_image,
            annotated,
            kept,
            alignment_used,
        )
        cv2.imwrite(str(config.VERIFIED_SEG_MASKS_DIR / f"pair_{pair.pair_id}.jpg"), review)
        with jsonl_lock:
            for row in rows:
                append_jsonl(config.VERIFIED_SEGMENTATION_JSONL, row)
    with checkpoint_lock:
        append_pair_id(config.SEGMENTATION_PROCESSED_PAIRS_TXT, pair.pair_id)
    if kept:
        return "detected"
    return "clean"


def main(
    model_path: Optional[str] = None,
    workers: Optional[int] = None,
    pair_id: Optional[str] = None,
    max_pairs: Optional[int] = None,
    resume: bool = False,
):
    ensure_runtime_dirs()
    prepare_outputs(resume=resume)
    device = get_device()
    yolo_weights = Path(model_path) if model_path else config.SEGMENTATION_MODEL_PATH
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
    pairs, skipped = filter_pairs_for_resume(pairs, resume=resume)
    if not pairs:
        if resume and skipped:
            print({"resume_skipped": skipped, "remaining": 0, "status": "already_complete"})
            return
        raise SystemExit("No matching image pairs found.")

    stats = Counter()
    max_workers = workers or config.DEFAULT_MAX_WORKERS
    jsonl_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    if max_workers <= 1:
        for pair in tqdm(
            pairs,
            total=len(pairs),
            desc="segment-full",
        ):
            try:
                stats[
                    process_pair(
                        pair,
                        yolo_model,
                        clip_model,
                        clip_processor,
                        device,
                        clip_device,
                        jsonl_lock,
                        checkpoint_lock,
                    )
                ] += 1
            except Exception as exc:
                record_exception(exc)
                stats["error"] += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_pair,
                    pair,
                    yolo_model,
                    clip_model,
                    clip_processor,
                    device,
                    clip_device,
                    jsonl_lock,
                    checkpoint_lock,
                ): pair
                for pair in pairs
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="segment-full",
            ):
                try:
                    stats[future.result()] += 1
                except Exception as exc:
                    record_exception(exc)
                    stats["error"] += 1

    print(f"device={device}")
    if resume:
        stats["resume_skipped"] = skipped
    total_rows, exported_points = export_detected_points(
        config.VERIFIED_SEGMENTATION_JSONL,
        config.VERIFIED_SEGMENTATION_POINTS_TXT,
        config.VERIFIED_SEGMENTATION_POINTS_CSV,
        config.VERIFIED_SEGMENTATION_POINTS_GEOJSON,
    )
    stats["json_rows"] = total_rows
    stats["point_exports"] = exported_points
    print(dict(stats))


if __name__ == "__main__":
    main()

