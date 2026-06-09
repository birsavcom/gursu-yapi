import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.pair_utils import (
    ImagePair,
    align_old_image_to_new,
    append_jsonl,
    bbox_distance,
    build_pairs,
    compute_visual_change,
    ensure_runtime_dirs,
    get_device,
    image_size,
    load_image,
    optional_geo_center,
    polygon_match_metrics,
    polygon_to_bbox,
)


def draw_labeled_polygon(image: np.ndarray, polygon: Polygon, label: str) -> np.ndarray:
    canvas = image.copy()
    center_x = int(round(polygon.centroid.x))
    center_y = int(round(polygon.centroid.y))
    cv2.circle(canvas, (center_x, center_y), 5, (0, 0, 255), -1)
    return canvas


def draw_labeled_box(image: np.ndarray, box, label: str) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = map(int, box[:4])
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


def detect_pair_change(pair: ImagePair, model: YOLO, device: str) -> str:
    old_image = load_image(pair.old_path)
    new_image = load_image(pair.new_path)

    old_result = model.predict(
        old_image,
        conf=config.YOLO_CONF_THRESHOLD,
        device=device,
        verbose=False,
        agnostic_nms=True,
    )[0]
    new_result = model.predict(
        new_image,
        conf=config.YOLO_CONF_THRESHOLD,
        device=device,
        verbose=False,
        agnostic_nms=True,
    )[0]

    old_boxes = old_result.boxes.xyxy.cpu().numpy() if old_result.boxes else []
    found = False
    annotated = new_image.copy()

    if new_result.boxes:
        new_boxes = new_result.boxes.xyxy.cpu().numpy()
        confs = new_result.boxes.conf.cpu().numpy()
        for index, box in enumerate(new_boxes):
            confidence = float(confs[index])
            matched = None
            best_distance = float("inf")

            for old_box in old_boxes:
                distance = bbox_distance(box, old_box)
                if distance < min(best_distance, config.DIST_THRESHOLD):
                    best_distance = distance
                    matched = old_box

            status = "new_building"
            visual_change = 0.0
            area_change_ratio = 1.0

            if matched is not None:
                old_area = max(
                    1.0,
                    float((matched[2] - matched[0]) * (matched[3] - matched[1])),
                )
                new_area = max(
                    1.0,
                    float((box[2] - box[0]) * (box[3] - box[1])),
                )
                area_change_ratio = abs(new_area - old_area) / max(old_area, new_area)
                visual_change = compute_visual_change(old_image, new_image, matched, box)

                if (
                    area_change_ratio >= config.AREA_CHANGE_RATIO
                    or visual_change >= config.VISUAL_CHANGE_THRESHOLD
                ):
                    status = "changed_building"
                else:
                    continue

            found = True
            annotated = draw_labeled_box(
                annotated,
                box,
                f"{status}:{confidence:.2f}",
            )
            width, height = image_size(pair.new_path)
            center_x = float((box[0] + box[2]) / 2.0)
            center_y = float((box[1] + box[3]) / 2.0)
            append_jsonl(
                config.RAW_DETECTION_JSONL,
                {
                    "pair_id": pair.pair_id,
                    "status": status,
                    "confidence": round(confidence, 4),
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

    if found:
        out_path = config.RAW_MASKS_DIR / f"pair_{pair.pair_id}.jpg"
        cv2.imwrite(str(out_path), annotated)
        return "detected"

    return "clean"


def segmentation_pair_change(pair: ImagePair, model: YOLO, device: str) -> str:
    old_image = load_image(pair.old_path)
    new_image = load_image(pair.new_path)
    old_image, alignment_used = align_old_image_to_new(old_image, new_image)

    old_result = model.predict(
        old_image,
        conf=config.YOLO_CONF_THRESHOLD,
        device=device,
        verbose=False,
        agnostic_nms=True,
    )[0]
    new_result = model.predict(
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
            polygon = Polygon(coords).buffer(config.BUFFER_SIZE_PX)
            if config.MIN_AREA <= polygon.area <= config.MAX_AREA:
                old_polygons.append(polygon)

    found = False
    annotated = new_image.copy()
    confs = new_result.boxes.conf.cpu().numpy() if new_result.boxes else []

    if new_result.masks:
        for index, coords in enumerate(new_result.masks.xy):
            if len(coords) < 3:
                continue
            new_polygon = Polygon(coords)
            if not (config.MIN_AREA <= new_polygon.area <= config.MAX_AREA):
                continue

            matched_polygon = None
            best_metrics = None
            best_score = -1.0
            for old_polygon in old_polygons:
                metrics = polygon_match_metrics(new_polygon, old_polygon)
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
                likely_same_building = (
                    direct_overlap >= config.CHANGE_OVERLAP_MIN
                    or (
                        buffered_overlap >= config.BUFFERED_OVERLAP_MIN
                        and centroid_distance <= best_metrics["distance_limit"]
                    )
                )
            else:
                likely_same_building = False

            if likely_same_building:
                continue

            bbox_new = polygon_to_bbox(new_polygon)
            visual_change = compute_visual_change(
                old_image,
                new_image,
                bbox_new,
                bbox_new,
            )
            if visual_change < config.NEW_BUILDING_VISUAL_THRESHOLD:
                continue

            found = True
            confidence = float(confs[index]) if len(confs) > index else 0.0
            annotated = draw_labeled_polygon(
                annotated,
                new_polygon,
                "",
            )
            center_x = float(new_polygon.centroid.x)
            center_y = float(new_polygon.centroid.y)
            width, height = image_size(pair.new_path)
            append_jsonl(
                config.RAW_SEGMENTATION_JSONL,
                {
                    "pair_id": pair.pair_id,
                    "status": "new_building",
                    "confidence": round(confidence, 4),
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
                    "bbox": list(polygon_to_bbox(new_polygon)),
                    "polygon": np.asarray(coords).round(2).tolist(),
                    "geo_center": optional_geo_center(pair.new_path, center_x, center_y),
                },
            )

    if found:
        out_path = config.RAW_SEG_MASKS_DIR / f"pair_{pair.pair_id}.jpg"
        cv2.imwrite(str(out_path), annotated)
        return "detected"

    return "clean"


def _run_pair(pair: ImagePair, method: str, model: YOLO, device: str) -> str:
    if method == "detect":
        return detect_pair_change(pair, model, device)
    if method == "segment":
        return segmentation_pair_change(pair, model, device)
    raise ValueError(f"Unsupported method: {method}")


def _prepare_output(method: str):
    if method == "detect":
        config.RAW_DETECTION_JSONL.unlink(missing_ok=True)
    else:
        config.RAW_SEGMENTATION_JSONL.unlink(missing_ok=True)


def main(
    model_path: Optional[str] = None,
    workers: Optional[int] = None,
    pair_id: Optional[str] = None,
    max_pairs: Optional[int] = None,
    method: str = "detect",
):
    ensure_runtime_dirs()
    _prepare_output(method)

    device = get_device()
    pairs = build_pairs(pair_id=pair_id, max_pairs=max_pairs)
    if not pairs:
        raise SystemExit("No matching image pairs found.")

    if method == "detect":
        resolved_model = Path(model_path) if model_path else config.DETECTION_MODEL_PATH
    else:
        resolved_model = Path(model_path) if model_path else config.SEGMENTATION_MODEL_PATH

    model = YOLO(str(resolved_model)).to(device)
    stats = Counter()
    max_workers = workers or config.DEFAULT_MAX_WORKERS

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_pair, pair, method, model, device): pair
            for pair in pairs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"analyzer:{method}",
        ):
            try:
                stats[future.result()] += 1
            except Exception:
                stats["error"] += 1

    print(f"device={device}")
    print(f"pairs={len(pairs)}")
    print(dict(stats))


if __name__ == "__main__":
    main()

