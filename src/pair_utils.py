import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src import config


PAIR_ID_PATTERN = re.compile(r"(\d+)$")
BASE_TILE_SIZE_PX = 256.0


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    old_path: Path
    new_path: Path


def ensure_runtime_dirs():
    for path in [
        config.RESULTS_DIR,
        config.RAW_MASKS_DIR,
        config.VERIFIED_MASKS_DIR,
        config.RAW_SEG_MASKS_DIR,
        config.VERIFIED_SEG_MASKS_DIR,
        config.ALL_PAIRS_DIR,
        config.SEGMENTATION_ALL_PAIRS_DIR,
        config.DEBUG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def clear_generated_files(directory: Path, pattern: str = "*"):
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def completed_pair_ids(review_dir: Path, prefix: str = "pair_") -> set[str]:
    if not review_dir.exists():
        return set()

    pair_ids: set[str] = set()
    for path in review_dir.iterdir():
        if not path.is_file():
            continue
        if not path.stem.startswith(prefix):
            continue
        suffix = path.stem[len(prefix):]
        if suffix:
            pair_ids.add(suffix.zfill(4))
    return pair_ids


def load_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    pair_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            pair_id = line.strip()
            if pair_id:
                pair_ids.add(pair_id.zfill(4))
    return pair_ids


def append_pair_id(path: Path, pair_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(str(pair_id).zfill(4) + "\n")


def extract_pair_id(path: Path) -> str:
    match = PAIR_ID_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Could not extract pair id from {path.name}")
    return match.group(1).zfill(4)


def infer_dataset_year(path: Path) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(path))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _build_pairs_by_tile(
    old_dir: Path,
    new_dir: Path,
    old_metadata: Dict[str, Dict[str, str]],
    new_metadata: Dict[str, Dict[str, str]],
    pair_id: Optional[str],
    max_pairs: Optional[int],
) -> List[ImagePair]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    # (x, y, z) -> path for each dataset
    old_tile_map: Dict[Tuple[int, int, int], Path] = {}
    for path in old_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
        pid = extract_pair_id(path)
        row = old_metadata.get(pid)
        if row is None:
            continue
        try:
            key = (int(row["x"]), int(row["y"]), int(row["z"]))
            old_tile_map[key] = path
        except (KeyError, ValueError):
            continue

    new_tile_map: Dict[Tuple[int, int, int], Path] = {}
    new_tile_to_pid: Dict[Tuple[int, int, int], str] = {}
    for path in new_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
        pid = extract_pair_id(path)
        row = new_metadata.get(pid)
        if row is None:
            continue
        try:
            key = (int(row["x"]), int(row["y"]), int(row["z"]))
            new_tile_map[key] = path
            new_tile_to_pid[key] = pid
        except (KeyError, ValueError):
            continue

    common_tiles = sorted(set(old_tile_map) & set(new_tile_map))

    if pair_id is not None:
        normalized = pair_id.zfill(4)
        common_tiles = [t for t in common_tiles if new_tile_to_pid[t] == normalized]
    elif max_pairs is not None:
        common_tiles = common_tiles[:max_pairs]

    pairs = [
        ImagePair(
            pair_id=new_tile_to_pid[tile],
            old_path=old_tile_map[tile],
            new_path=new_tile_map[tile],
        )
        for tile in common_tiles
    ]
    print(f"[build_pairs] tile-coordinate matching: {len(pairs)} common pairs found.")
    return pairs


def build_pairs(
    old_dir: Path = config.OLD_DATASET_DIR,
    new_dir: Path = config.NEW_DATASET_DIR,
    pair_id: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> List[ImagePair]:
    old_metadata = load_metadata_map(str(old_dir))
    new_metadata = load_metadata_map(str(new_dir))

    if old_metadata and new_metadata:
        return _build_pairs_by_tile(old_dir, new_dir, old_metadata, new_metadata, pair_id, max_pairs)

    # Fallback: pair index matching (same bbox datasets)
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    old_files = sorted(
        [p for p in old_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes]
    )
    new_files = sorted(
        [p for p in new_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes]
    )

    old_map = {extract_pair_id(path): path for path in old_files}
    new_map = {extract_pair_id(path): path for path in new_files}
    common_ids = sorted(set(old_map) & set(new_map))

    if pair_id is not None:
        normalized = pair_id.zfill(4)
        common_ids = [pid for pid in common_ids if pid == normalized]
    elif max_pairs is not None:
        common_ids = common_ids[:max_pairs]

    return [
        ImagePair(pair_id=pid, old_path=old_map[pid], new_path=new_map[pid])
        for pid in common_ids
    ]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    pil_image = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def resize_to_match(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_height, ref_width = reference.shape[:2]
    if image.shape[:2] == (ref_height, ref_width):
        return image
    interpolation = cv2.INTER_AREA
    if image.shape[0] < ref_height or image.shape[1] < ref_width:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (ref_width, ref_height), interpolation=interpolation)


def image_size(path: Path) -> Tuple[int, int]:
    image = load_image(path)
    height, width = image.shape[:2]
    return width, height


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def bbox_center(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_distance(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.hypot(ax - bx, ay - by)


def crop_with_padding(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    left = max(0, int(x1) - padding)
    top = max(0, int(y1) - padding)
    right = min(width, int(x2) + padding)
    bottom = min(height, int(y2) + padding)
    return image[top:bottom, left:right]


def _prepare_compare_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def align_old_image_to_new(
    old_image: np.ndarray,
    new_image: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    resized_old = resize_to_match(old_image, new_image)
    if not config.ENABLE_PAIR_ALIGNMENT:
        return resized_old, False

    new_gray = _prepare_compare_gray(new_image)
    old_gray = _prepare_compare_gray(resized_old)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ALIGNMENT_ECC_ITERATIONS,
        config.ALIGNMENT_ECC_EPSILON,
    )

    for motion in (cv2.MOTION_EUCLIDEAN, cv2.MOTION_TRANSLATION):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(
                new_gray,
                old_gray,
                warp,
                motion,
                criteria,
                None,
                1,
            )
            aligned = cv2.warpAffine(
                resized_old,
                warp,
                (new_image.shape[1], new_image.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned, True
        except cv2.error:
            continue

    return resized_old, False


def compute_visual_change(
    old_image: np.ndarray,
    new_image: np.ndarray,
    bbox_old: Iterable[float],
    bbox_new: Iterable[float],
) -> float:
    x1 = min(float(bbox_old[0]), float(bbox_new[0]))
    y1 = min(float(bbox_old[1]), float(bbox_new[1]))
    x2 = max(float(bbox_old[2]), float(bbox_new[2]))
    y2 = max(float(bbox_old[3]), float(bbox_new[3]))

    old_crop = crop_with_padding(
        old_image,
        x1,
        y1,
        x2,
        y2,
        padding=config.VISUAL_COMPARE_PADDING,
    )
    new_crop = crop_with_padding(
        new_image,
        x1,
        y1,
        x2,
        y2,
        padding=config.VISUAL_COMPARE_PADDING,
    )
    if old_crop.size == 0 or new_crop.size == 0:
        return 0.0

    if old_crop.shape != new_crop.shape:
        new_crop = resize_to_match(new_crop, old_crop)

    old_gray = _prepare_compare_gray(old_crop)
    new_gray = _prepare_compare_gray(new_crop)

    intensity_diff = cv2.absdiff(old_gray, new_gray).mean()
    old_edges = cv2.Canny(old_gray, 40, 120)
    new_edges = cv2.Canny(new_gray, 40, 120)
    edge_diff = cv2.absdiff(old_edges, new_edges).mean()
    return float((0.7 * intensity_diff) + (0.3 * edge_diff))


def polygon_to_bbox(polygon) -> Tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = polygon.bounds
    return int(min_x), int(min_y), int(max_x), int(max_y)


def normalize_polygon(polygon):
    if polygon.is_empty:
        return polygon
    if polygon.is_valid:
        return polygon
    try:
        fixed = polygon.buffer(0)
        if not fixed.is_empty:
            return fixed
    except Exception:
        pass
    return polygon


def polygon_match_metrics(source_polygon, target_polygon) -> Dict[str, float]:
    source_polygon = normalize_polygon(source_polygon)
    target_polygon = normalize_polygon(target_polygon)
    if source_polygon.is_empty or target_polygon.is_empty:
        return {
            "direct_overlap": 0.0,
            "buffered_overlap": 0.0,
            "centroid_distance": float("inf"),
            "distance_limit": float(config.CENTROID_DISTANCE_MIN_PX),
            "area_change_ratio": 1.0,
        }

    try:
        intersection = source_polygon.intersection(target_polygon).area
    except Exception:
        intersection = 0.0
    direct_overlap = intersection / max(1.0, source_polygon.area)

    buffered_source = normalize_polygon(source_polygon.buffer(config.MATCH_BUFFER_PX))
    buffered_target = normalize_polygon(target_polygon.buffer(config.MATCH_BUFFER_PX))
    try:
        buffered_intersection = buffered_source.intersection(buffered_target).area
    except Exception:
        buffered_intersection = 0.0
    buffered_overlap = buffered_intersection / max(1.0, buffered_source.area)

    centroid_distance = float(source_polygon.centroid.distance(target_polygon.centroid))
    distance_limit = max(
        config.CENTROID_DISTANCE_MIN_PX,
        math.sqrt(max(source_polygon.area, target_polygon.area)) * config.CENTROID_DISTANCE_SCALE,
    )
    area_change_ratio = abs(source_polygon.area - target_polygon.area) / max(
        1.0,
        source_polygon.area,
        target_polygon.area,
    )

    return {
        "direct_overlap": float(direct_overlap),
        "buffered_overlap": float(buffered_overlap),
        "centroid_distance": centroid_distance,
        "distance_limit": float(distance_limit),
        "area_change_ratio": float(area_change_ratio),
    }


@lru_cache(maxsize=8)
def load_metadata_map(dataset_dir: str) -> Dict[str, Dict[str, str]]:
    metadata_csv = Path(dataset_dir) / "metadata.csv"
    if not metadata_csv.exists():
        return {}

    pair_map: Dict[str, Dict[str, str]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_key = str(row.get("pair_index", "")).zfill(4)
            if pair_key:
                pair_map[pair_key] = row
    return pair_map


def metadata_row_for_pair(dataset_dir: Path, pair_id: str) -> Optional[Dict[str, str]]:
    return load_metadata_map(str(dataset_dir)).get(str(pair_id).zfill(4))


def world_tile_to_lonlat(tile_x: float, tile_y: float, zoom: int) -> Tuple[float, float]:
    scale = 2 ** zoom
    lon = (tile_x / scale) * 360.0 - 180.0
    mercator = math.pi * (1.0 - (2.0 * tile_y / scale))
    lat = math.degrees(math.atan(math.sinh(mercator)))
    return lon, lat


def optional_geo_center(path: Path, x: float, y: float) -> Optional[Dict[str, float]]:
    pair_id = extract_pair_id(path)
    metadata = metadata_row_for_pair(path.parent, pair_id)
    if metadata is None:
        return None

    try:
        zoom = int(metadata["z"])
        tile_x = int(metadata["x"])
        tile_y = int(metadata["y"])
    except (KeyError, TypeError, ValueError):
        return None

    image_width, image_height = image_size(path)
    if image_width <= 0 or image_height <= 0:
        return None

    world_tile_x = tile_x + ((float(x) / image_width) * (image_width / BASE_TILE_SIZE_PX))
    world_tile_y = tile_y + ((float(y) / image_height) * (image_height / BASE_TILE_SIZE_PX))
    lon, lat = world_tile_to_lonlat(world_tile_x, world_tile_y, zoom)

    return {
        "lat": round(lat, 8),
        "lon": round(lon, 8),
        "z": zoom,
        "block_x": tile_x,
        "block_y": tile_y,
    }


def append_jsonl(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_rows_by_pair(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"]).zfill(4)].append(row)
    return grouped


def detected_row_to_geo_point(row: Dict) -> Optional[Dict[str, float]]:
    new_image = row.get("new_image")
    centroid = row.get("centroid_px", {})
    if not new_image or "x" not in centroid or "y" not in centroid:
        return None

    geo = optional_geo_center(
        Path(new_image),
        float(centroid["x"]),
        float(centroid["y"]),
    )
    if geo is None:
        return None

    return {
        "lat": float(geo["lat"]),
        "lon": float(geo["lon"]),
        "z": int(geo["z"]),
        "block_x": int(geo["block_x"]),
        "block_y": int(geo["block_y"]),
    }


def export_detected_points(
    jsonl_path: Path,
    txt_path: Path,
    csv_path: Path,
    geojson_path: Path,
) -> Tuple[int, int]:
    rows = load_jsonl(jsonl_path)
    point_rows = []
    features = []

    for row in rows:
        point = detected_row_to_geo_point(row)
        if point is None:
            continue

        export_row = {
            "pair_id": str(row.get("pair_id", "")).zfill(4),
            "status": row.get("status", ""),
            "lat": round(float(point["lat"]), 8),
            "lon": round(float(point["lon"]), 8),
            "confidence": row.get("confidence"),
            "clip_score": row.get("clip_score"),
            "visual_change_score": row.get("visual_change_score"),
            "pixel_x": row.get("centroid_px", {}).get("x"),
            "pixel_y": row.get("centroid_px", {}).get("y"),
            "z": point.get("z"),
            "block_x": point.get("block_x"),
            "block_y": point.get("block_y"),
            "old_image": row.get("old_image"),
            "new_image": row.get("new_image"),
        }
        point_rows.append(export_row)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [export_row["lon"], export_row["lat"]],
                },
                "properties": export_row,
            }
        )

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)

    with txt_path.open("w", encoding="utf-8") as handle:
        for row in point_rows:
            handle.write(
                f"{row['pair_id']}\t{row['lon']:.8f}\t{row['lat']:.8f}\t"
                f"{row['status']}\t{row['new_image']}\n"
            )

    fieldnames = [
        "pair_id",
        "status",
        "lat",
        "lon",
        "confidence",
        "clip_score",
        "visual_change_score",
        "pixel_x",
        "pixel_y",
        "z",
        "block_x",
        "block_y",
        "old_image",
        "new_image",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(point_rows)

    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(rows), len(point_rows)

