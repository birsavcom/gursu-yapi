import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("CHANGE_MODELS_DIR", str(PROJECT_ROOT / "models")))
DATASET_ROOT = PROJECT_ROOT / "dataset"
PROJECT_PREFIX = os.getenv("CHANGE_PROJECT_PREFIX", "gursu").strip() or "gursu"


def _dataset_dir_name(year: int) -> str:
    return str(year)


LEGACY_OLD_DATASET_DIR = DATASET_ROOT / _dataset_dir_name(2021)
LEGACY_NEW_DATASET_DIR = DATASET_ROOT / _dataset_dir_name(2026)

_default_old_dataset = LEGACY_OLD_DATASET_DIR
_default_new_dataset = LEGACY_NEW_DATASET_DIR

OLD_DATASET_DIR = Path(os.getenv("GURSU_OLD_DATASET", str(_default_old_dataset)))
NEW_DATASET_DIR = Path(os.getenv("GURSU_NEW_DATASET", str(_default_new_dataset)))
RUN_TAG = os.getenv("GURSU_RUN_TAG", "").strip()


def _suffix_tag(path: Path) -> Path:
    if not RUN_TAG:
        return path
    safe_tag = RUN_TAG.replace("-", "_").replace(" ", "_")
    return path.with_name(f"{path.stem}_{safe_tag}{path.suffix}")

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_MASKS_DIR = _suffix_tag(RESULTS_DIR / "masks_raw")
VERIFIED_MASKS_DIR = _suffix_tag(RESULTS_DIR / "masks_verified")
RAW_SEG_MASKS_DIR = _suffix_tag(RESULTS_DIR / "masks_segmentation_raw")
VERIFIED_SEG_MASKS_DIR = _suffix_tag(RESULTS_DIR / "masks_segmentation_verified")
ALL_PAIRS_DIR = _suffix_tag(RESULTS_DIR / "pairs_all")
SEGMENTATION_ALL_PAIRS_DIR = _suffix_tag(RESULTS_DIR / "pairs_segmentation_all")
DETECTION_ERROR_LOG = RESULTS_DIR / "detect_errors.log"
RAW_DETECTION_JSONL = _suffix_tag(RESULTS_DIR / f"{PROJECT_PREFIX}_change_raw_detection.jsonl")
VERIFIED_DETECTION_JSONL = _suffix_tag(RESULTS_DIR / f"{PROJECT_PREFIX}_change_verified_detection.jsonl")
RAW_SEGMENTATION_JSONL = _suffix_tag(RESULTS_DIR / f"{PROJECT_PREFIX}_change_raw_segmentation.jsonl")
VERIFIED_SEGMENTATION_JSONL = _suffix_tag(RESULTS_DIR / f"{PROJECT_PREFIX}_change_verified_segmentation.jsonl")

DEBUG_DIR = RESULTS_DIR / "debug"
SEGMENTATION_PROCESSED_PAIRS_TXT = _suffix_tag(DEBUG_DIR / "segmentation_processed_pairs.txt")
VERIFIED_SEGMENTATION_POINTS_TXT = _suffix_tag(DEBUG_DIR / f"{PROJECT_PREFIX}_change_verified_segmentation_points.txt")
VERIFIED_SEGMENTATION_POINTS_CSV = _suffix_tag(DEBUG_DIR / f"{PROJECT_PREFIX}_change_verified_segmentation_points.csv")
VERIFIED_SEGMENTATION_POINTS_GEOJSON = _suffix_tag(DEBUG_DIR / f"{PROJECT_PREFIX}_change_verified_segmentation_points.geojson")

DETECTION_MODEL_PATH = MODELS_DIR / "detection_best.pt"
SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_best.pt"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "best.pt"

IMAGE_SIZE = 512
YOLO_CONF_THRESHOLD = 0.30
DIST_THRESHOLD = 24.0
BUFFER_SIZE_PX = 15.0
MATCH_BUFFER_PX = 18.0
NEW_BUILDING_OVERLAP_THRESHOLD = 0.25
CHANGE_OVERLAP_MIN = 0.25
BUFFERED_OVERLAP_MIN = 0.12
CENTROID_DISTANCE_MIN_PX = 28.0
CENTROID_DISTANCE_SCALE = 0.35
AREA_CHANGE_RATIO = 0.30
VISUAL_CHANGE_THRESHOLD = 20.0
NEW_BUILDING_VISUAL_THRESHOLD = 20.0
VISUAL_COMPARE_PADDING = 12
MIN_AREA = 80.0
MAX_AREA = 190000.0

ADJACENT_YEAR_BUFFER_SIZE_PX = 22.0
ADJACENT_YEAR_CHANGE_OVERLAP_MIN = 0.12
ADJACENT_YEAR_BUFFERED_OVERLAP_MIN = 0.05
ADJACENT_YEAR_DISTANCE_LIMIT_SCALE = 1.6
ADJACENT_YEAR_VISUAL_THRESHOLD = 28.0
ADJACENT_YEAR_CLIP_CONF_THRESHOLD = 0.50
ADJACENT_YEAR_OLD_YOLO_CONF_THRESHOLD = 0.20

LONG_GAP_BUFFER_SIZE_PX = 26.0
LONG_GAP_CHANGE_OVERLAP_MIN = 0.10
LONG_GAP_BUFFERED_OVERLAP_MIN = 0.04
LONG_GAP_DISTANCE_LIMIT_SCALE = 1.75
LONG_GAP_VISUAL_THRESHOLD = 28.0
LONG_GAP_CLIP_CONF_THRESHOLD = 0.50
LONG_GAP_OLD_YOLO_CONF_THRESHOLD = 0.20
LONG_GAP_AREA_CHANGE_RATIO = 0.55

ENABLE_PAIR_ALIGNMENT = True
ALIGNMENT_ECC_ITERATIONS = 80
ALIGNMENT_ECC_EPSILON = 1e-4

DEFAULT_MAX_WORKERS = 1

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_CONF_THRESHOLD = 0.45
CLIP_PADDING = 70

POSITIVE_PROMPTS = [
    "aerial view of a residential house with a roof",
    "satellite image of a red tiled roof building",
    "aerial view of a grey metal roof structure",
    "satellite view of an industrial factory or warehouse",
    "rectangular building with a metallic roof",
    "aerial view of a bright white building roof",
    "flat concrete roof of a building",
    "aerial view of a black or dark grey roof house",
    "building roof with solar panels",
    "dark rectangular construction on the ground",
    "aerial view of a greenhouse structure",
    "transparent or plastic roof farm building",
    "agricultural barn or shed structure",
]

NEGATIVE_PROMPTS = [
    "aerial view of a green forest or trees",
    "satellite image of a green agricultural field",
    "texture of grass and vegetation",
    "aerial view of dry yellow grass or hay fields",
    "satellite image of harvested golden wheat fields",
    "texture of dry yellowish agricultural land",
    "aerial view of brown plowed soil with parallel furrows",
    "satellite image of empty brown dirt land",
    "bare brown soil without any buildings",
    "aerial view of an asphalt road or intersection",
    "concrete pavement or parking lot ground",
    "aerial view of a swimming pool or water",
    "dark shadow of a tree",
    "white clouds covering the ground",
    "shadow of a cloud on the earth",
]

