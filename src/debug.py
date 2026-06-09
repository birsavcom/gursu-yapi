import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.pair_utils import build_pairs, get_device, load_image


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect one image pair.")
    parser.add_argument("--pair-id", required=True, help="Pair id like 0000.")
    parser.add_argument(
        "--method",
        choices=["detect", "segment"],
        default="detect",
    )
    parser.add_argument("--model", default=None, help="Override model path.")
    return parser


def main():
    args = build_parser().parse_args()
    pairs = build_pairs(pair_id=args.pair_id)
    if not pairs:
        raise SystemExit("Pair not found.")

    pair = pairs[0]
    device = get_device()
    if args.method == "detect":
        model_path = Path(args.model) if args.model else config.DETECTION_MODEL_PATH
    else:
        model_path = Path(args.model) if args.model else config.SEGMENTATION_MODEL_PATH

    model = YOLO(str(model_path)).to(device)
    old_image = load_image(pair.old_path)
    new_image = load_image(pair.new_path)

    old_result = model.predict(old_image, conf=config.YOLO_CONF_THRESHOLD, device=device, verbose=False)[0]
    new_result = model.predict(new_image, conf=config.YOLO_CONF_THRESHOLD, device=device, verbose=False)[0]

    old_plot = old_result.plot(labels=True, conf=True)
    new_plot = new_result.plot(labels=True, conf=True)
    combined = np.hstack([old_plot, new_plot])

    out_path = config.DEBUG_DIR / f"debug_pair_{pair.pair_id}_{args.method}.jpg"
    cv2.imwrite(str(out_path), combined)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

