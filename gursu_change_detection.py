import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_dataset_path(value: str | None, fallback_year: int) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "dataset" / str(fallback_year)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Gursu change detection runner."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "detect-full",
            "detect-raw",
            "segment-full",
            "segment-raw",
            "clean",
        ],
        default="detect-full",
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model path. Best used with detection modes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count override.",
    )
    parser.add_argument(
        "--pair-id",
        default=None,
        help="Optional single pair id such as 0000.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pairs to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete full run by skipping completed pairs.",
    )
    parser.add_argument(
        "--year-from",
        type=int,
        default=2021,
        help="Before/source imagery year.",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=2026,
        help="After/target imagery year.",
    )
    parser.add_argument(
        "--old-dataset",
        default=None,
        help="Optional explicit before/source dataset directory.",
    )
    parser.add_argument(
        "--new-dataset",
        default=None,
        help="Optional explicit after/target dataset directory.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional output suffix. Defaults to YEAR_FROM_YEAR_TO.",
    )
    return parser


def apply_gursu_environment(
    year_from: int,
    year_to: int,
    run_tag: str | None,
    old_dataset: str | None = None,
    new_dataset: str | None = None,
):
    os.environ["CHANGE_PROJECT_PREFIX"] = "gursu"
    os.environ["GURSU_OLD_DATASET"] = str(_resolve_dataset_path(old_dataset, year_from))
    os.environ["GURSU_NEW_DATASET"] = str(_resolve_dataset_path(new_dataset, year_to))
    os.environ["GURSU_RUN_TAG"] = run_tag or f"{year_from}_{year_to}"


def main():
    args = build_parser().parse_args()
    apply_gursu_environment(
        args.year_from,
        args.year_to,
        args.run_tag,
        old_dataset=args.old_dataset,
        new_dataset=args.new_dataset,
    )

    if args.mode == "detect-full":
        from src.full.analyzer_cleaner_detection import main as run
        run(
            model_path=args.model,
            workers=args.workers,
            pair_id=args.pair_id,
            max_pairs=args.limit,
        )
        return

    if args.mode == "detect-raw":
        from src.analyzer import main as run
        run(
            model_path=args.model,
            workers=args.workers,
            pair_id=args.pair_id,
            max_pairs=args.limit,
            method="detect",
        )
        return

    if args.mode == "segment-full":
        from src.full.analyzer_cleaner_segmentation import main as run
        run(
            model_path=args.model,
            workers=args.workers,
            pair_id=args.pair_id,
            max_pairs=args.limit,
            resume=args.resume,
        )
        return

    if args.mode == "segment-raw":
        from src.analyzer import main as run
        run(
            model_path=args.model,
            workers=args.workers,
            pair_id=args.pair_id,
            max_pairs=args.limit,
            method="segment",
        )
        return

    if args.mode == "clean":
        from src.cleaner import main as run
        run()
        return

    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
