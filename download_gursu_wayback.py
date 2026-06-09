import argparse
import csv
import datetime as dt
import io
import json
import math
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageStat


WMTS_CAPABILITIES_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"
)
WAYBACK_TILE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/MapServer/tile/{release_id}/{z}/{y}/{x}"
)

# Gursu district study extent (min_lon, min_lat, max_lon, max_lat).
DEFAULT_BBOX = (29.131191, 40.198367, 29.306497, 40.339645)
PROJECT_PREFIX = "gursu"

NS = {
    "wmts": "https://www.opengis.net/wmts/1.0",
    "ows": "https://www.opengis.net/ows/1.1",
}


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return x, y


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download 512x512 Gursu imagery datasets from Esri World Imagery Wayback."
        )
    )
    parser.add_argument("--year-a", type=int, default=2021, help="First dataset year.")
    parser.add_argument("--year-b", type=int, default=2026, help="Second dataset year.")
    parser.add_argument(
        "--only-year",
        type=int,
        default=None,
        help="Download only a single year instead of both datasets.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=DEFAULT_BBOX,
        help="Bounding box in WGS84. Default is the Gursu study bbox.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="Web Mercator zoom level. Existing dataset scale suggests 18 is reasonable.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset",
        help="Output root directory. Year folders will be created underneath this path.",
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality when --image-format jpg.",
    )
    parser.add_argument(
        "--release-mode",
        choices=["latest", "earliest"],
        default="latest",
        help="Choose latest or earliest Wayback release inside the target year.",
    )
    parser.add_argument(
        "--release-date",
        type=str,
        default=None,
        help=(
            "Exact Wayback release date in YYYY-MM-DD. "
            "When set, overrides --release-mode for the selected year."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Optional custom output folder name for --only-year downloads. "
            "Example: 2025"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers.",
    )
    parser.add_argument(
        "--skip-dark",
        action="store_true",
        help="Skip 512 tiles whose mean brightness is very low.",
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=3.0,
        help="Mean brightness threshold used with --skip-dark.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retry count per 256 tile when a request fails.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=2.0,
        help="Base wait time in seconds between retries.",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=None,
        help="Optional max number of 512 tiles to create per year for testing.",
    )
    parser.add_argument(
        "--missing-from-year",
        type=int,
        default=None,
        help=(
            "When used with --only-year, compare against the other year's metadata "
            "and attempt only blocks missing from this year."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help=(
            "Disable resume mode and re-download everything from scratch. "
            "By default the script resumes from where it left off."
        ),
    )
    return parser.parse_args()


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "GursuWaybackDownloader/1.0",
        }
    )
    return session


def fetch_releases(session: requests.Session, timeout: int) -> List[Dict]:
    response = session.get(WMTS_CAPABILITIES_URL, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    releases: List[Dict] = []
    for layer in root.findall(".//wmts:Layer", NS):
        title = layer.findtext("ows:Title", default="", namespaces=NS)
        identifier = layer.findtext("ows:Identifier", default="", namespaces=NS)
        resource_url = layer.find("wmts:ResourceURL", NS)
        if not title or "Wayback" not in title or resource_url is None:
            continue

        title_match = re.search(r"Wayback\s+(\d{4}-\d{2}-\d{2})", title)
        template = resource_url.attrib.get("template", "")
        release_match = re.search(r"/tile/(\d+)/\{TileMatrix\}/\{TileRow\}/\{TileCol\}", template)
        if not title_match or not release_match:
            continue

        release_date = dt.date.fromisoformat(title_match.group(1))
        releases.append(
            {
                "title": title,
                "identifier": identifier,
                "date": release_date,
                "release_id": int(release_match.group(1)),
                "template": template,
            }
        )

    releases.sort(key=lambda item: item["date"])
    return releases


def pick_release(releases: List[Dict], year: int, mode: str) -> Dict:
    filtered = [item for item in releases if item["date"].year == year]
    if not filtered:
        raise ValueError(f"No Wayback release found for year {year}.")

    if mode == "earliest":
        return filtered[0]
    return filtered[-1]


def pick_release_by_date(releases: List[Dict], release_date: dt.date) -> Dict:
    for item in releases:
        if item["date"] == release_date:
            return item
    raise ValueError(f"No Wayback release found for date {release_date.isoformat()}.")


def build_block_grid(
    bbox: Tuple[float, float, float, float],
    zoom: int,
) -> List[Tuple[int, int, int]]:
    lon_min, lat_min, lon_max, lat_max = bbox
    x_min, y_max = lonlat_to_tile(lon_min, lat_min, zoom)
    x_max, y_min = lonlat_to_tile(lon_max, lat_max, zoom)

    x_start = min(x_min, x_max)
    x_end = max(x_min, x_max)
    y_start = min(y_min, y_max)
    y_end = max(y_min, y_max)

    blocks: List[Tuple[int, int, int]] = []
    for x in range(x_start, x_end + 1, 2):
        for y in range(y_start, y_end + 1, 2):
            blocks.append((zoom, x, y))
    return blocks


def download_tile(
    session: requests.Session,
    release_id: int,
    z: int,
    x: int,
    y: int,
    timeout: int,
    retries: int,
    retry_wait: float,
) -> Optional[bytes]:
    url = WAYBACK_TILE_URL.format(release_id=release_id, z=z, y=y, x=x)
    last_error = None
    for attempt in range(retries + 1):
        try:
            # Request the tile directly so Esri can return blank/sea tiles too.
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("image/"):
                return None
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= retries:
                raise
            sleep_seconds = retry_wait * (attempt + 1)
            print(
                f"retry tile z={z} x={x} y={y} release={release_id} "
                f"attempt={attempt + 1}/{retries} wait={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
    if last_error:
        raise last_error
    return None


def open_tile(tile_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(tile_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def stitch_512(tiles: List[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (512, 512))
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (256, 0))
    canvas.paste(tiles[2], (0, 256))
    canvas.paste(tiles[3], (256, 256))
    return canvas


def is_dark(image: Image.Image, threshold: float) -> bool:
    stat = ImageStat.Stat(image.convert("L"))
    return stat.mean[0] <= threshold


def save_image(image: Image.Image, output_path: Path, image_format: str, quality: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpg":
        image.save(output_path, format="JPEG", quality=quality, subsampling=0)
    else:
        image.save(output_path, format="PNG")


def process_block(
    release: Dict,
    block: Tuple[int, int, int],
    output_dir: Path,
    pair_index: int,
    image_format: str,
    quality: int,
    skip_dark: bool,
    dark_threshold: float,
    timeout: int,
    retries: int,
    retry_wait: float,
) -> Optional[Dict]:
    z, x, y = block
    ext = "jpg" if image_format == "jpg" else "png"
    file_name = f"{PROJECT_PREFIX}_{release['date'].year}_{pair_index:04d}.{ext}"
    output_path = output_dir / file_name
    if output_path.exists():
        return {
            "pair_index": pair_index,
            "file_name": file_name,
            "z": z,
            "x": x,
            "y": y,
            "release_id": release["release_id"],
            "release_date": release["date"].isoformat(),
            "status": "exists",
        }

    with get_session() as session:
        tile_coords = [
            (z, x, y),
            (z, x + 1, y),
            (z, x, y + 1),
            (z, x + 1, y + 1),
        ]
        images = []
        for _, tx, ty in tile_coords:
            tile_bytes = download_tile(
                session=session,
                release_id=release["release_id"],
                z=z,
                x=tx,
                y=ty,
                timeout=timeout,
                retries=retries,
                retry_wait=retry_wait,
            )
            if tile_bytes is None:
                return {
                    "pair_index": pair_index,
                    "file_name": file_name,
                    "z": z,
                    "x": x,
                    "y": y,
                    "release_id": release["release_id"],
                    "release_date": release["date"].isoformat(),
                    "status": "missing_tile",
                }
            images.append(open_tile(tile_bytes))

        stitched = stitch_512(images)
        if skip_dark and is_dark(stitched, dark_threshold):
            return {
                "pair_index": pair_index,
                "file_name": file_name,
                "z": z,
                "x": x,
                "y": y,
                "release_id": release["release_id"],
                "release_date": release["date"].isoformat(),
                "status": "skip_dark",
            }

        save_image(stitched, output_path, image_format=image_format, quality=quality)
        return {
            "pair_index": pair_index,
            "file_name": file_name,
            "z": z,
            "x": x,
            "y": y,
            "release_id": release["release_id"],
            "release_date": release["date"].isoformat(),
            "status": "saved",
        }


def write_metadata(output_dir: Path, rows: List[Dict], release: Dict):
    metadata_csv = output_dir / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_index",
                "file_name",
                "z",
                "x",
                "y",
                "release_id",
                "release_date",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    saved_count = sum(
        1
        for row in rows
        if row.get("status") in {"saved", "exists", "saved_missing_retry"}
    )
    skipped_count = sum(
        1
        for row in rows
        if row.get("status") in {"missing_tile", "skip_dark"}
    )
    summary = {
        "release_id": release["release_id"],
        "release_date": release["date"].isoformat(),
        "identifier": release["identifier"],
        "title": release["title"],
        "dataset_size": saved_count,
        "processed_count": len(rows),
        "skipped_count": skipped_count,
    }
    (output_dir / "release_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


_CSV_FIELDNAMES = [
    "pair_index", "file_name", "z", "x", "y",
    "release_id", "release_date", "status",
]


def download_dataset(
    release: Dict,
    blocks: List[Tuple[int, int, int]],
    output_dir: Path,
    image_format: str,
    quality: int,
    workers: int,
    skip_dark: bool,
    dark_threshold: float,
    timeout: int,
    retries: int,
    retry_wait: float,
    pair_limit: Optional[int],
    resume: bool = True,
) -> List[Dict]:
    metadata_csv = output_dir / "metadata.csv"
    selected_blocks = blocks[:pair_limit] if pair_limit else blocks
    year = release["date"].year
    successful_statuses = {"saved", "exists", "saved_missing_retry"}

    # --- Resume: load already-completed tiles from existing metadata.csv ---
    existing_rows: List[Dict] = []
    completed_keys: set = set()
    if resume and metadata_csv.exists():
        raw_existing_rows = read_metadata_rows(metadata_csv)
        existing_rows = [
            r
            for r in raw_existing_rows
            if r.get("status") in successful_statuses
        ]
        completed_keys = {
            (int(r["z"]), int(r["x"]), int(r["y"]))
            for r in existing_rows
        }
        if raw_existing_rows:
            print(
                f"[{year}] Resume: {len(completed_keys)} saved tiles already done, "
                f"{len(selected_blocks) - len(completed_keys)} remaining."
            )

    # Keep original pair_index values so filenames are stable across resumed runs.
    pending: List[Tuple[int, Tuple[int, int, int]]] = [
        (pair_index, block)
        for pair_index, block in enumerate(selected_blocks)
        if (block[0], block[1], block[2]) not in completed_keys
    ]

    if not pending:
        print(f"[{year}] All {len(selected_blocks)} tiles already complete.")
        write_metadata(output_dir, existing_rows, release)
        return existing_rows

    print(f"[{year}] Downloading {len(pending)} tiles ({len(existing_rows)} already done).")

    # --- Thread-safe incremental writer ---
    # Write the CSV header once upfront if the file doesn't exist yet
    csv_lock = threading.Lock()
    all_rows: List[Dict] = list(existing_rows)

    if not metadata_csv.exists():
        with metadata_csv.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES).writeheader()

    def _append_row(row: Dict) -> None:
        """Append one row to metadata.csv immediately (thread-safe)."""
        with csv_lock:
            all_rows.append(row)
            with metadata_csv.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES).writerow(row)

    # --- Thread pool download ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_block,
                release,
                block,
                output_dir,
                pair_index,
                image_format,
                quality,
                skip_dark,
                dark_threshold,
                timeout,
                retries,
                retry_wait,
            ): (pair_index, block)
            for pair_index, block in pending
        }

        done_count = len(existing_rows)
        total_count = len(selected_blocks)
        for future in as_completed(futures):
            pair_index, block = futures[future]
            try:
                result = future.result()
                if result is not None:
                    _append_row(result)
                    done_count += 1
                    if (
                        done_count % 100 == 0
                        or done_count == total_count
                        or result["status"] in {"missing_tile", "skip_dark"}
                    ):
                        print(
                            f"  [{year}] progress: {done_count}/{total_count} "
                            f"last_status={result['status']} pair={pair_index:04d}"
                        )
            except Exception as exc:
                z, x, y = block
                print(f"error pair={pair_index:04d} block={z}_{x}_{y}: {exc}")

    # Final sorted rewrite so the CSV is always clean and ordered
    with csv_lock:
        all_rows.sort(key=lambda item: int(item["pair_index"]))
        write_metadata(output_dir, all_rows, release)

    return all_rows


def read_metadata_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_blocks_from_metadata(rows: List[Dict]) -> List[Tuple[int, int, int]]:
    return [(int(r["z"]), int(r["x"]), int(r["y"])) for r in rows]


def next_pair_index(rows: List[Dict]) -> int:
    if not rows:
        return 0
    return max(int(r["pair_index"]) for r in rows) + 1


def write_metadata_rows(output_dir: Path, rows: List[Dict], release: Dict):
    metadata_csv = output_dir / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "release_id": release["release_id"],
        "release_date": release["date"].isoformat(),
        "identifier": release["identifier"],
        "title": release["title"],
        "dataset_size": len(rows),
    }
    (output_dir / "release_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_saved_x_columns(output_root: Path, year: int) -> set:
    metadata_csv = output_root / str(year) / "metadata.csv"
    if not metadata_csv.exists():
        return set()
    rows = read_metadata_rows(metadata_csv)
    return {
        int(r["x"])
        for r in rows
        if r.get("status") in {"saved", "exists", "saved_missing_retry"}
    }


def probe_block_has_full_coverage(
    session: requests.Session,
    release_id: int,
    block: Tuple[int, int, int],
    timeout: int,
    retries: int,
    retry_wait: float,
) -> bool:
    z, x, y = block
    for tx, ty in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]:
        tile_bytes = download_tile(
            session=session,
            release_id=release_id,
            z=z,
            x=tx,
            y=ty,
            timeout=timeout,
            retries=retries,
            retry_wait=retry_wait,
        )
        if tile_bytes is None:
            return False
    return True


def prioritize_pending_blocks(
    release: Dict,
    pending: List[Tuple[int, Tuple[int, int, int]]],
    output_root: Path,
    timeout: int,
    retries: int,
    retry_wait: float,
) -> List[Tuple[int, Tuple[int, int, int]]]:
    if not pending:
        return pending

    year = release["date"].year
    for prior_year in (year - 1, year - 2):
        saved_x_columns = find_saved_x_columns(output_root, prior_year)
        if saved_x_columns:
            prioritized = sorted(
                pending,
                key=lambda item: (
                    0 if item[1][1] in saved_x_columns else 1,
                    item[0],
                ),
            )
            print(
                f"[{year}] Prioritized using Gursu {prior_year} coverage "
                f"({len(saved_x_columns)} saved columns)."
            )
            return prioritized

    y_values = sorted({block[2] for _, block in pending})
    sample_y = y_values[len(y_values) // 2]

    column_first_block = {}
    for pair_index, block in pending:
        x = block[1]
        if x not in column_first_block:
            column_first_block[x] = (pair_index, (block[0], x, sample_y))

    available_columns = set()
    with get_session() as session:
        for x in sorted(column_first_block):
            _, sample_block = column_first_block[x]
            if probe_block_has_full_coverage(
                session=session,
                release_id=release["release_id"],
                block=sample_block,
                timeout=timeout,
                retries=0,
                retry_wait=retry_wait,
            ):
                available_columns.add(x)

    prioritized = sorted(
        pending,
        key=lambda item: (
            0 if item[1][1] in available_columns else 1,
            item[0],
        ),
    )

    if available_columns:
        print(
            f"[{release['date'].year}] Prioritized {len(available_columns)} likely-covered "
            f"columns first out of {len(column_first_block)} total columns."
        )
    else:
        print(
            f"[{release['date'].year}] No covered sample columns found during pre-scan; "
            "keeping default order."
        )
    return prioritized


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.time()
    session = get_session()
    releases = fetch_releases(session, timeout=args.timeout)

    if args.only_year is not None:
        if args.release_date is not None:
            release_date = dt.date.fromisoformat(args.release_date)
            if release_date.year != args.only_year:
                raise ValueError(
                    f"--release-date year ({release_date.year}) must match --only-year ({args.only_year})."
                )
            release = pick_release_by_date(releases, release_date)
        else:
            release = pick_release(releases, args.only_year, args.release_mode)
        dataset_name = args.dataset_name or str(args.only_year)
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        existing_rows = read_metadata_rows(dataset_dir / "metadata.csv")

        if args.missing_from_year is not None:
            other_dir = output_root / str(args.missing_from_year)
            current_rows = list(existing_rows)
            other_rows = read_metadata_rows(other_dir / "metadata.csv")
            current_keys = {f"{r['z']}_{r['x']}_{r['y']}" for r in current_rows}
            missing_rows = [
                r for r in other_rows if f"{r['z']}_{r['x']}_{r['y']}" not in current_keys
            ]
            selected_missing_rows = missing_rows[: args.pair_limit] if args.pair_limit else missing_rows
        else:
            blocks = build_block_grid(tuple(args.bbox), args.zoom)

        print("Selected single-year release")
        print(
            json.dumps(
                {
                    "year": args.only_year,
                    "release_date": release["date"].isoformat(),
                    "release_id": release["release_id"],
                    "zoom": args.zoom,
                    "bbox": args.bbox,
                    "block_count": (
                        len(selected_missing_rows)
                        if args.missing_from_year is not None
                        else len(blocks if args.pair_limit is None else blocks[: args.pair_limit])
                    ),
                    "missing_from_year": args.missing_from_year,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        added_rows: List[Dict] = []
        total_rows: List[Dict]
        if args.missing_from_year is not None:
            merged_rows = list(existing_rows)
            pair_index = next_pair_index(existing_rows)
            session = get_session()
            ext = "jpg" if args.image_format == "jpg" else "png"
            for source_row in selected_missing_rows:
                z = int(source_row["z"])
                x = int(source_row["x"])
                y = int(source_row["y"])
                print(f"trying missing block={z}_{x}_{y}")
                try:
                    tile_coords = [
                        (z, x, y),
                        (z, x + 1, y),
                        (z, x, y + 1),
                        (z, x + 1, y + 1),
                    ]
                    images = []
                    ok = True
                    for _, tx, ty in tile_coords:
                        tile_bytes = download_tile(
                            session=session,
                            release_id=release["release_id"],
                            z=z,
                            x=tx,
                            y=ty,
                            timeout=args.timeout,
                            retries=args.retries,
                            retry_wait=args.retry_wait,
                        )
                        if tile_bytes is None:
                            ok = False
                            break
                        images.append(open_tile(tile_bytes))

                    if not ok:
                        print(f"skip pair={source_row['pair_index']} block={z}_{x}_{y}")
                        continue

                    stitched = stitch_512(images)
                    if args.skip_dark and is_dark(stitched, args.dark_threshold):
                        print(f"skip-dark pair={source_row['pair_index']} block={z}_{x}_{y}")
                        continue

                    file_name = f"{PROJECT_PREFIX}_{args.only_year}_{pair_index:04d}.{ext}"
                    save_image(
                        stitched,
                        dataset_dir / file_name,
                        image_format=args.image_format,
                        quality=args.quality,
                    )
                    row = {
                        "pair_index": pair_index,
                        "file_name": file_name,
                        "z": z,
                        "x": x,
                        "y": y,
                        "release_id": release["release_id"],
                        "release_date": release["date"].isoformat(),
                        "status": "saved_missing_retry",
                    }
                    added_rows.append(row)
                    merged_rows.append(row)
                    pair_index += 1
                except Exception as exc:
                    print(f"error pair={source_row['pair_index']} block={z}_{x}_{y}: {exc}")

            write_metadata_rows(dataset_dir, merged_rows, release)
            total_rows = merged_rows
        else:
            total_rows = download_dataset(
                release=release,
                blocks=blocks,
                output_dir=dataset_dir,
                image_format=args.image_format,
                quality=args.quality,
                workers=args.workers,
                skip_dark=args.skip_dark,
                dark_threshold=args.dark_threshold,
                timeout=args.timeout,
                retries=args.retries,
                retry_wait=args.retry_wait,
                pair_limit=args.pair_limit,
                resume=not args.no_resume,
            )
        summary = {
            "dataset": {
                "dir": str(dataset_dir),
                "count": len(total_rows),
                "release_date": release["date"].isoformat(),
                "release_id": release["release_id"],
            },
            "elapsed_seconds": round(time.time() - start, 2),
            "missing_from_year": args.missing_from_year,
        }
        if args.missing_from_year is not None:
            summary["added_count"] = len(added_rows)
        (output_root / f"{PROJECT_PREFIX}_{args.only_year}_single_download_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nDownload complete")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    release_a = pick_release(releases, args.year_a, args.release_mode)
    release_b = pick_release(releases, args.year_b, args.release_mode)
    blocks = build_block_grid(tuple(args.bbox), args.zoom)

    dataset_a_dir = output_root / str(args.year_a)
    dataset_b_dir = output_root / str(args.year_b)
    dataset_a_dir.mkdir(parents=True, exist_ok=True)
    dataset_b_dir.mkdir(parents=True, exist_ok=True)

    print("Selected releases")
    print(
        json.dumps(
            {
                "year_a": {
                    "release_date": release_a["date"].isoformat(),
                    "release_id": release_a["release_id"],
                },
                "year_b": {
                    "release_date": release_b["date"].isoformat(),
                    "release_id": release_b["release_id"],
                },
                "zoom": args.zoom,
                "bbox": args.bbox,
                "block_count": len(blocks if args.pair_limit is None else blocks[: args.pair_limit]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    rows_a = download_dataset(
        release=release_a,
        blocks=blocks,
        output_dir=dataset_a_dir,
        image_format=args.image_format,
        quality=args.quality,
        workers=args.workers,
        skip_dark=args.skip_dark,
        dark_threshold=args.dark_threshold,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        pair_limit=args.pair_limit,
        resume=not args.no_resume,
    )
    rows_b = download_dataset(
        release=release_b,
        blocks=blocks,
        output_dir=dataset_b_dir,
        image_format=args.image_format,
        quality=args.quality,
        workers=args.workers,
        skip_dark=args.skip_dark,
        dark_threshold=args.dark_threshold,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        pair_limit=args.pair_limit,
        resume=not args.no_resume,
    )

    summary = {
        "dataset_a": {
            "dir": str(dataset_a_dir),
            "count": len(rows_a),
            "release_date": release_a["date"].isoformat(),
            "release_id": release_a["release_id"],
        },
        "dataset_b": {
            "dir": str(dataset_b_dir),
            "count": len(rows_b),
            "release_date": release_b["date"].isoformat(),
            "release_id": release_b["release_id"],
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (output_root / f"{PROJECT_PREFIX}_wayback_download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDownload complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


