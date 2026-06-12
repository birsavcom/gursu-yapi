"""Monthly and annual Gursu update orchestrator.

The script is safe before the configured activation date: when the server timer
runs early, it returns ``waiting``. After activation it checks Esri Wayback
releases, downloads only the required temporary datasets, runs the existing
segmentation pipeline with resume support, classifies results, updates the map
JSON, rebuilds ``index.html``, and removes temporary datasets after success.
"""

from __future__ import annotations

import argparse
import calendar
import json
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "automation" / "config.json"
TIMEZONE = ZoneInfo("Europe/Istanbul")
MONTH_NAMES_TR = {
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık",
}


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_period(period: str) -> tuple[int, int]:
    year_text, month_text = period.split("-", 1)
    return int(year_text), int(month_text)


def period_end(period: str) -> date:
    year, month = parse_period(period)
    return date(year, month, calendar.monthrange(year, month)[1])


def previous_period(period: str) -> str:
    year, month = parse_period(period)
    month -= 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:04d}-{month:02d}"


def current_month_period(now: datetime) -> tuple[str, str]:
    current = f"{now.year:04d}-{now.month:02d}"
    return previous_period(current), current


def period_month_label(period: str) -> str:
    year, month = parse_period(period)
    return f"{MONTH_NAMES_TR[month]} {year}"


def period_label(from_period: str, to_period: str) -> str:
    return f"{period_month_label(from_period)} - {period_month_label(to_period)}"


def monthly_run_key(from_period: str, to_period: str) -> str:
    return f"{from_period.replace('-', '_')}_{to_period.replace('-', '_')}"


def monthly_run_id(from_period: str, to_period: str) -> str:
    return f"monthly_{monthly_run_key(from_period, to_period)}"


def annual_run_id(baseline_year: int, target_year: int) -> str:
    return f"annual_{baseline_year}_{target_year}"


def annual_run_key(baseline_year: int, target_year: int) -> str:
    return f"{baseline_year}_{target_year}"


def ensure_monthly_file(path: Path) -> dict:
    default = {
        "version": 1,
        "district": "gursu",
        "description": "Initial and automated incremental detections for the Gursu map.",
        "runs": [],
    }
    payload = read_json(path, default)
    payload.setdefault("version", 1)
    payload.setdefault("district", "gursu")
    payload.setdefault("runs", [])
    write_json(path, payload)
    return payload


def run_exists(payload: dict, run_id: str) -> bool:
    return any(run.get("id") == run_id for run in payload.get("runs", []))


def upsert_run(payload: dict, new_run: dict) -> None:
    runs = payload.setdefault("runs", [])
    for index, run in enumerate(runs):
        if run.get("id") == new_run.get("id"):
            runs[index] = new_run
            return
    runs.append(new_run)


def logs_root_from_config(config: dict) -> Path:
    return ROOT / config.get("paths", {}).get("logs_root", "logs/automation")


def command_python(config: dict) -> str:
    configured = config.get("pipeline", {}).get("python")
    return configured or sys.executable


def stream_command(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[command] " + " ".join(str(part) for part in args))
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[command] " + " ".join(str(part) for part in args) + "\n")
        process = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(args)}")


def fetch_releases(config: dict) -> list[dict]:
    from download_gursu_wayback import fetch_releases as fetch
    from download_gursu_wayback import get_session

    timeout = int(config.get("pipeline", {}).get("timeout", 30))
    session = get_session()
    return fetch(session, timeout=timeout)


def latest_release_in_year(releases: list[dict], year: int) -> dict:
    candidates = [release for release in releases if release["date"].year == year]
    if not candidates:
        raise RuntimeError(f"No Esri Wayback release found for {year}.")
    return candidates[-1]


def latest_release_on_or_before(releases: list[dict], period: str) -> dict | None:
    cutoff = period_end(period)
    candidates = [release for release in releases if release["date"] <= cutoff]
    return candidates[-1] if candidates else None


def dataset_dir_for_period(period: str) -> Path:
    return ROOT / "dataset" / "automation" / period


def baseline_dataset_dir(year: int) -> Path:
    return ROOT / "dataset" / "baseline" / str(year)


def release_info_matches(dataset_dir: Path, release: dict) -> bool:
    info_path = dataset_dir / "release_info.json"
    if not info_path.exists():
        return False
    try:
        info = read_json(info_path, {})
    except Exception:
        return False
    return str(info.get("release_id")) == str(release.get("release_id"))


def bbox_args(config: dict) -> list[str]:
    bbox = config["bbox"]
    return [
        str(bbox["min_lon"]),
        str(bbox["min_lat"]),
        str(bbox["max_lon"]),
        str(bbox["max_lat"]),
    ]


def download_dataset(config: dict, release: dict, dataset_dir: Path, dataset_name: str, logs_root: Path) -> None:
    if release_info_matches(dataset_dir, release):
        print(f"[download] {dataset_name} already has release {release['date'].isoformat()}.")
        return

    pipeline = config.get("pipeline", {})
    imagery = config.get("imagery", {})
    args = [
        command_python(config),
        pipeline.get("download_script", "download_gursu_wayback.py"),
        "--only-year",
        str(release["date"].year),
        "--release-date",
        release["date"].isoformat(),
        "--dataset-name",
        dataset_name.replace("\\", "/"),
        "--output-root",
        "dataset",
        "--bbox",
        *bbox_args(config),
        "--zoom",
        str(imagery.get("zoom", 18)),
        "--image-format",
        pipeline.get("image_format", "png"),
        "--workers",
        str(pipeline.get("download_workers", 8)),
    ]
    stream_command(args, logs_root / f"download_{dataset_name.replace('/', '_')}.log")


def run_detection(
    config: dict,
    run_tag: str,
    old_dir: Path,
    new_dir: Path,
    year_from: int,
    year_to: int,
    logs_root: Path,
) -> None:
    pipeline = config.get("pipeline", {})
    args = [
        command_python(config),
        pipeline.get("detection_script", "gursu_change_detection.py"),
        "--mode",
        pipeline.get("detection_mode", "segment-full"),
        "--year-from",
        str(year_from),
        "--year-to",
        str(year_to),
        "--run-tag",
        run_tag,
        "--old-dataset",
        str(old_dir),
        "--new-dataset",
        str(new_dir),
        "--resume",
    ]
    if pipeline.get("model"):
        args.extend(["--model", pipeline["model"]])
    if pipeline.get("workers") is not None:
        args.extend(["--workers", str(pipeline["workers"])])
    stream_command(args, logs_root / f"detect_{run_tag}.log")


def collect_items(run: dict) -> list[dict]:
    from build_gursu_html import read_detections_for_run

    return read_detections_for_run(run)


def rebuild_html(config: dict, logs_root: Path) -> None:
    stream_command(
        [command_python(config), "build_gursu_html.py"],
        logs_root / "build_gursu_html.log",
    )


def cleanup_dataset(path: Path, config: dict) -> None:
    incremental = config.get("incremental", {})
    if not incremental.get("cleanup_temp_dataset_after_success", True):
        return
    resolved = path.resolve()
    allowed_root = (ROOT / "dataset" / "automation").resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        return
    shutil.rmtree(resolved, ignore_errors=True)


def append_monthly_run(
    config: dict,
    payload: dict,
    from_period: str,
    to_period: str,
    from_release: dict,
    to_release: dict,
) -> dict:
    from_year, _ = parse_period(from_period)
    to_year, _ = parse_period(to_period)
    run_tag = monthly_run_key(from_period, to_period)
    run = {
        "id": monthly_run_id(from_period, to_period),
        "key": run_tag,
        "label": period_label(from_period, to_period),
        "type": "monthly",
        "period_type": "monthly",
        "year_from": from_year,
        "year_to": to_year,
        "display_year": to_year,
        "from_period": from_period,
        "to_period": to_period,
        "from_period_label": period_month_label(from_period),
        "to_period_label": period_month_label(to_period),
        "from_release_date": from_release["date"].isoformat(),
        "to_release_date": to_release["date"].isoformat(),
        "from_release_id": from_release["release_id"],
        "to_release_id": to_release["release_id"],
    }
    run["items"] = collect_items(run)
    upsert_run(payload, run)
    return run


def append_annual_run(
    config: dict,
    payload: dict,
    baseline_year: int,
    target_period: str,
    baseline_release: dict,
    target_release: dict,
) -> dict:
    target_year, _ = parse_period(target_period)
    run_tag = annual_run_key(baseline_year, target_year)
    run = {
        "id": annual_run_id(baseline_year, target_year),
        "key": run_tag,
        "label": f"{baseline_year}-{target_year} Genel Tespit",
        "type": "annual",
        "period_type": "annual",
        "year_from": baseline_year,
        "year_to": target_year,
        "display_year": target_year,
        "from_period": str(baseline_year),
        "to_period": target_period,
        "from_period_label": str(baseline_year),
        "to_period_label": period_month_label(target_period),
        "from_release_date": baseline_release["date"].isoformat(),
        "to_release_date": target_release["date"].isoformat(),
        "from_release_id": baseline_release["release_id"],
        "to_release_id": target_release["release_id"],
    }
    run["items"] = collect_items(run)
    upsert_run(payload, run)
    return run


def load_state(config: dict) -> tuple[Path, dict]:
    state_path = ROOT / config.get("paths", {}).get("state", "automation/state.json")
    return state_path, read_json(state_path, {})


def derive_from_period(config: dict, state: dict, payload: dict) -> str:
    last_period = state.get("last_processed_period") or {}
    if last_period.get("to"):
        return last_period["to"]

    monthly_runs = [run for run in payload.get("runs", []) if run.get("type") == "monthly" and run.get("to_period")]
    if monthly_runs:
        monthly_runs.sort(key=lambda run: run.get("to_period", ""))
        return monthly_runs[-1]["to_period"]

    return config.get("incremental", {}).get("first_compare_from", "2026-07")


def derive_to_period(config: dict, state: dict, now: datetime) -> str:
    if not (state.get("last_processed_period") or {}):
        return config.get("incremental", {}).get("first_compare_to", "2026-08")
    return current_month_period(now)[1]


def run_pipeline(config: dict, now: datetime) -> dict:
    monthly_path = ROOT / config.get("paths", {}).get("monthly_detections", "data/monthly_detections.json")
    payload = ensure_monthly_file(monthly_path)
    state_path, state = load_state(config)
    logs_root = logs_root_from_config(config)

    from_period = derive_from_period(config, state, payload)
    to_period = derive_to_period(config, state, now)
    if from_period == to_period:
        return {"status": "already_current", "period": to_period}

    releases = fetch_releases(config)
    from_release = latest_release_on_or_before(releases, from_period)
    to_release = latest_release_on_or_before(releases, to_period)
    if not from_release or not to_release:
        return {"status": "missing_release", "from_period": from_period, "to_period": to_period}
    if to_release["date"] <= from_release["date"]:
        return {
            "status": "no_new_release",
            "from_period": from_period,
            "to_period": to_period,
            "from_release_date": from_release["date"].isoformat(),
            "to_release_date": to_release["date"].isoformat(),
        }

    from_dir = dataset_dir_for_period(from_period)
    to_dir = dataset_dir_for_period(to_period)
    download_dataset(config, from_release, from_dir, f"automation/{from_period}", logs_root)
    download_dataset(config, to_release, to_dir, f"automation/{to_period}", logs_root)

    processed_runs: list[dict] = []
    policy = config.get("automation_policy", {})
    baseline_year = int(policy.get("annual_baseline_year", 2021))
    to_year, _ = parse_period(to_period)
    initial_to_year = int(config.get("initial_inventory", {}).get("to_year", 2026))

    if policy.get("create_annual_rollup_on_new_year", True) and to_year > initial_to_year:
        annual_id = annual_run_id(baseline_year, to_year)
        if not run_exists(payload, annual_id):
            baseline_release = latest_release_in_year(releases, baseline_year)
            baseline_dir = baseline_dataset_dir(baseline_year)
            download_dataset(config, baseline_release, baseline_dir, f"baseline/{baseline_year}", logs_root)
            run_detection(
                config,
                annual_run_key(baseline_year, to_year),
                baseline_dir,
                to_dir,
                baseline_year,
                to_year,
                logs_root,
            )
            processed_runs.append(
                append_annual_run(config, payload, baseline_year, to_period, baseline_release, to_release)
            )

    monthly_id = monthly_run_id(from_period, to_period)
    if not run_exists(payload, monthly_id):
        run_detection(
            config,
            monthly_run_key(from_period, to_period),
            from_dir,
            to_dir,
            parse_period(from_period)[0],
            to_year,
            logs_root,
        )
        processed_runs.append(append_monthly_run(config, payload, from_period, to_period, from_release, to_release))

    write_json(monthly_path, payload)
    rebuild_html(config, logs_root)

    state["last_processed_period"] = {
        "from": from_period,
        "to": to_period,
        "release_date": to_release["date"].isoformat(),
        "release_id": to_release["release_id"],
    }
    state["last_run_at"] = now.isoformat()
    write_json(state_path, state)

    cleanup_dataset(from_dir, config)
    if config.get("incremental", {}).get("keep_latest_reference_dataset", True):
        print(f"[cleanup] Keeping latest reference dataset: {to_dir}")
    else:
        cleanup_dataset(to_dir, config)

    return {
        "status": "updated",
        "from_period": from_period,
        "to_period": to_period,
        "processed_runs": [
            {"id": run.get("id"), "label": run.get("label"), "items": len(run.get("items", []))}
            for run in processed_runs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gursu monthly detection update orchestrator")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--now", default=None, help="Optional ISO datetime for dry-run testing")
    parser.add_argument("--dry-run", action="store_true", help="Only report the next planned period; do not download or detect")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_json(config_path, {})
    now = parse_dt(args.now) if args.now else datetime.now(TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TIMEZONE)

    logs_root_from_config(config).mkdir(parents=True, exist_ok=True)

    if not config.get("enabled", False):
        result = {"status": "disabled", "now": now.isoformat(), "first_run_at": config.get("first_run_at")}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    first_run_at = parse_dt(config["first_run_at"])
    if first_run_at.tzinfo is None:
        first_run_at = first_run_at.replace(tzinfo=TIMEZONE)
    if now < first_run_at:
        result = {"status": "waiting", "now": now.isoformat(), "first_run_at": first_run_at.isoformat()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    monthly_path = ROOT / config.get("paths", {}).get("monthly_detections", "data/monthly_detections.json")
    payload = ensure_monthly_file(monthly_path)
    _, state = load_state(config)
    from_period = derive_from_period(config, state, payload)
    to_period = derive_to_period(config, state, now)
    if args.dry_run:
        result = {
            "status": "dry_run",
            "now": now.isoformat(),
            "from_period": from_period,
            "to_period": to_period,
            "monthly_label": period_label(from_period, to_period),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result = run_pipeline(config, now)
    result["now"] = now.isoformat()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
