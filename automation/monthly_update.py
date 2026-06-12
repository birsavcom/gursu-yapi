"""Deferred monthly update orchestrator for the Gursu map.

This script is intentionally conservative. It does not run before the configured
first_run_at date and does not run while enabled=false. The heavy Esri download,
change detection, and classification stages should be wired into the marked
functions before the server timer is enabled in production.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "automation" / "config.json"
MONTH_NAMES_TR = {
    1: "Ocak",
    2: "Subat",
    3: "Mart",
    4: "Nisan",
    5: "Mayis",
    6: "Haziran",
    7: "Temmuz",
    8: "Agustos",
    9: "Eylul",
    10: "Ekim",
    11: "Kasim",
    12: "Aralik",
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


def current_month_period(now: datetime) -> tuple[str, str]:
    previous_month = now.month - 1 or 12
    previous_year = now.year if now.month > 1 else now.year - 1
    return f"{previous_year:04d}-{previous_month:02d}", f"{now.year:04d}-{now.month:02d}"


def period_label(from_period: str, to_period: str) -> str:
    from_year, from_month = [int(part) for part in from_period.split("-")]
    to_year, to_month = [int(part) for part in to_period.split("-")]
    return f"{MONTH_NAMES_TR[from_month]} {from_year} - {MONTH_NAMES_TR[to_month]} {to_year}"


def ensure_monthly_file(path: Path) -> None:
    if path.exists():
        return
    write_json(path, {"version": 1, "district": "gursu", "runs": []})


def run_pipeline(config: dict, from_period: str, to_period: str) -> dict:
    """Placeholder for the production pipeline.

    Production wiring should perform these stages:
    1. Check Esri snapshot availability for the configured bbox.
    2. Download temporary 512x512 zoom-18 imagery for the new period.
    3. Compare previous and new imagery with the segmentation model.
    4. Produce triplet evidence images under results/.
    5. Classify detections using ruhsatli and imarsiz GeoJSON sources.
    6. Append a monthly run to data/monthly_detections.json.
    7. Rebuild index.html.
    8. Delete temporary dataset files after success.
    """
    return {
        "status": "pending_pipeline_wiring",
        "from_period": from_period,
        "to_period": to_period,
        "label": period_label(from_period, to_period),
        "message": "Automation scaffold is installed, but production detection stages are not enabled yet.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gursu monthly detection update orchestrator")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--now", default=None, help="Optional ISO datetime for dry-run testing")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_json(config_path, {})
    timezone = ZoneInfo("Europe/Istanbul")
    now = parse_dt(args.now) if args.now else datetime.now(timezone)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone)

    logs_root = ROOT / config.get("paths", {}).get("logs_root", "logs/automation")
    logs_root.mkdir(parents=True, exist_ok=True)

    if not config.get("enabled", False):
        result = {"status": "disabled", "now": now.isoformat(), "first_run_at": config.get("first_run_at")}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    first_run_at = parse_dt(config["first_run_at"])
    if first_run_at.tzinfo is None:
        first_run_at = first_run_at.replace(tzinfo=timezone)
    if now < first_run_at:
        result = {"status": "waiting", "now": now.isoformat(), "first_run_at": first_run_at.isoformat()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    monthly_path = ROOT / config.get("paths", {}).get("monthly_detections", "data/monthly_detections.json")
    ensure_monthly_file(monthly_path)

    first_from = config.get("incremental", {}).get("first_compare_from")
    first_to = config.get("incremental", {}).get("first_compare_to")
    state_path = ROOT / "automation" / "state.json"
    state = read_json(state_path, {})
    last_period = state.get("last_processed_period") or {}
    from_period = last_period.get("to") or first_from
    to_period = current_month_period(now)[1] if last_period else first_to

    result = run_pipeline(config, from_period, to_period)
    result["now"] = now.isoformat()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
