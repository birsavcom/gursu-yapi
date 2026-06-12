"""
Build the Gursu change-detection review map.

The generated index.html is self-contained apart from Leaflet CDN assets and the
local evidence triplet images under results/masks_segmentation_verified_*.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEBUG = RESULTS / "debug"
OUT_PATH = ROOT / "index.html"
DATA = ROOT / "data"
IMARSIZ_GEOJSON = DATA / "imarsiz-gursu.geojson"
RUHSATLI_GEOJSON = DATA / "ruhsatli-yapi-parseller-gursu.geojson"
MONTHLY_DETECTIONS = DATA / "monthly_detections.json"
CACHE_TOKEN = "20260612a"

BOUNDS = {
    "west": 29.131191,
    "south": 40.198367,
    "east": 29.306497,
    "north": 40.339645,
}

RUNS = [
    {
        "key": "2021_2026",
        "year_from": 2021,
        "year_to": 2026,
        "period_id": "initial_2021_2026",
        "period_label": "2021-2026 \u0130lk Tespit",
        "period_type": "initial",
        "display_year": 2026,
    },
]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _walk_coordinate_pairs(value):
    if isinstance(value, list) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
        yield float(value[0]), float(value[1])
    elif isinstance(value, list):
        for item in value:
            yield from _walk_coordinate_pairs(item)


def _geometry_bbox(geometry: dict) -> tuple[float, float, float, float] | None:
    points = list(_walk_coordinate_pairs((geometry or {}).get("coordinates")))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    j = len(ring) - 1
    for i, current in enumerate(ring):
        previous = ring[j]
        xi, yi = float(current[0]), float(current[1])
        xj, yj = float(previous[0]), float(previous[1])
        if (yi > lat) != (yj > lat):
            x_intersect = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, polygon: list) -> bool:
    if not polygon or not _point_in_ring(lon, lat, polygon[0]):
        return False
    return not any(_point_in_ring(lon, lat, hole) for hole in polygon[1:])


def _point_in_geometry(lon: float, lat: float, geometry: dict) -> bool:
    geom_type = (geometry or {}).get("type")
    coordinates = (geometry or {}).get("coordinates") or []
    if geom_type == "Polygon":
        return _point_in_polygon(lon, lat, coordinates)
    if geom_type == "MultiPolygon":
        return any(_point_in_polygon(lon, lat, polygon) for polygon in coordinates)
    return False


def load_geojson_index(path: Path) -> list[tuple[tuple[float, float, float, float], dict]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    indexed = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        bbox = _geometry_bbox(geometry)
        if bbox:
            indexed.append((bbox, geometry))
    return indexed


def load_imarsiz_index() -> list[tuple[tuple[float, float, float, float], dict]]:
    return load_geojson_index(IMARSIZ_GEOJSON)


def load_ruhsatli_index() -> list[tuple[tuple[float, float, float, float], dict]]:
    return load_geojson_index(RUHSATLI_GEOJSON)


def is_in_polygon_index(lon: float, lat: float, polygon_index: list) -> bool:
    for (west, south, east, north), geometry in polygon_index:
        if west <= lon <= east and south <= lat <= north and _point_in_geometry(lon, lat, geometry):
            return True
    return False


def run_interval_text(run: dict) -> str:
    from_label = run.get("from_period_label")
    to_label = run.get("to_period_label")
    if from_label and to_label:
        return f"{from_label} döneminde görünmeyip {to_label} döneminde görünen"
    return f"{run['year_from']} yılında görünmeyip {run['year_to']} yılında görünen"


def detection_classification(
    lon: float,
    lat: float,
    run: dict,
    imarsiz_index: list,
    ruhsatli_index: list,
) -> dict:
    interval_text = run_interval_text(run)
    if is_in_polygon_index(lon, lat, ruhsatli_index):
        return {
            "category": "yapi_ruhsatli",
            "title": "Yapı Ruhsatlı",
            "status": "Yapı ruhsatlı",
            "description": f"{interval_text} yapı ruhsatlı yapı",
            "imar_status": "Yapı ruhsatlı",
            "accent_color": "#8b5a2b",
            "accent_text_color": "#ffffff",
        }
    if is_in_polygon_index(lon, lat, imarsiz_index):
        return {
            "category": "kacak_yapi",
            "title": "Kaçak Yapı",
            "status": "Kaçak yapı adayı",
            "description": f"{interval_text} kaçak yapı adayı",
            "imar_status": "İmarsız alan",
            "accent_color": "#ff2d2d",
            "accent_text_color": "#ffffff",
        }
    return {
        "category": "yapi_farki",
        "title": "İmarlı",
        "status": "İmarlı yapı farkı",
        "description": f"{interval_text} fark yapı",
        "imar_status": "İmarlı alan",
        "accent_color": "#f4c430",
        "accent_text_color": "#111111",
    }


def read_detections_for_run(run: dict) -> list[dict]:
    key = run["key"]
    csv_path = DEBUG / f"gursu_change_verified_segmentation_points_{key}.csv"
    verified_dir = RESULTS / f"masks_segmentation_verified_{key}"
    rows = _read_csv_rows(csv_path)
    imarsiz_index = load_imarsiz_index()
    ruhsatli_index = load_ruhsatli_index()

    detections = []
    seen_detections = set()
    for index, row in enumerate(rows, start=1):
        pair_id = str(row.get("pair_id", "")).zfill(4)
        if not pair_id:
            continue
        image_path = verified_dir / f"pair_{pair_id}.jpg"
        if not image_path.exists():
            continue
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        dedupe_key = (pair_id, round(lat, 8), round(lon, 8))
        if dedupe_key in seen_detections:
            continue
        seen_detections.add(dedupe_key)
        classification = detection_classification(lon, lat, run, imarsiz_index, ruhsatli_index)
        detections.append(
            {
                "detection_id": f"GUR_{key}_{index:06d}",
                "pair_id": pair_id,
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "merged_triplet_rel": (
                    f"results/masks_segmentation_verified_{key}/pair_{pair_id}.jpg?v={CACHE_TOKEN}"
                ),
                "date_label": run.get("period_label") or f"{run['year_from']}-{run['year_to']} K\u0131yaslamas\u0131",
                "year_from": run["year_from"],
                "year_to": run["year_to"],
                "period_id": run.get("period_id", key),
                "period_label": run.get("period_label", f"{run['year_from']}-{run['year_to']} K\u0131yaslamas\u0131"),
                "period_type": run.get("period_type", "initial"),
                "display_year": run.get("display_year", run["year_to"]),
                "from_period": run.get("from_period"),
                "to_period": run.get("to_period"),
                **classification,
            }
        )
    return detections


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_monthly_runs() -> list[dict]:
    payload = _read_json(MONTHLY_DETECTIONS, {"runs": []})
    return payload.get("runs", []) if isinstance(payload, dict) else []


def normalize_monthly_item(item: dict, run: dict) -> dict:
    normalized = dict(item)
    normalized.setdefault("period_id", run["id"])
    normalized.setdefault("period_label", run["label"])
    normalized.setdefault("period_type", run.get("type", "monthly"))
    normalized.setdefault("date_label", run["label"])
    normalized.setdefault("display_year", run.get("display_year") or run.get("year_to"))
    normalized.setdefault("year_from", run.get("year_from"))
    normalized.setdefault("year_to", run.get("year_to"))
    normalized.setdefault("from_period", run.get("from_period"))
    normalized.setdefault("to_period", run.get("to_period"))
    return normalized


def build_year_datasets() -> dict[str, list[dict]]:
    datasets = {run["key"]: read_detections_for_run(run) for run in RUNS}
    for run in load_monthly_runs():
        run_id = run.get("id")
        if not run_id:
            continue
        items = run.get("items", [])
        datasets[run_id] = [normalize_monthly_item(item, run) for item in items]
    return datasets


def build_period_options(year_datasets: dict[str, list[dict]]) -> list[dict]:
    periods = []
    seen = set()
    for run in RUNS:
        period_id = run.get("period_id", run["key"])
        if period_id in seen:
            continue
        seen.add(period_id)
        periods.append({
            "id": period_id,
            "label": run.get("period_label", f"{run['year_from']}-{run['year_to']}"),
            "type": run.get("period_type", "initial"),
            "year_from": run.get("year_from"),
            "year_to": run.get("year_to"),
            "display_year": run.get("display_year", run.get("year_to")),
            "from_period": run.get("from_period"),
            "to_period": run.get("to_period"),
        })
    for run in load_monthly_runs():
        run_id = run.get("id")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        periods.append({
            "id": run_id,
            "label": run.get("label", run_id),
            "type": run.get("type", "monthly"),
            "year_from": run.get("year_from"),
            "year_to": run.get("year_to"),
            "display_year": run.get("display_year") or run.get("year_to"),
            "from_period": run.get("from_period"),
            "to_period": run.get("to_period"),
        })
    return [period for period in periods if any(
        item.get("period_id") == period["id"] for items in year_datasets.values() for item in items
    )]


def build_html(year_datasets: dict[str, list[dict]], period_options: list[dict]) -> str:
    bounds_json = json.dumps(BOUNDS, ensure_ascii=False)
    datasets_json = json.dumps(year_datasets, ensure_ascii=False)
    periods_json = json.dumps(period_options, ensure_ascii=False)
    runs_json = json.dumps(RUNS, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>G\u00fcrsu Yap\u0131 Tespit Haritas\u0131</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
  <style>
    html, body {{ height: 100%; margin: 0; font-family: Arial, sans-serif; background: #101418; color: #fff; }}
    #map {{ width: 100%; height: 100%; }}
    #cpopup {{ display: none; position: fixed; z-index: 1100; background: transparent; color: #fff; border-radius: 10px; box-shadow: 0 3px 14px rgba(0,0,0,.4); padding: 0; max-width: calc(100vw - 24px); max-height: calc(100vh - 24px); overflow: auto; }}
    #cpopup.visible {{ display: block; }}
    .cpopup-close {{ position: absolute; top: 0; right: 0; border: none; width: 24px; height: 24px; font: 16px/24px Tahoma, Verdana, sans-serif; color: #aeb6bd; background: transparent; cursor: pointer; z-index: 10; }}
    .cpopup-close:hover {{ color: #ff5454; }}
    .cpopup-content {{ margin: 0; }}
    .popup-wrap {{ width: min(500px, calc(100vw - 24px)); }}
    .popup-head {{ display: grid; grid-template-columns: repeat(3, 1fr); background: #ffffff; color: #111111; border-radius: 8px 8px 0 0; overflow: hidden; font-weight: 700; text-align: center; }}
    .popup-head div {{ padding: 8px 6px; border-right: 1px solid #d8dde3; }}
    .popup-head div:last-child {{ border-right: 0; }}
    .triplet-frame {{ position: relative; aspect-ratio: 3 / 1; margin-bottom: 12px; border: 1px solid #333; border-radius: 0 0 8px 8px; overflow: hidden; background: #000; }}
    .popup-image {{ display: block; width: 100%; height: 100%; object-fit: cover; }}
    .triplet-sep {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #000; pointer-events: none; z-index: 2; }}
    .triplet-sep.sep1 {{ left: 33.3333%; transform: translateX(-1px); }}
    .triplet-sep.sep2 {{ left: 66.6666%; transform: translateX(-1px); }}
    .popup-card {{ background: #171b20; border-radius: 10px; overflow: hidden; border: 1px solid #2d333b; }}
    .popup-body {{ padding: 0 14px 14px; }}
    .popup-title {{ font-size: 16px; font-weight: 700; margin: 4px 0 10px; }}
    .popup-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-top: 1px solid #2b3138; font-size: 13px; line-height: 1.4; }}
    .popup-row:first-of-type {{ border-top: 0; }}
    .popup-label {{ color: #c8d0d7; font-weight: 600; min-width: 96px; }}
    .popup-value {{ color: #ffffff; text-align: right; flex: 1; }}
    .popup-btn {{ margin-top: 14px; width: 100%; border: 0; border-radius: 8px; background: #f03232; color: #fff; font-weight: 700; padding: 12px 14px; font-size: 15px; cursor: default; }}
    .legend {{ display: flex; align-items: center; gap: 10px; margin-top: 9px; color: #d6dee6; font-size: 12px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; border: 1px solid #2b2f34; }}
    .legend-dot.red {{ background: #ff2d2d; }}
    .legend-dot.yellow {{ background: #f4c430; }}
    .legend-dot.brown {{ background: #8b5a2b; }}
    .map-panel {{ position: fixed; z-index: 999; top: 14px; left: 14px; background: rgba(10,20,30,0.90); color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 13px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); min-width: 330px; max-width: calc(100vw - 28px); }}
    .panel-title {{ font-weight: 700; margin-bottom: 8px; }}
    .controls {{ display: flex; align-items: center; gap: 8px; }}
    .controls label {{ color: #c8d0d7; }}
    .year-select {{ background: #1f2933; color: #fff; border: 1px solid #405161; border-radius: 6px; padding: 5px 7px; }}
    .period-select {{ min-width: 178px; }}
    .leaflet-top.leaflet-left {{ top: 110px; }}
  </style>
</head>
<body>
  <div class="map-panel">
    <div class="panel-title">G\u00fcrsu Haritas\u0131 - Yap\u0131 Tespitleri</div>
    <div class="controls">
      <label for="fromYear">Ba\u015flang\u0131\u00e7</label>
      <select class="year-select" id="fromYear"></select>
      <label for="toYear">Biti\u015f</label>
      <select class="year-select" id="toYear"></select>
      <label for="periodSelect">Tespit D\u00f6nemi</label>
      <select class="year-select period-select" id="periodSelect"></select>
    </div>
    <div class="legend"><span class="legend-item"><span class="legend-dot brown"></span>Yap\u0131 Ruhsatl\u0131</span><span class="legend-item"><span class="legend-dot yellow"></span>\u0130marl\u0131 / Yap\u0131 Fark\u0131</span><span class="legend-item"><span class="legend-dot red"></span>Ka\u00e7ak Yap\u0131</span></div>
  </div>
  <div id="map"></div>
  <div id="cpopup"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const boundsInfo = {bounds_json};
    const yearDatasets = {datasets_json};
    const periodOptions = {periods_json};
    const runs = {runs_json};
    const calendarYear = new Date().getFullYear();
    let activeDisplayYear = pickActiveDisplayYear();
    const map = L.map('map', {{ zoomControl: true, attributionControl: false, dragging: true, touchZoom: true, doubleClickZoom: true, scrollWheelZoom: true, boxZoom: false, keyboard: false, zoomSnap: 0.5 }});
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}').addTo(map);
    const defaultBounds = L.latLngBounds([boundsInfo.south, boundsInfo.west], [boundsInfo.north, boundsInfo.east]);
    map.fitBounds(defaultBounds);
    map.setMaxBounds(defaultBounds.pad(0.08));
    const markerPalette = {{
      yapi_ruhsatli: {{ fill: '#8b5a2b', stroke: '#3f2a14' }},
      kacak_yapi: {{ fill: '#ff2d2d', stroke: '#2b2f34' }},
      yapi_farki: {{ fill: '#f4c430', stroke: '#6c5400' }}
    }};
    function markerStyle(item, state) {{
      const palette = markerPalette[item.category] || markerPalette.kacak_yapi;
      if (state === 'hover') return {{ radius: 6, fillColor: '#4fc3f7', color: palette.stroke, weight: 1, opacity: 1, fillOpacity: 0.95 }};
      if (state === 'selected') return {{ radius: 9, fillColor: '#4caf50', color: palette.stroke, weight: 1, opacity: 1, fillOpacity: 0.95 }};
      return {{ radius: 6, fillColor: palette.fill, color: palette.stroke, weight: 1, opacity: 1, fillOpacity: 0.95 }};
    }}
    let selectedMarker = null;
    let markerLayer = L.layerGroup().addTo(map);
    const cpopup = document.getElementById('cpopup');
    const fromSelect = document.getElementById('fromYear');
    const toSelect = document.getElementById('toYear');
    const periodSelect = document.getElementById('periodSelect');

    function fillSelect(select, years, selected) {{
      select.innerHTML = '';
      years.forEach(function(year) {{
        const option = document.createElement('option');
        option.value = String(year);
        option.textContent = String(year);
        if (year === selected) option.selected = true;
        select.appendChild(option);
      }});
    }}

    function periodDisplayYear(period) {{
      return Number(period.display_year || period.year_to || 0);
    }}

    function dataDisplayYears() {{
      return Array.from(new Set(periodOptions.map(periodDisplayYear).filter(Boolean))).sort(function(a, b) {{ return a - b; }});
    }}

    function toYearOptions() {{
      const dataYears = dataDisplayYears();
      const startYear = dataYears.length ? Math.min.apply(null, dataYears) : 2026;
      const endYear = Math.max(calendarYear, dataYears.length ? Math.max.apply(null, dataYears) : startYear);
      const years = [];
      for (let year = startYear; year <= endYear; year += 1) years.push(year);
      return years;
    }}

    function pickActiveDisplayYear() {{
      const dataYears = dataDisplayYears();
      if (dataYears.includes(calendarYear)) return calendarYear;
      return dataYears.length ? dataYears[dataYears.length - 1] : calendarYear;
    }}

    function visiblePeriodOptions() {{
      return periodOptions.filter(function(period) {{ return periodDisplayYear(period) === activeDisplayYear; }});
    }}

    function visiblePeriodIds() {{
      return new Set(visiblePeriodOptions().map(function(period) {{ return period.id; }}));
    }}

    function refreshYearSelects() {{
      const periods = visiblePeriodOptions();
      const fromYear = periods.length ? Math.min.apply(null, periods.map(function(period) {{ return Number(period.year_from || 2021); }})) : 2021;
      fillSelect(fromSelect, [fromYear], fromYear);
      fillSelect(toSelect, toYearOptions(), activeDisplayYear);
    }}

    function fillPeriodSelect() {{
      periodSelect.innerHTML = '';
      const visiblePeriods = visiblePeriodOptions();
      if (!visiblePeriods.length) {{
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = 'Tespit yok';
        periodSelect.appendChild(emptyOption);
        return;
      }}
      if (visiblePeriods.length > 1) {{
        const allOption = document.createElement('option');
        allOption.value = 'all';
        allOption.textContent = 'Tüm Tespitler';
        periodSelect.appendChild(allOption);
      }}
      visiblePeriods.forEach(function(period) {{
        const option = document.createElement('option');
        option.value = period.id;
        option.textContent = period.label;
        periodSelect.appendChild(option);
      }});
      if (visiblePeriods.length === 1) periodSelect.value = visiblePeriods[0].id;
    }}


    function allFeatures() {{
      return Object.values(yearDatasets).reduce(function(items, current) {{
        return items.concat(current);
      }}, []);
    }}


    function makePopupHTML(item) {{
      return '<button class="cpopup-close" onclick="hidePopup()">&times;</button>'
        + '<div class="cpopup-content"><div class="popup-wrap"><div class="popup-card">'
        + '<div class="popup-head"><div>Before</div><div>After</div><div>Detected</div></div>'
        + '<div class="triplet-frame">'
        + '<img class="popup-image" src="' + item.merged_triplet_rel + '" alt="' + item.detection_id + '">'
        + '<div class="triplet-sep sep1"></div><div class="triplet-sep sep2"></div>'
        + '</div>'
        + '<div class="popup-body">'
        + '<div class="popup-title">' + item.title + '</div>'
        + '<div class="popup-row"><div class="popup-label">Nokta ID:</div><div class="popup-value">' + item.detection_id + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">D\u00f6nem:</div><div class="popup-value">' + item.date_label + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">Durum:</div><div class="popup-value">' + item.status + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">\u0130mar:</div><div class="popup-value">' + item.imar_status + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">A\u00e7\u0131klama:</div><div class="popup-value">' + item.description + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">Koordinatlar:</div><div class="popup-value">' + item.lat.toFixed(6) + ', ' + item.lon.toFixed(6) + '</div></div>'
        + '<button class="popup-btn" type="button" style="background:' + item.accent_color + '; color:' + item.accent_text_color + ';">Detay</button>'
        + '</div></div></div></div></div>';
    }}

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(value, max));
    }}

    function positionPopup(markerLatLng) {{
      const pt = map.latLngToContainerPoint(markerLatLng);
      const mapRect = map.getContainer().getBoundingClientRect();
      const markerX = mapRect.left + pt.x;
      const markerY = mapRect.top + pt.y;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const gap = 14;
      const margin = 12;
      const pw = Math.min(cpopup.offsetWidth, vw - margin * 2);
      const ph = Math.min(cpopup.offsetHeight, vh - margin * 2);
      const spaceLeft = markerX - margin - gap;
      const spaceRight = vw - markerX - margin - gap;
      const spaceAbove = markerY - margin - gap;
      const spaceBelow = vh - markerY - margin - gap;
      let left = markerX > vw / 2 ? markerX - pw - gap : markerX + gap;
      let top = markerY > vh / 2 ? markerY - ph - gap : markerY + gap;

      if (left < margin && spaceRight >= pw) left = markerX + gap;
      if (left + pw > vw - margin && spaceLeft >= pw) left = markerX - pw - gap;
      if (top < margin && spaceBelow >= ph) top = markerY + gap;
      if (top + ph > vh - margin && spaceAbove >= ph) top = markerY - ph - gap;

      cpopup.style.left = clamp(left, margin, vw - pw - margin) + 'px';
      cpopup.style.top = clamp(top, margin, vh - ph - margin) + 'px';
    }}

    function showPopup(item, markerLatLng) {{
      cpopup.innerHTML = makePopupHTML(item);
      cpopup.style.left = '-9999px';
      cpopup.style.top = '-9999px';
      cpopup.classList.add('visible');
      requestAnimationFrame(function() {{
        positionPopup(markerLatLng);
        const image = cpopup.querySelector('.popup-image');
        if (image && !image.complete) {{
          image.addEventListener('load', function() {{ positionPopup(markerLatLng); }}, {{ once: true }});
        }}
      }});
    }}
    window.addEventListener('resize', function() {{
      if (cpopup.classList.contains('visible') && selectedMarker) positionPopup(selectedMarker.getLatLng());
    }});
    map.on('zoom move', function() {{
      if (cpopup.classList.contains('visible') && selectedMarker) positionPopup(selectedMarker.getLatLng());
    }});

    window.hidePopup = function() {{
      cpopup.classList.remove('visible');
      cpopup.innerHTML = '';
      if (selectedMarker) {{
        selectedMarker.setStyle(markerStyle(selectedMarker._item, 'default'));
        selectedMarker = null;
      }}
    }};

    function selectedFeatures() {{
      const periodId = periodSelect.value;
      const allowedPeriodIds = visiblePeriodIds();
      const features = allFeatures().filter(function(item) {{ return allowedPeriodIds.has(item.period_id); }});
      if (!periodId || periodId === 'all') return features;
      return features.filter(function(item) {{ return item.period_id === periodId; }});
    }}


    function renderMarkers() {{
      hidePopup();
      markerLayer.clearLayers();
      const features = selectedFeatures();
      const seen = new Set();
      features.forEach(function(item) {{
        const dedupeKey = item.lat.toFixed(6) + '_' + item.lon.toFixed(6) + '_' + item.date_label;
        if (seen.has(dedupeKey)) return;
        seen.add(dedupeKey);
        const marker = L.circleMarker([item.lat, item.lon], markerStyle(item, 'default'));
        marker._item = item;
        marker.on('mouseover', function() {{ if (this !== selectedMarker) this.setStyle(markerStyle(this._item, 'hover')); }});
        marker.on('mouseout', function() {{ if (this !== selectedMarker) this.setStyle(markerStyle(this._item, 'default')); }});
        marker.on('click', function(e) {{
          L.DomEvent.stopPropagation(e);
          if (selectedMarker && selectedMarker !== this) selectedMarker.setStyle(markerStyle(selectedMarker._item, 'default'));
          selectedMarker = this;
          this.setStyle(markerStyle(this._item, 'selected'));
          showPopup(this._item, this.getLatLng());
        }});
        markerLayer.addLayer(marker);
      }});
    }}

    function normalizeYearSelection() {{
      refreshYearSelects();
    }}


    normalizeYearSelection();
    fillPeriodSelect();
    fromSelect.addEventListener('change', function() {{
      normalizeYearSelection();
      renderMarkers();
    }});
    toSelect.addEventListener('change', function() {{
      activeDisplayYear = Number(toSelect.value);
      refreshYearSelects();
      fillPeriodSelect();
      renderMarkers();
    }});
    periodSelect.addEventListener('change', renderMarkers);
    renderMarkers();
    map.on('click', hidePopup);
  </script>
</body>
</html>"""


def main() -> None:
    year_datasets = build_year_datasets()
    period_options = build_period_options(year_datasets)
    OUT_PATH.write_text(build_html(year_datasets, period_options), encoding="utf-8")
    summary = {
        key: {
            "total": len(value),
            "by_category": {
                category: sum(1 for item in value if item.get("category") == category)
                for category in sorted({item.get("category") for item in value})
            },
        }
        for key, value in year_datasets.items()
    }
    print(json.dumps({"output": str(OUT_PATH), "detections": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
