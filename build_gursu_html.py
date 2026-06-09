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
CACHE_TOKEN = "20260608a"

BOUNDS = {
    "west": 29.131191,
    "south": 40.198367,
    "east": 29.306497,
    "north": 40.339645,
}

RUNS = [
    {"key": "2021_2026", "year_from": 2021, "year_to": 2026},
]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_detections_for_run(run: dict) -> list[dict]:
    key = run["key"]
    csv_path = DEBUG / f"gursu_change_verified_segmentation_points_{key}.csv"
    verified_dir = RESULTS / f"masks_segmentation_verified_{key}"
    rows = _read_csv_rows(csv_path)

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
        detections.append(
            {
                "detection_id": f"GUR_{key}_{index:06d}",
                "pair_id": pair_id,
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "merged_triplet_rel": (
                    f"results/masks_segmentation_verified_{key}/pair_{pair_id}.jpg?v={CACHE_TOKEN}"
                ),
                "status": "Tespit Edildi",
                "title": "Tespit",
                "description": (
                    f"{run['year_from']} yilinda gorunmeyip "
                    f"{run['year_to']} yilinda gorunen yapi adayi"
                ),
                "date_label": f"{run['year_from']}-{run['year_to']} Kiyaslamasi",
                "year_from": run["year_from"],
                "year_to": run["year_to"],
            }
        )
    return detections


def build_year_datasets() -> dict[str, list[dict]]:
    return {run["key"]: read_detections_for_run(run) for run in RUNS}


def build_html(year_datasets: dict[str, list[dict]]) -> str:
    bounds_json = json.dumps(BOUNDS, ensure_ascii=False)
    datasets_json = json.dumps(year_datasets, ensure_ascii=False)
    runs_json = json.dumps(RUNS, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gursu Yeni Bina Tespit Haritasi</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
  <style>
    html, body {{ height: 100%; margin: 0; font-family: Arial, sans-serif; background: #101418; color: #fff; }}
    #map {{ width: 100%; height: 100%; }}
    #cpopup {{ display: none; position: fixed; z-index: 1100; background: transparent; color: #fff; border-radius: 10px; box-shadow: 0 3px 14px rgba(0,0,0,.4); padding: 0; }}
    #cpopup.visible {{ display: block; }}
    .cpopup-close {{ position: absolute; top: 0; right: 0; border: none; width: 24px; height: 24px; font: 16px/24px Tahoma, Verdana, sans-serif; color: #aeb6bd; background: transparent; cursor: pointer; z-index: 10; }}
    .cpopup-close:hover {{ color: #ff5454; }}
    .cpopup-content {{ margin: 0; }}
    .popup-wrap {{ width: 500px; max-width: 82vw; }}
    .popup-head {{ display: grid; grid-template-columns: repeat(3, 1fr); background: #ffffff; color: #111111; border-radius: 8px 8px 0 0; overflow: hidden; font-weight: 700; text-align: center; }}
    .popup-head div {{ padding: 8px 6px; border-right: 1px solid #d8dde3; }}
    .popup-head div:last-child {{ border-right: 0; }}
    .triplet-frame {{ position: relative; margin-bottom: 12px; border: 1px solid #333; border-radius: 0 0 8px 8px; overflow: hidden; background: #000; }}
    .popup-image {{ display: block; width: 100%; }}
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
    .map-panel {{ position: fixed; z-index: 999; top: 14px; left: 14px; background: rgba(10,20,30,0.90); color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 13px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); min-width: 278px; }}
    .panel-title {{ font-weight: 700; margin-bottom: 8px; }}
    .controls {{ display: flex; align-items: center; gap: 8px; }}
    .controls label {{ color: #c8d0d7; }}
    .year-select {{ background: #1f2933; color: #fff; border: 1px solid #405161; border-radius: 6px; padding: 5px 7px; }}
    .leaflet-top.leaflet-left {{ top: 110px; }}
  </style>
</head>
<body>
  <div class="map-panel">
    <div class="panel-title">Gursu Haritasi - Yeni Bina Adaylari</div>
    <div class="controls">
      <label for="fromYear">Baslangic</label>
      <select class="year-select" id="fromYear"></select>
      <label for="toYear">Bitis</label>
      <select class="year-select" id="toYear"></select>
    </div>
  </div>
  <div id="map"></div>
  <div id="cpopup"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const boundsInfo = {bounds_json};
    const yearDatasets = {datasets_json};
    const runs = {runs_json};
    const fromYears = [2021];
    const toYears = [2026];
    const map = L.map('map', {{ zoomControl: true, attributionControl: false, dragging: true, touchZoom: true, doubleClickZoom: true, scrollWheelZoom: true, boxZoom: false, keyboard: false, zoomSnap: 0.5 }});
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}').addTo(map);
    const defaultBounds = L.latLngBounds([boundsInfo.south, boundsInfo.west], [boundsInfo.north, boundsInfo.east]);
    map.fitBounds(defaultBounds);
    map.setMaxBounds(defaultBounds.pad(0.08));
    const markerDefaultStyle = {{ radius: 6, fillColor: '#ff2d2d', color: '#2b2f34', weight: 1, opacity: 1, fillOpacity: 0.95 }};
    const markerHoverStyle = {{ radius: 6, fillColor: '#4fc3f7', color: '#2b2f34', weight: 1, opacity: 1, fillOpacity: 0.95 }};
    const markerSelectedStyle = {{ radius: 9, fillColor: '#4caf50', color: '#2b2f34', weight: 1, opacity: 1, fillOpacity: 0.95 }};
    let selectedMarker = null;
    let markerLayer = L.layerGroup().addTo(map);
    const cpopup = document.getElementById('cpopup');
    const fromSelect = document.getElementById('fromYear');
    const toSelect = document.getElementById('toYear');

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
        + '<div class="popup-row"><div class="popup-label">Donem:</div><div class="popup-value">' + item.date_label + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">Durum:</div><div class="popup-value">' + item.status + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">Aciklama:</div><div class="popup-value">' + item.description + '</div></div>'
        + '<div class="popup-row"><div class="popup-label">Koordinatlar:</div><div class="popup-value">' + item.lat.toFixed(6) + ', ' + item.lon.toFixed(6) + '</div></div>'
        + '<button class="popup-btn" type="button">Detay</button>'
        + '</div></div></div></div></div>';
    }}

    function showPopup(item, markerLatLng) {{
      cpopup.innerHTML = makePopupHTML(item);
      cpopup.classList.add('visible');
      requestAnimationFrame(function() {{
        const pw = cpopup.offsetWidth;
        const ph = cpopup.offsetHeight;
        const pt = map.latLngToContainerPoint(markerLatLng);
        const mapRect = map.getContainer().getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const gap = 12;
        const margin = 10;
        const goLeft = (mapRect.left + pt.x) > vw / 2;
        const goUp = (mapRect.top + pt.y) > vh / 2;
        let left = mapRect.left + (goLeft ? pt.x - pw - gap : pt.x + gap);
        let top = mapRect.top + (goUp ? pt.y - ph - gap : pt.y + gap);
        left = Math.max(margin, Math.min(left, vw - pw - margin));
        top = Math.max(margin, Math.min(top, vh - ph - margin));
        cpopup.style.left = left + 'px';
        cpopup.style.top = top + 'px';
      }});
    }}

    window.hidePopup = function() {{
      cpopup.classList.remove('visible');
      cpopup.innerHTML = '';
      if (selectedMarker) {{
        selectedMarker.setStyle(markerDefaultStyle);
        selectedMarker = null;
      }}
    }};

    function selectedFeatures() {{
      const fromYear = Number(fromSelect.value);
      const toYear = Number(toSelect.value);
      const key = fromYear + '_' + toYear;
      return yearDatasets[key] || [];
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
        const marker = L.circleMarker([item.lat, item.lon], markerDefaultStyle);
        marker._item = item;
        marker.on('mouseover', function() {{ if (this !== selectedMarker) this.setStyle(markerHoverStyle); }});
        marker.on('mouseout', function() {{ if (this !== selectedMarker) this.setStyle(markerDefaultStyle); }});
        marker.on('click', function(e) {{
          L.DomEvent.stopPropagation(e);
          if (selectedMarker && selectedMarker !== this) selectedMarker.setStyle(markerDefaultStyle);
          selectedMarker = this;
          this.setStyle(markerSelectedStyle);
          showPopup(this._item, this.getLatLng());
        }});
        markerLayer.addLayer(marker);
      }});
    }}

    function normalizeYearSelection() {{
      const fromYear = Number(fromSelect.value);
      const validToYears = toYears.filter(function(year) {{ return year > fromYear; }});
      fillSelect(toSelect, validToYears, Math.max(...validToYears));
    }}

    fillSelect(fromSelect, fromYears, 2021);
    normalizeYearSelection();
    fromSelect.addEventListener('change', function() {{
      normalizeYearSelection();
      renderMarkers();
    }});
    toSelect.addEventListener('change', renderMarkers);
    renderMarkers();
    map.on('click', hidePopup);
  </script>
</body>
</html>"""


def main() -> None:
    year_datasets = build_year_datasets()
    OUT_PATH.write_text(build_html(year_datasets), encoding="utf-8")
    summary = {key: len(value) for key, value in year_datasets.items()}
    print(json.dumps({"output": str(OUT_PATH), "detections": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


