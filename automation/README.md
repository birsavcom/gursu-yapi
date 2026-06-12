# Gursu Monthly Automation

This folder contains the server-side monthly update pipeline for the Gursu map.
It is configured to wait until `2026-08-01T03:00:00+03:00` before doing any
heavy work.

Current behavior:

- `automation/config.json` has `enabled: true`, but the script returns `waiting`
  until the configured first run date.
- No timer is installed by this repository. The systemd files are templates for
  the production server.
- Running `python automation/monthly_update.py --dry-run` reports the next planned
  comparison without downloading imagery or running detection.
- Existing 2021-2026 results remain untouched until a successful future run adds
  a new run to `data/monthly_detections.json` and rebuilds `index.html`.

Planned first run:

- Monthly comparison: `Temmuz 2026 - Agustos 2026`
- It downloads the latest Esri Wayback releases available on or before each
  period end. If no newer Esri release exists, it exits with `no_new_release`.

Year transition behavior:

- Monthly detections remain stored in `data/monthly_detections.json`.
- The map filter shows the active/current year's periods by default.
- In 2026, the filter shows `2021-2026 İlk Tespit` plus monthly periods whose
  target period is in 2026.
- In 2027, the automation creates a `2021-2027 Genel Tespit` annual rollup the
  first time a 2027 period is processed, then continues with monthly periods such
  as `Aralik 2026 - Ocak 2027` and `Ocak 2027 - Subat 2027`.
- Older monthly periods stay in JSON for audit/history but are not shown in the
  active-year filter once the map has newer-year periods.

Production flow after activation:

1. Check the configured first run date.
2. Check Esri Wayback releases for the next monthly period.
3. Download the required temporary 512x512 zoom-18 datasets.
4. Run `gursu_change_detection.py --mode segment-full --resume` with explicit
   old/new dataset directories.
5. Generate before / after / detected popup triplet images.
6. Classify points as `yapi_ruhsatli`, `kacak_yapi`, or `yapi_farki`.
7. Append or update the run in `data/monthly_detections.json`.
8. Rebuild `index.html` with `python build_gursu_html.py`.
9. Delete the older temporary monthly dataset after success and keep the latest
   reference dataset for the next month.

The repository changes alone do not install the server timer. The timer must be
installed on the production server when deployment is approved.
