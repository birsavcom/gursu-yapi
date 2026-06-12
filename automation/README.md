# Gursu Monthly Automation

This folder prepares the deferred monthly update pipeline without changing the current 2021-2026 map results.

Current behavior:

- No timer is installed by this repository.
- `automation/config.json` contains the planned first run date `2026-08-01T03:00:00+03:00`, but `enabled` is currently `false`.
- If `monthly_update.py` is run now, it exits without downloading imagery or running detection because automation is disabled. After activation, it will also wait until the first run date.
- Existing `results/`, `data/imarsiz-gursu.geojson`, `data/ruhsatli-yapi-parseller-gursu.geojson`, and `index.html` results remain untouched.

Planned monthly flow after activation:

1. Check whether the first run date has arrived.
2. Check whether Esri has a newer usable snapshot for the Gursu bbox.
3. Download only the needed temporary 512x512 zoom-18 tiles.
4. Compare the previous processed period with the new period.
5. Generate popup triplet evidence images.
6. Classify points as `yapi_ruhsatli`, `kacak_yapi`, or `yapi_farki`.
7. Append the new monthly run to `data/monthly_detections.json`.
8. Rebuild `index.html` with `python build_gursu_html.py`.
9. Delete temporary dataset files only after a successful run.

The current committed state only adds the safe scheduling/configuration skeleton. The production server timer should be installed later, when the municipality approves activation.

