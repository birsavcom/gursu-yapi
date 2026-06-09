import argparse
import json
from pathlib import Path


def jsonl_to_kml(input_path: Path, output_path: Path):
    placemarks = []
    skipped = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            geo = row.get("geo_center")
            if not geo:
                skipped += 1
                continue

            lat = geo["lat"]
            lon = geo["lon"]
            name = f'{row["status"]}_{row["pair_id"]}'
            description = (
                f"<b>Status:</b> {row['status']}<br/>"
                f"<b>Pair:</b> {row['pair_id']}<br/>"
                f"<b>Confidence:</b> {row.get('confidence', 0)}<br/>"
                f"<b>CLIP:</b> {row.get('clip_score', 0)}"
            )
            placemarks.append(
                f"""    <Placemark>
      <name>{name}</name>
      <description><![CDATA[{description}]]></description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>"""
            )

    output_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
"""
        + "\n".join(placemarks)
        + """
  </Document>
</kml>
""",
        encoding="utf-8",
    )
    print(f"written={output_path}")
    print(f"placemarks={len(placemarks)}")
    print(f"skipped_without_geo={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL detections to KML.")
    parser.add_argument("input", help="Input JSONL path.")
    parser.add_argument("output", help="Output KML path.")
    args = parser.parse_args()
    jsonl_to_kml(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

