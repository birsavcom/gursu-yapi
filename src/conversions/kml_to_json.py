import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def kml_to_jsonl(input_path: Path, output_path: Path):
    tree = ET.parse(input_path)
    root = tree.getroot()
    with output_path.open("w", encoding="utf-8") as handle:
        for placemark in root.findall(".//kml:Placemark", KML_NS):
            name = placemark.findtext("kml:name", default="", namespaces=KML_NS)
            coords = placemark.findtext(
                ".//kml:coordinates",
                default="",
                namespaces=KML_NS,
            ).strip()
            if not coords:
                continue
            lon, lat, *_ = coords.split(",")
            row = {
                "name": name,
                "geo_center": {"lat": float(lat), "lon": float(lon)},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"written={output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert KML to JSONL.")
    parser.add_argument("input", help="Input KML path.")
    parser.add_argument("output", help="Output JSONL path.")
    args = parser.parse_args()
    kml_to_jsonl(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

