"""Download the regulation PDFs listed in data/documents.json into data/pdfs.

    python -m app.download

The PDFs are official publications of the Sharjah Prevention and Safety Authority.
They are not stored in this repository; this script fetches them from the source.
"""

import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    target_dir = DATA_DIR / "pdfs"
    target_dir.mkdir(parents=True, exist_ok=True)
    for entry in json.loads((DATA_DIR / "documents.json").read_text(encoding="utf-8")):
        target = target_dir / entry["file"]
        if target.exists():
            print(f"{entry['file']}: already downloaded")
            continue
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        print(f"{entry['file']}: downloaded")


if __name__ == "__main__":
    main()
