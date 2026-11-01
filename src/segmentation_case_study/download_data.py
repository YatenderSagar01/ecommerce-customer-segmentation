"""Download and extract the UCI Online Retail workbook."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def download(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "online-retail.zip"
    print(f"Downloading UCI Online Retail from {DATASET_URL}")
    with urlopen(DATASET_URL, timeout=60) as response, archive_path.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)

    with ZipFile(archive_path) as archive:
        workbook = next(
            (item for item in archive.infolist() if Path(item.filename).name.lower() == "online retail.xlsx"),
            None,
        )
        if workbook is None:
            raise RuntimeError("The UCI archive did not contain Online Retail.xlsx.")
        output_path = output_dir / "Online Retail.xlsx"
        with archive.open(workbook) as source, output_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)

    archive_path.unlink()
    print(f"Saved dataset to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
