"""Download the raw gzip-compressed CSV dataset."""

from pathlib import Path
from urllib.request import urlretrieve

URL = "https://data.insideairbnb.com/germany/bv/munich/2025-09-27/data/listings.csv.gz"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "listings.csv.gz"
)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, OUTPUT_PATH)
    print(f"Downloaded dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()