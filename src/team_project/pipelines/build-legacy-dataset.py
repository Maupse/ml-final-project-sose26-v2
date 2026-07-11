from pathlib import Path
from team_project.helper.parsing import (
    PROJECT_ROOT,
    RAW_DATA_PATH,
)

from team_project.data.loading import load_raw_listings
from team_project.data.preprocessing import legacy_clean_raw_listings
from team_project.data.feature_construction import add_distance_to_city_center

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "legacy.csv"

def build_legacy_dataset(
    raw_data_path: Path = RAW_DATA_PATH,
    processed_data_path: Path = OUT_PATH,
) -> Path:
    """Build and save the deterministic processed listings dataset."""
    listings = load_raw_listings(raw_data_path)
    cleaned = legacy_clean_raw_listings(listings)
    cleaned = add_distance_to_city_center(cleaned, "distance_km")

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_data_path, index=False)

    return processed_data_path


def main() -> None:
    output_path = build_legacy_dataset()
    print(f"Wrote processed dataset to {output_path}")


if __name__ == "__main__":
    main()
