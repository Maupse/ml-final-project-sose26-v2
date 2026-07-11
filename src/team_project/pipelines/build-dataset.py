import argparse

import numpy as np
import pandas as pd

from team_project.data.feature_construction import (
    AMENITY_PATTERNS,
    add_amenity_features,
    add_bathroom_features,
    add_capacity_features,
    add_distance_to_city_center,
    add_license_features,
)
from team_project.data.preprocessing import clean_price, simplify_property_type
from team_project.helper.parsing import (
    PROJECT_ROOT,
    RAW_DATA_PATH,
)

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "structural-v1.csv"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["structural"],
        required=True,
        help="Feature groups to build. Currently supported: structural.",
    )
    args = parser.parse_args()

    if args.groups != ["structural"]:
        raise ValueError("Only '--groups structural' is supported for now.")

    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    df = df.dropna(subset=["price"]).copy()
    df = clean_price(df)
    df["log1_price"] = np.log1p(df["price"])
    
    df["property_type_clean"] = df["property_type"].apply(simplify_property_type).replace({
        "other": "rare_or_unusual",
        "unusual_stay": "rare_or_unusual",
    })
    df = add_distance_to_city_center(df)
    df = add_capacity_features(df)
    df = add_bathroom_features(df)
    df = add_license_features(df)
    df = add_amenity_features(df)

    columns = [
        "id",
        "price",
        "log1_price",
        "latitude",
        "longitude",
        "distance_to_city_center_km",
        "neighbourhood_cleansed",
        "property_type_clean",
        "room_type",
        "accommodates",
        "bedrooms",
        "bedrooms_missing",
        "beds",
        "beds_missing",
        "bathroom_count",
        "bathroom_privacy",
        "bathroom_information_missing",
        "has_license",
        "amenity_count",
        *AMENITY_PATTERNS,
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(OUT_PATH, index=False)
    print(f"Wrote processed dataset to {OUT_PATH}")

if __name__ == "__main__":
    main()
