"""Preprocessing definitions for tabular listing data."""

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

LEGACY_RELEVANT_COLUMNS = [
    "id",
    "review_scores_rating",
    "number_of_reviews",
    "name",
    "description",
    "neighborhood_overview",
    "host_since",
    "host_response_time",
    "host_response_rate",
    "latitude",
    "longitude",
    "bathrooms",
    "bedrooms",
    "beds",
    "price",
    "minimum_nights",
    "maximum_nights",
]


def legacy_clean_raw_listings(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw listings data."""
    cleaned = data.copy()

    cleaned = legacy_filter_relevant(cleaned)
    cleaned = clean_price(cleaned)

    return cleaned


def legacy_filter_relevant(data: pd.DataFrame) -> pd.DataFrame:
    """Keep modeled columns and rows with a known price."""
    not_na_msk = data["price"].notna()

    return data.loc[not_na_msk, LEGACY_RELEVANT_COLUMNS]


def clean_price(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the price column in place."""
    data["price"] = pd.to_numeric(
        data["price"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="raise",
    )
    return data


def sanity_check(name: str, X, y):
    missing = pd.concat([X, y], axis=1).isna().sum(axis=0)
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"{name} contains missing values: \n {missing}")
    
def fit_transform(preprocessor, X: pd.DataFrame):
    return pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=X.columns,
        index=X.index 
    )
    
def transform(preprocessor, X: pd.DataFrame):
    return pd.DataFrame(
        preprocessor.transform(X),
        columns=X.columns,
        index=X.index 
    )
    

PROPERTY_TYPE_KEYWORDS = {
    "serviced_apartment": [
        "serviced apartment",
    ],
    "hotel_like": [
        "hotel",
        "aparthotel",
        "hostel",
        "pension",
        "bed and breakfast",
    ],
    "apartment_like": [
        "rental unit",
        "condo",
        "loft",
    ],
    "house_like": [
        "home",
        "townhouse",
        "villa",
        "guesthouse",
        "guest suite",
        "bungalow",
        "chalet",
        "cabin",
        "farm stay",
        "casa particular",
    ],
    "unusual_stay": [
        "camper",
        "rv",
        "tent",
        "tiny home",
        "hut",
        "windmill",
    ],
}

def simplify_property_type(value):
    if pd.isna(value):
        return "unknown"

    value = str(value).lower().strip()

    for category, keywords in PROPERTY_TYPE_KEYWORDS.items():
        if any(keyword in value for keyword in keywords):
            return category

    return "other"