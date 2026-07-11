"""Preprocessing definitions for tabular listing data."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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

    
def feature_summary(data, columns):
    """Return modelling-oriented type, missingness, and cardinality checks."""
    return pd.DataFrame({
        "dtype": data[columns].dtypes.astype(str),
        "missing_count": data[columns].isna().sum(),
        "missing_percent": data[columns].isna().mean().mul(100).round(2),
        "unique_values": data[columns].nunique(dropna=True),
    })

def _get_feature_names(preprocessor, columns):
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out()
    return columns


def fit_transform(preprocessor, X: pd.DataFrame):
    return pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=_get_feature_names(preprocessor, X.columns),
        index=X.index 
    )
    
def transform(preprocessor, X: pd.DataFrame):
    return pd.DataFrame(
        preprocessor.transform(X),
        columns=_get_feature_names(preprocessor, X.columns),
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

def get_preprocessor_v1(config):
    imputer = config["preprocessing"]["imputer"]
    fill_value = config["preprocessing"]["fill_value"] # only used if imputer is "constant"

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer, fill_value=fill_value)),
        ("scaler", MinMaxScaler()),        
    ])
    
    return preprocessor


V2_CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "property_type_clean",
    "room_type",
    "bathroom_privacy",
]

V2_NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "distance_to_city_center_km",
    "accommodates",
    "bedrooms",
    "bedrooms_missing",
    "beds",
    "beds_missing",
    "bathroom_count",
    "bathroom_information_missing",
    "has_license",
    "amenity_count",
    "has_wifi",
    "has_kitchen",
    "has_parking",
    "has_washer",
    "has_air_conditioning",
    "has_workspace",
    "has_balcony",
    "has_dryer",
    "has_elevator",
    "has_smoke_alarm",
    "allows_pets",
]


def get_preprocessor_v2(config):
    features = config["experiment"]["features"]
    numeric_features = [feature for feature in V2_NUMERIC_FEATURES if feature in features]
    categorical_features = [feature for feature in V2_CATEGORICAL_FEATURES if feature in features]
    known_features = set(V2_NUMERIC_FEATURES) | set(V2_CATEGORICAL_FEATURES)
    unknown_features = sorted(set(features) - known_features)
    if unknown_features:
        raise ValueError(f"Unknown v2 features: {unknown_features}")

    numeric_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, numeric_features),
            ("categorical", categorical_preprocessor, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_preprocessor(config):
    version = config.get("preprocessing", {}).get("version", "v1")

    if version == "v1":
        return get_preprocessor_v1(config)
    if version == "v2":
        return get_preprocessor_v2(config)

    raise ValueError(f"Unknown preprocessing version: {version}")
