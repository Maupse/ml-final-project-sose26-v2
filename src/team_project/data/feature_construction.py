"""Scaffold for feature construction utilities."""

import numpy as np
import pandas as pd
import re
import json

MUNICH_CENTER_LATITUDE = 48.13738
MUNICH_CENTER_LONGITUDE = 11.57544
EARTH_RADIUS_KM = 6371.0

def haversine_distance_km(latitude, longitude, center_latitude=MUNICH_CENTER_LATITUDE, center_longitude=MUNICH_CENTER_LONGITUDE):
    """Calculate great-circle distance from coordinates to a fixed centre."""
    latitude_radians = np.radians(pd.to_numeric(latitude, errors="coerce"))
    longitude_radians = np.radians(pd.to_numeric(longitude, errors="coerce"))
    center_latitude_radians = np.radians(center_latitude)
    center_longitude_radians = np.radians(center_longitude)

    latitude_delta = latitude_radians - center_latitude_radians
    longitude_delta = longitude_radians - center_longitude_radians
    haversine = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(center_latitude_radians)
        * np.cos(latitude_radians)
        * np.sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))


def add_distance_to_city_center(data: pd.DataFrame, feature_name="distance_to_city_center_km") -> pd.DataFrame:
    """Add distance to Munich city center in kilometers, mutating the input data."""
    data[feature_name] = haversine_distance_km(
        data["latitude"],
        data["longitude"],
    )
    return data


def parse_bathroom_count(value):
    """
    Extract a numeric bathroom count from an Airbnb bathroom label.
    In case the bathroom only has text but no count.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().casefold()
    if "half-bath" in text:
        return 0.5

    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan

def get_bathroom_privacy(value):
    if pd.isna(value):
        return "missing"

    text = str(value).casefold()
    if "private" in text:
        return "private"
    if "shared" in text:
        return "shared"
    return "unspecified"

def add_bathroom_features(data):
    """Reconcile bathroom fields and derive bathroom type indicators."""
    result = data.copy()
    bathroom_text = result["bathrooms_text"].astype("string")
    parsed_count = bathroom_text.map(parse_bathroom_count)

    result["bathroom_count"] = result["bathrooms"].combine_first(parsed_count)
    result["bathroom_privacy"] = result['bathrooms_text'].map(get_bathroom_privacy)
    result["bathroom_information_missing"] = result["bathroom_count"].isna().astype("int8")
    return result

AMENITY_PATTERNS = {
    "has_wifi": r"\bwifi\b",
    "has_kitchen": r"\bkitchen(?:ette)?\b",
    "has_parking": r"\bparking\b",
    "has_washer": r"\bwasher\b",
    "has_air_conditioning": r"\bair conditioning\b",
    "has_workspace": r"\bworkspace\b",
    "has_balcony": r"\bbalcony\b",
    "has_dryer": r"^(?:free |paid )?dryer\b",
    "has_elevator": r"\belevator\b",
    "has_smoke_alarm": r"\bsmoke alarm\b",
    "allows_pets": r"\bpets allowed\b",
}

def parse_amenities(value):
    """Parse one amenities value into a normalized immutable set."""
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return frozenset()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid amenities JSON: {value!r}") from error
    if not isinstance(value, (list, tuple, set)):
        raise TypeError("Amenities must be a JSON list or a list-like object.")
    return frozenset(str(item).strip().casefold() for item in value if str(item).strip())

def add_amenity_features(data, patterns=AMENITY_PATTERNS):
    """Convert raw amenity lists into a count and selected binary features."""
    result = data.copy()
    parsed = result["amenities"].map(parse_amenities)
    result["amenity_count"] = parsed.map(len).astype("int16")

    for feature_name, pattern in patterns.items():
        compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)
        result[feature_name] = parsed.map(
            lambda amenities: any(compiled_pattern.search(item) for item in amenities)
        ).astype("int8")
    return result

def add_license_features(data):
    """Represent license information without exposing identifier strings."""
    result = data.copy()
    license_text = result["license"].astype("string").str.strip()
    result["has_license"] = license_text.fillna("").ne("").astype("int8")
    return result

def add_capacity_features(data):
    """Preserve capacity counts and add explicit missing-value indicators."""
    result = data.copy()
    for feature in ("bedrooms", "beds"):
        result[f"{feature}_missing"] = result[feature].isna().astype("int8")
    return result
