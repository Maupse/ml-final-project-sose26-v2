"""Scaffold for feature construction utilities."""

import numpy as np
import pandas as pd

MUNICH_CENTER_LATITUDE = 48.13737881325798
MUNICH_CENTER_LONGITUDE = 11.575437772781985
EARTH_RADIUS_KM = 6371


def haversine_km(
    lat1: float | pd.Series,
    lon1: float | pd.Series,
    lat2: float | pd.Series,
    lon2: float | pd.Series,
) -> float | pd.Series:
    """Calculate the Haversine distance in kilometers between two coordinates."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def add_distance_to_city_center(data: pd.DataFrame) -> pd.DataFrame:
    """Add distance to Munich city center in kilometers, mutating the input data."""
    data["distance_km"] = haversine_km(
        MUNICH_CENTER_LATITUDE,
        MUNICH_CENTER_LONGITUDE,
        data["latitude"],
        data["longitude"],
    )
    return data
