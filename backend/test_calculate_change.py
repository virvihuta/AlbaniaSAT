# Tests for the change between two years in the time series (Regis)
from classification import calculate_change


def test_calculate_change():
    last_year = {"cropland": 0.26, "water": 0.05}
    this_year = {"cropland": 0.24, "water": 0.05}
    assert calculate_change(last_year, this_year) == {"cropland": -0.02, "water": 0.0}


def test_calculate_change_first_year():
    assert calculate_change(None, {"water": 0.05}) is None