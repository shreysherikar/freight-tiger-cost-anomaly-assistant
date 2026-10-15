import pytest
import pandas as pd
from src.temporal_parser import TemporalWindowParser, parse_note_window

def test_explicit_cross_month_range():
    base_date = pd.to_datetime("2025-02-24")
    text = "Heavy flooding disrupted normal truck movement from Feb 24 to Mar 8, forcing longer detours."
    start, end, wtype = parse_note_window(base_date, text)
    assert start == pd.to_datetime("2025-02-24")
    assert end == pd.to_datetime("2025-03-08")
    assert wtype == "explicit_range"

def test_same_month_range():
    base_date = pd.to_datetime("2024-10-14")
    text = "Highway maintenance was scheduled from Oct 14 to 28."
    start, end, wtype = parse_note_window(base_date, text)
    assert start == pd.to_datetime("2024-10-14")
    assert end == pd.to_datetime("2024-10-28")
    assert wtype == "same_month_range"

def test_relative_ongoing_fuel_window():
    base_date = pd.to_datetime("2025-05-05")  # Monday
    text = "Diesel prices rose nationwide starting this week, pushing up transportation costs."
    
    # Default 2-week window (14 days total: May 5 to May 18)
    start_2w, end_2w, wtype_2w = parse_note_window(base_date, text, fuel_window_weeks=2)
    assert start_2w == pd.to_datetime("2025-05-05")
    assert end_2w == pd.to_datetime("2025-05-18")
    assert wtype_2w == "relative_2_weeks"

    # 1-week window (7 days: May 5 to May 11)
    start_1w, end_1w, wtype_1w = parse_note_window(base_date, text, fuel_window_weeks=1)
    assert start_1w == pd.to_datetime("2025-05-05")
    assert end_1w == pd.to_datetime("2025-05-11")
    assert wtype_1w == "relative_1_weeks"

    # 3-week window (21 days: May 5 to May 25)
    start_3w, end_3w, wtype_3w = parse_note_window(base_date, text, fuel_window_weeks=3)
    assert start_3w == pd.to_datetime("2025-05-05")
    assert end_3w == pd.to_datetime("2025-05-25")
    assert wtype_3w == "relative_3_weeks"

def test_iso_date_range():
    base_date = pd.to_datetime("2025-01-01")
    text = "Route closure scheduled 2025-02-10 to 2025-02-20."
    start, end, wtype = parse_note_window(base_date, text)
    assert start == pd.to_datetime("2025-02-10")
    assert end == pd.to_datetime("2025-02-20")
    assert wtype == "iso_range"

def test_single_week_fallback():
    base_date = pd.to_datetime("2025-01-22")  # Wednesday
    text = "A regional festival week saw a temporary surcharge."
    start, end, wtype = parse_note_window(base_date, text)
    # Should resolve to Monday Jan 20 to Sunday Jan 26
    assert start == pd.to_datetime("2025-01-20")
    assert end == pd.to_datetime("2025-01-26")
    assert wtype == "single_week"
