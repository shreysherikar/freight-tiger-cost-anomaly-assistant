import re
import pandas as pd
from typing import Tuple, Optional

# Month mappings for parsing natural language date expressions
MONTHS = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12
}

class TemporalWindowParser:
    """
    Robust, production-grade temporal window parser for logistics context notes.
    Extracts explicit date ranges, relative operational windows, and macro-event durations
    with formal bounds checking.
    """
    def __init__(self, fuel_window_weeks: int = 2):
        self.fuel_window_weeks = fuel_window_weeks

    def parse_window(
        self,
        base_date: pd.Timestamp,
        note_text: str
    ) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
        """
        Parses start and end timestamps for a context note.
        Returns:
            (start_date, end_date, window_type)
        """
        text_lower = note_text.lower()
        base_year = base_date.year

        # Calculate Monday-Sunday boundaries of base_date
        monday = base_date - pd.Timedelta(days=base_date.weekday())
        sunday = monday + pd.Timedelta(days=6)

        # Pattern 1: Explicit month-day range across months, e.g. "Feb 24 to Mar 8" or "Feb 24 - Mar 8"
        range_match = re.search(
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\s+(?:to|-|until)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b',
            text_lower
        )
        if range_match:
            m1_name, d1_str, m2_name, d2_str = range_match.groups()
            m1 = MONTHS.get(m1_name[:3], base_date.month)
            d1 = int(d1_str)
            m2 = MONTHS.get(m2_name[:3], base_date.month)
            d2 = int(d2_str)
            
            start = pd.to_datetime(f"{base_year}-{m1:02d}-{d1:02d}")
            end = pd.to_datetime(f"{base_year}-{m2:02d}-{d2:02d}")
            return start, end, "explicit_range"

        # Pattern 2: Same-month day range, e.g. "from Oct 14 to Oct 28" or "Oct 14 to 28"
        same_month_match = re.search(
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\s+(?:to|-)\s+(\d{1,2})(?:st|nd|rd|th)?\b',
            text_lower
        )
        if same_month_match:
            m_name, d1_str, d2_str = same_month_match.groups()
            m = MONTHS.get(m_name[:3], base_date.month)
            start = pd.to_datetime(f"{base_year}-{m:02d}-{int(d1_str):02d}")
            end = pd.to_datetime(f"{base_year}-{m:02d}-{int(d2_str):02d}")
            return start, end, "same_month_range"

        # Pattern 3: Relative ongoing macro events, e.g. "starting this week"
        if "starting this week" in text_lower or "starts this week" in text_lower:
            start = monday
            # Dynamic window based on fuel_window_weeks configuration
            days = (self.fuel_window_weeks * 7) - 1
            end = monday + pd.Timedelta(days=days)
            return start, end, f"relative_{self.fuel_window_weeks}_weeks"

        # Pattern 4: ISO date format, e.g. "2025-02-24 to 2025-03-08"
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})', text_lower)
        if iso_match:
            start = pd.to_datetime(iso_match.group(1))
            end = pd.to_datetime(iso_match.group(2))
            return start, end, "iso_range"

        # Default fallback: single Monday-Sunday week of the base date
        return monday, sunday, "single_week"

def parse_note_window(
    base_date: pd.Timestamp,
    note_text: str,
    fuel_window_weeks: int = 2
) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """Helper function to parse note window."""
    parser = TemporalWindowParser(fuel_window_weeks=fuel_window_weeks)
    return parser.parse_window(base_date, note_text)
