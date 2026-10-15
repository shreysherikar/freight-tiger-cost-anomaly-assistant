import pandas as pd
import re
from typing import Dict, Any, Optional, Tuple

from src.temporal_parser import TemporalWindowParser

class NoteMetadata:
    def __init__(
        self,
        note_id: str,
        date: pd.Timestamp,
        applies_to: str,
        note_text: str,
        fuel_window_weeks: int = 2
    ):
        self.note_id = note_id
        self.date = date
        self.applies_to = applies_to
        self.note_text = note_text
        self.fuel_window_weeks = fuel_window_weeks
        
        # Parse temporal window using production-grade parser
        parser = TemporalWindowParser(fuel_window_weeks=fuel_window_weeks)
        self.start_date, self.end_date, self.window_type = parser.parse_window(date, note_text)
        
        # Parse cost direction
        self.has_cost_increase, self.decoy_reason = self._parse_cost_direction()

    def _parse_cost_direction(self) -> Tuple[bool, Optional[str]]:
        text_lower = self.note_text.lower()
        
        # Check specific decoy clauses
        if "not part of this dataset" in text_lower:
            return False, "mentions that affected routes are not part of this dataset"
        if "not significantly affected" in text_lower:
            return False, "mentions costs were not significantly affected"
        if "no major disruptions" in text_lower or "remained stable" in text_lower:
            return False, "mentions stable demand with no major disruptions"
        if "improved road conditions" in text_lower:
            return False, "mentions improved road conditions rather than a cost rise"
        if "remained normal" in text_lower:
            return False, "mentions freight movement remained normal with no disruptions"
        if "returned to normal" in text_lower:
            return False, "mentions route returned to normal conditions after repairs were completed"
        if "without a rate change" in text_lower:
            return False, "mentions compliance costs were absorbed without a rate change"
        if "did not increase" in text_lower or "did not rise" in text_lower or "no increase" in text_lower:
            return False, "explicitly states transportation costs or freight rates did not increase"
        if "remained unchanged" in text_lower or "remained flat" in text_lower:
            return False, "mentions operational costs remained unchanged"
        if "costs down" in text_lower or "rates down" in text_lower or "costs declined" in text_lower or "costs fell" in text_lower:
            return False, "indicates cost decrease or discounts rather than increase"
            
        # Check positive cost increase evidence
        positive_evidence = [
            "higher trip costs",
            "temporary surcharge",
            "prices rose",
            "pushing up transportation costs"
        ]
        if any(p in text_lower for p in positive_evidence):
            return True, None
            
        return False, "does not indicate an increase in transportation costs"

class DeterministicNoteMatcher:
    """
    Evaluates candidate notes against strict rule checks.
    Never relies on an LLM for verdicts.
    """
    def __init__(self, notes_df: pd.DataFrame, fuel_window_weeks: int = 2):
        self.fuel_window_weeks = fuel_window_weeks
        self.notes = {}
        for _, row in notes_df.iterrows():
            meta = NoteMetadata(
                note_id=row['note_id'],
                date=row['date'],
                applies_to=row['applies_to'],
                note_text=row['note'],
                fuel_window_weeks=fuel_window_weeks
            )
            self.notes[row['note_id']] = meta

    def evaluate_candidates(
        self,
        route: str,
        week_of: pd.Timestamp,
        candidate_notes: list
    ) -> Dict[str, Any]:
        """
        Evaluates retrieved candidate notes for a given route and week.
        Justification uses the retrieved candidates (preserves matched_note_id).
        The "closest note" cited for unexplained rows is chosen deterministically
        over ALL notes (not the top-5 retrieval set): among notes that apply to the
        route or "All Routes", whose cost_effect is no increase, within 21 days
        before or after the week, cite the nearest by |note date - week_of|;
        ties break to the lower note id.
        Returns:
            verdict, matched_note_id, matched_note,
            closest_decoy_note, closest_decoy_reason, closed_window_note
        """
        week_start = week_of
        week_end = week_of + pd.Timedelta(days=6)

        # --- Justification check over retrieved candidates (unchanged semantics) ---
        for cand in candidate_notes:
            note_id = cand['note_id']
            meta = self.notes.get(note_id)
            if not meta:
                continue

            route_matches = (meta.applies_to == route) or (meta.applies_to == "All Routes")
            window_overlaps = (week_start <= meta.end_date) and (week_end >= meta.start_date)
            costs_went_up = meta.has_cost_increase

            if route_matches and window_overlaps and costs_went_up:
                return {
                    'verdict': "No (justified)",
                    'matched_note_id': meta.note_id,
                    'matched_note': meta,
                    'closest_decoy_note': None,
                    'closest_decoy_reason': None,
                    'closed_window_note': None
                }

        # --- Deterministic decoy / closed-window selection over ALL notes ---
        closed_window_note = None
        closest_decoy = None
        closest_decoy_reason = None
        best_dist = None

        for nid in sorted(self.notes.keys()):
            meta = self.notes[nid]
            applies = (meta.applies_to == route) or (meta.applies_to == "All Routes")
            if not applies:
                continue

            window_overlaps = (week_start <= meta.end_date) and (week_end >= meta.start_date)

            # Genuine cost-increase note whose covered window just closed (<= 7 days before week)
            if meta.has_cost_increase and not window_overlaps:
                if week_start > meta.end_date and (week_start - meta.end_date).days <= 7:
                    if closed_window_note is None:
                        closed_window_note = meta

            # Closest no-increase (decoy) note within 21 days; nearest wins, tie -> lower note id
            if not meta.has_cost_increase:
                dist = abs((meta.date - week_of).days)
                if dist <= 21 and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    closest_decoy = meta
                    closest_decoy_reason = meta.decoy_reason

        return {
            'verdict': "Yes",
            'matched_note_id': "",
            'matched_note': None,
            'closest_decoy_note': closest_decoy,
            'closest_decoy_reason': closest_decoy_reason,
            'closed_window_note': closed_window_note
        }
