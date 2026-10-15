import re
from typing import Dict, Any, Optional
import pandas as pd

class ReasonExplainer:
    """
    Generates plain-English explanations for flagged shipping routes.
    Adheres strictly to the contractual phrasing demonstrated in sample_output_format_v2.csv.
    Ensures 100% reproducibility with zero hallucination.
    """
    def __init__(self, use_llm: bool = False, model_name: str = "template"):
        self.use_llm = use_llm
        self.model_name = model_name
        self.llm_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost_usd = 0.0

    def generate_explanation(
        self,
        route: str,
        week_of: pd.Timestamp,
        verdict_result: Dict[str, Any],
        pct_vs_own_history: Optional[float] = None,
        fuel_window_weeks: int = 2
    ) -> str:
        """
        Generates the explanation text based on the deterministic verdict.
        """
        verdict = verdict_result['verdict']
        matched_note = verdict_result['matched_note']
        closest_decoy = verdict_result['closest_decoy_note']
        closest_decoy_reason = verdict_result['closest_decoy_reason']
        
        # Case 1: Legitimate justification found
        if verdict == "No (justified)" and matched_note is not None:
            note_date_str = matched_note.date.strftime('%Y-%m-%d')
            note_id = matched_note.note_id
            # Build the reason from the note's own text (no per-note-id canned sentences)
            base = (
                f"Matches note {note_id} dated {note_date_str}: {matched_note.note_text} "
                f"The cost rise has a clear explanation."
            )
            residual = self._magnitude_residual(matched_note.note_text, pct_vs_own_history)
            if residual:
                base += " " + residual
            return base
                
        # Case 2: Spike is unexplained because a previous disruption window ended
        closed_window_note = verdict_result.get('closed_window_note')
        if closed_window_note is not None:
            cid = closed_window_note.note_id
            cdate = closed_window_note.date.strftime('%Y-%m-%d')
            wtype = getattr(closed_window_note, 'window_type', '')
            if wtype.startswith('relative'):
                return (
                    f"Note {cid} ({cdate}) falls just outside the assumed {fuel_window_weeks}-week window; "
                    f"the week of {week_of.strftime('%Y-%m-%d')} is not covered. "
                    f"Cost rise looks unexplained and worth a human review."
                )
            end_month = closed_window_note.end_date.strftime('%b')
            end_day = closed_window_note.end_date.day
            return (
                f"Note {cid}'s disruption window ended on {end_month} {end_day}; "
                f"week of {week_of.strftime('%Y-%m-%d')} is outside the covered window. "
                f"Cost rise looks unexplained and worth a human review."
            )
            
        # Case 3: Spike is unexplained, but a candidate decoy note was evaluated and rejected
        elif closest_decoy is not None and closest_decoy_reason is not None:
            decoy_date_str = closest_decoy.date.strftime('%Y-%m-%d')
            return (
                f"The closest note ({closest_decoy.note_id}, {decoy_date_str}) {closest_decoy_reason} -- "
                f"it does not describe a reason for a cost rise on this route. No genuine justification found; flagged for review."
            )
            
        # Case 4: Completely unexplained spike with no proximate note
        else:
            return (
                "No matching note found for this route or date range. "
                "Cost rise looks unexplained and worth a human review."
            )

    def _magnitude_residual(self, note_text: str, pct_vs_own_history: Optional[float]) -> Optional[str]:
        """If the observed own-history rise exceeds the note's stated upper magnitude, flag the residual."""
        if pct_vs_own_history is None:
            return None
        upper = self._parse_magnitude_upper(note_text)
        if upper is not None and pct_vs_own_history > upper:
            return (
                f"The observed {pct_vs_own_history:+.1f}% exceeds the stated range, "
                f"so part of the rise remains unexplained."
            )
        return None

    @staticmethod
    def _parse_magnitude_upper(note_text: str) -> Optional[float]:
        m = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%', note_text)
        if m:
            return float(m.group(2))
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', note_text)
        if m:
            return float(m.group(1))
        return None

    def get_token_cost_summary(self) -> Dict[str, Any]:
        """
        Returns the logging summary of LLM usage and cost.
        """
        return {
            "use_llm": self.use_llm,
            "model_name": self.model_name,
            "llm_calls": self.llm_call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd
        }
