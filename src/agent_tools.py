import pandas as pd
from typing import Dict, Any, List, Optional
from src.data_loader import load_context_notes

class AssistantTools:
    """
    Deterministic domain tools exposing pipeline records and context notes
    to the assistant agent.
    """
    def __init__(self, output_csv_path: str = "output.csv", notes_csv_path: str = "context_notes.csv"):
        self.output_df = pd.read_csv(output_csv_path)
        self.output_df['week_of_dt'] = pd.to_datetime(self.output_df['week_of'])
        
        self.notes_df = load_context_notes(notes_csv_path)
        self.notes_map = {r['note_id']: r for _, r in self.notes_df.iterrows()}
        
        self.valid_routes = sorted(self.output_df['route'].unique().tolist())

    def get_flagged_weeks(self, route: Optional[str] = None) -> List[Dict[str, Any]]:
        """Tool 1: Returns all flagged weeks for a route (or all routes)."""
        df = self.output_df
        if route:
            df = df[df['route'] == route]
        return df.to_dict(orient="records")

    def get_weekly_metrics(self, route: str, week_of: str) -> Optional[Dict[str, Any]]:
        """Tool 2: Returns metric row for a specific route and week."""
        sub = self.output_df[(self.output_df['route'] == route) & (self.output_df['week_of'] == week_of)]
        if len(sub) == 0:
            return None
        return sub.iloc[0].to_dict()

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Tool 3: Returns context note metadata by note ID."""
        note_id = note_id.strip().upper()
        if note_id in self.notes_map:
            row = self.notes_map[note_id]
            return {
                "note_id": row['note_id'],
                "date": row['date'].strftime('%Y-%m-%d'),
                "applies_to": row['applies_to'],
                "note": row['note']
            }
        return None

    def compare_to_peers(self, route: str, week_of: str) -> Optional[Dict[str, Any]]:
        """Tool 4: Returns peer comparison details for a route and week."""
        row = self.get_weekly_metrics(route, week_of)
        if not row:
            return None
        return {
            "route": route,
            "week_of": week_of,
            "cost_per_tonne_km": row['cost_per_tonne_km'],
            "vs_similar_routes": row['vs_similar_routes'],
            "flagged": row['flagged']
        }
