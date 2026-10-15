import pandas as pd
import numpy as np
from collections import deque, defaultdict
from typing import Dict, Any, List, Tuple
from src.data_loader import load_shipments

class StreamingCostAggregator:
    """
    Incremental streaming aggregator that replays shipment events chronologically.
    Maintains running sums for weekly aggregation and rolling windows using collections.deque.
    """
    def __init__(self, rolling_window: int = 8):
        self.rolling_window = rolling_window
        
        # State per route: deque of past weekly cost_per_tonne_km (max length rolling_window)
        self.route_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.rolling_window))
        
        # Current active week accumulator: (route, route_type, week_of) -> {cost, tonne_km}
        self.current_week_acc = defaultdict(lambda: {"cost": 0.0, "tonne_km": 0.0, "shipments": 0})
        
        # Final completed weekly records
        self.completed_weeks: List[Dict[str, Any]] = []

    def process_shipment_stream(self, shipments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replays shipments sorted strictly by shipment_date.
        """
        # Sort chronologically
        sorted_df = shipments_df.sort_values(by=['shipment_date', 'shipment_id']).copy()
        
        # First group by week and route as shipments arrive
        current_week_marker = None
        
        # We group arrivals into chronological weekly epochs
        unique_weeks = sorted(sorted_df['week_of'].unique())
        
        for w in unique_weeks:
            week_shipments = sorted_df[sorted_df['week_of'] == w]
            
            # Step 1: Accumulate weekly sums for this week across all active routes
            week_metrics = {}
            for _, row in week_shipments.iterrows():
                key = (row['route'], row['route_type'])
                if key not in week_metrics:
                    week_metrics[key] = {"cost": 0.0, "tonne_km": 0.0, "count": 0}
                week_metrics[key]["cost"] += row['freight_cost_inr']
                week_metrics[key]["tonne_km"] += row['quantity_tonnes'] * row['distance_km']
                week_metrics[key]["count"] += 1
                
            # Step 2: Compute weekly cost_per_tonne_km and record baselines BEFORE updating deques
            week_records = []
            for (route, r_type), acc in week_metrics.items():
                cptk = acc["cost"] / acc["tonne_km"]
                
                # Baseline 1: Trailing rolling average from deque (strictly prior weeks)
                hist_deque = self.route_history[route]
                if len(hist_deque) > 0:
                    own_baseline = float(np.mean(hist_deque))
                    pct_own = (cptk - own_baseline) / own_baseline * 100.0
                else:
                    own_baseline = np.nan
                    pct_own = np.nan
                    
                week_records.append({
                    "route": route,
                    "route_type": r_type,
                    "week_of": w,
                    "cost_per_tonne_km": cptk,
                    "total_freight_cost": acc["cost"],
                    "total_tonne_km": acc["tonne_km"],
                    "own_history_baseline": own_baseline,
                    "pct_vs_own_history": pct_own
                })
                
            # Step 3: Compute same-week peer baselines
            for rec in week_records:
                peers = [
                    other["cost_per_tonne_km"]
                    for other in week_records
                    if other["route_type"] == rec["route_type"] and other["route"] != rec["route"]
                ]
                if len(peers) > 0:
                    peer_avg = float(np.mean(peers))
                    pct_peer = (rec["cost_per_tonne_km"] - peer_avg) / peer_avg * 100.0
                else:
                    peer_avg = np.nan
                    pct_peer = np.nan
                rec["peer_baseline"] = peer_avg
                rec["pct_vs_similar_routes"] = pct_peer
                self.completed_weeks.append(rec)
                
            # Step 4: Advance deques for each route with the finalized week's metric
            for rec in week_records:
                self.route_history[rec["route"]].append(rec["cost_per_tonne_km"])
                
        out_df = pd.DataFrame(self.completed_weeks)
        return out_df.sort_values(by=['route', 'week_of']).reset_index(drop=True)
