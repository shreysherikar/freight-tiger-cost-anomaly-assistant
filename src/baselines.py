import pandas as pd
import numpy as np

def compute_baselines(weekly_df: pd.DataFrame, rolling_window: int = 8) -> pd.DataFrame:
    """
    Computes comparison baselines exactly as specified:
    1. vs_own_history: Trailing rolling average of this route's weekly cost per tonne-km,
       using only weeks strictly before the current week (shift(1), rolling(window, min_periods=1)).
       If fewer than 8 prior weeks exist, all available prior weeks are averaged.
       Week 1 has no history and will have NaN.
    2. vs_similar_routes: Average cost per tonne-km that week across all OTHER routes
       sharing the same route_type (Short / Medium / Long), excluding the route itself.
    """
    df = weekly_df.sort_values(by=['route', 'week_of']).copy()
    
    # Baseline 1: Own history
    df['own_history_baseline'] = df.groupby('route')['cost_per_tonne_km'].transform(
        lambda s: s.shift(1).rolling(window=rolling_window, min_periods=1).mean()
    )
    df['pct_vs_own_history'] = (
        (df['cost_per_tonne_km'] - df['own_history_baseline']) / df['own_history_baseline'] * 100.0
    )
    
    # Baseline 2: Similar routes (peers)
    peer_averages = []
    for idx, row in df.iterrows():
        peers = df[
            (df['week_of'] == row['week_of']) &
            (df['route_type'] == row['route_type']) &
            (df['route'] != row['route'])
        ]
        if len(peers) > 0:
            peer_avg = peers['cost_per_tonne_km'].mean()
            peer_pct = (row['cost_per_tonne_km'] - peer_avg) / peer_avg * 100.0
        else:
            peer_avg = np.nan
            peer_pct = np.nan
            
        peer_averages.append((peer_avg, peer_pct))
        
    df['peer_baseline'] = [p[0] for p in peer_averages]
    df['pct_vs_similar_routes'] = [p[1] for p in peer_averages]
    
    return df

def format_deviation_own(pct_val: float) -> str:
    """Formats percentage deviation vs own history matching contract: +35.5% vs this route's past average"""
    if pd.isna(pct_val):
        return "N/A (week 1 has no history)"
    return f"{pct_val:+.1f}% vs this route's past average"

def format_deviation_peer(pct_val: float) -> str:
    """Formats percentage deviation vs peer routes matching contract: +21.0% vs similar-length routes this week"""
    if pd.isna(pct_val):
        return "N/A (no peer routes)"
    return f"{pct_val:+.1f}% vs similar-length routes this week"
