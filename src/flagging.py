import pandas as pd

def identify_candidate_spikes(
    df: pd.DataFrame,
    threshold_own_pct: float = 6.0,
    threshold_peer_pct: float = 20.0
) -> pd.DataFrame:
    """
    Identifies route-weeks with unusual cost increases.
    
    Rule:
    A route-week is flagged as a candidate spike if:
      1. Cost is rising vs its own recent baseline (pct_vs_own_history > 0)
      AND
      2. It is out of the ordinary against its own history OR against similar routes:
         (pct_vs_own_history >= threshold_own_pct OR pct_vs_similar_routes >= threshold_peer_pct)
         
    Week 1 is skipped as it has no prior history.
    """
    eval_df = df.dropna(subset=['pct_vs_own_history']).copy()
    
    is_rising = eval_df['pct_vs_own_history'] > 0.0
    is_out_of_ordinary = (
        (eval_df['pct_vs_own_history'] >= threshold_own_pct) |
        (eval_df['pct_vs_similar_routes'] >= threshold_peer_pct)
    )
    
    eval_df['is_candidate_spike'] = is_rising & is_out_of_ordinary
    return eval_df
