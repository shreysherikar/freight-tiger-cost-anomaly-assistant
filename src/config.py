from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration parameters for shipping cost anomaly detection pipeline.
    All thresholds and paths are centralized here for easy tuning and defense.
    """
    shipment_records_path: str = "shipment_records.csv"
    context_notes_path: str = "context_notes.csv"
    output_path: str = "output.csv"
    
    # Flagging thresholds:
    # A route-week is flagged if cost is rising (pct_vs_own > 0)
    # AND (pct_vs_own >= threshold_own_pct OR pct_vs_peer >= threshold_peer_pct)
    threshold_own_pct: float = 6.0
    threshold_peer_pct: float = 20.0
    
    # Baseline comparison parameters
    rolling_history_weeks: int = 8
    fuel_surcharge_window_weeks: int = 2
    
    # Output scope: 'flagged_only' outputs rows audited for justification,
    # exactly matching the contract in sample_output_format_v2.csv.
    output_scope: str = "flagged_only"
    
    # Optional LLM rewording configuration
    use_llm: bool = False
    live: bool = False
    cost_log_path: str = "cost_log.json"
