import pytest
import pandas as pd
import numpy as np
from src.config import PipelineConfig
from src.data_loader import load_shipments, load_context_notes
from src.metrics import compute_weekly_route_metrics
from src.baselines import compute_baselines
from src.flagging import identify_candidate_spikes
from src.retriever import NoteRetriever
from src.note_matcher import DeterministicNoteMatcher, NoteMetadata
from src.pipeline import AnomalyDetectionPipeline

def test_data_loader():
    df_shipments = load_shipments("shipment_records.csv")
    assert len(df_shipments) == 2940
    assert 'route' in df_shipments.columns
    assert 'week_of' in df_shipments.columns
    assert 'tonne_km' in df_shipments.columns
    
    df_notes = load_context_notes("context_notes.csv")
    assert len(df_notes) == 10
    assert 'note_id' in df_notes.columns

def test_ratio_of_sums_metrics():
    df_shipments = load_shipments("shipment_records.csv")
    weekly = compute_weekly_route_metrics(df_shipments)
    
    # Check Delhi-Jaipur on 2024-11-11
    dj = weekly[(weekly['route'] == 'Delhi-Jaipur') & (weekly['week_of'] == '2024-11-11')]
    assert len(dj) == 1
    assert round(dj.iloc[0]['cost_per_tonne_km'], 2) == 4.17
    
    # Check Ahmedabad-Mumbai on 2025-01-20
    am = weekly[(weekly['route'] == 'Ahmedabad-Mumbai') & (weekly['week_of'] == '2025-01-20')]
    assert len(am) == 1
    assert round(am.iloc[0]['cost_per_tonne_km'], 2) == 3.29

def test_baselines_exact_match():
    df_shipments = load_shipments("shipment_records.csv")
    weekly = compute_weekly_route_metrics(df_shipments)
    baselines = compute_baselines(weekly, rolling_window=8)
    
    # Delhi-Jaipur 2024-11-11
    dj = baselines[(baselines['route'] == 'Delhi-Jaipur') & (baselines['week_of'] == '2024-11-11')].iloc[0]
    assert f"{dj['pct_vs_own_history']:+.1f}%" == "+35.5%"
    assert f"{dj['pct_vs_similar_routes']:+.1f}%" == "+21.0%"
    
    # Ahmedabad-Mumbai 2025-01-20
    am = baselines[(baselines['route'] == 'Ahmedabad-Mumbai') & (baselines['week_of'] == '2025-01-20')].iloc[0]
    assert f"{am['pct_vs_own_history']:+.1f}%" == "+29.5%"
    assert f"{am['pct_vs_similar_routes']:+.1f}%" == "+22.5%"

def test_chennai_boundary_and_decoys():
    df_notes = load_context_notes("context_notes.csv")
    matcher = DeterministicNoteMatcher(df_notes)
    retriever = NoteRetriever(df_notes)
    
    # Week 2025-02-24: in flood window
    cands_1 = retriever.retrieve_candidates("Chennai-Bangalore", pd.to_datetime("2025-02-24"))
    res_1 = matcher.evaluate_candidates("Chennai-Bangalore", pd.to_datetime("2025-02-24"), cands_1)
    assert res_1['verdict'] == "No (justified)"
    assert res_1['matched_note_id'] == "N001"
    
    # Week 2025-03-10: flood ended Mar 8 -> must be Yes (unjustified)
    cands_2 = retriever.retrieve_candidates("Chennai-Bangalore", pd.to_datetime("2025-03-10"))
    res_2 = matcher.evaluate_candidates("Chennai-Bangalore", pd.to_datetime("2025-03-10"), cands_2)
    assert res_2['verdict'] == "Yes"
    assert res_2['matched_note_id'] == ""
    
    # Mumbai-Pune on 2025-09-15: N006 is a decoy -> must be Yes (unjustified)
    cands_3 = retriever.retrieve_candidates("Mumbai-Pune", pd.to_datetime("2025-09-15"))
    res_3 = matcher.evaluate_candidates("Mumbai-Pune", pd.to_datetime("2025-09-15"), cands_3)
    assert res_3['verdict'] == "Yes"
    assert res_3['matched_note_id'] == ""

def test_pipeline_output_contract():
    pipeline = AnomalyDetectionPipeline(PipelineConfig(output_scope="flagged_only"))
    out_df, stats = pipeline.run()
    
    expected_cols = [
        'route', 'week_of', 'cost_per_tonne_km',
        'vs_own_history', 'vs_similar_routes',
        'flagged', 'matched_note_id', 'reason'
    ]
    assert list(out_df.columns) == expected_cols
    assert len(out_df) > 0
    assert stats['token_cost_log']['llm_calls'] == 0
