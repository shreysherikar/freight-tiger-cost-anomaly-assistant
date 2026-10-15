import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_shipments
from src.metrics import compute_weekly_route_metrics
from src.baselines import compute_baselines
from src.streaming import StreamingCostAggregator

def test_streaming_equals_batch():
    """
    Asserts that the incremental streaming aggregator produces
    identical cost_per_tonne_km, own_history_baseline, and peer_baseline
    as the batch pandas implementation.
    """
    shipments_df = load_shipments("shipment_records.csv")
    
    # 1. Batch Computation
    batch_weekly = compute_weekly_route_metrics(shipments_df)
    batch_df = compute_baselines(batch_weekly, rolling_window=8)
    batch_df = batch_df.sort_values(by=['route', 'week_of']).reset_index(drop=True)
    
    # 2. Streaming Computation
    streaming_agg = StreamingCostAggregator(rolling_window=8)
    stream_df = streaming_agg.process_shipment_stream(shipments_df)
    stream_df = stream_df.sort_values(by=['route', 'week_of']).reset_index(drop=True)
    
    # Verify row counts
    assert len(batch_df) == len(stream_df), "Row counts differ between batch and streaming!"
    
    # Verify cost_per_tonne_km
    assert np.allclose(
        batch_df['cost_per_tonne_km'].values,
        stream_df['cost_per_tonne_km'].values,
        atol=1e-8
    ), "cost_per_tonne_km mismatch between batch and streaming!"
    
    # Verify own_history_baseline (ignoring NaNs on week 1)
    mask_own = batch_df['own_history_baseline'].notna()
    assert np.allclose(
        batch_df.loc[mask_own, 'own_history_baseline'].values,
        stream_df.loc[mask_own, 'own_history_baseline'].values,
        atol=1e-8
    ), "own_history_baseline mismatch between batch and streaming!"
    
    # Verify peer_baseline
    mask_peer = batch_df['peer_baseline'].notna()
    assert np.allclose(
        batch_df.loc[mask_peer, 'peer_baseline'].values,
        stream_df.loc[mask_peer, 'peer_baseline'].values,
        atol=1e-8
    ), "peer_baseline mismatch between batch and streaming!"
    
    # Verify percentage deviations
    assert np.allclose(
        batch_df.loc[mask_own, 'pct_vs_own_history'].values,
        stream_df.loc[mask_own, 'pct_vs_own_history'].values,
        atol=1e-7
    ), "pct_vs_own_history mismatch between batch and streaming!"
    
    print("Streaming aggregator perfectly matches batch pipeline calculations.")
