import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.config import PipelineConfig
from src.pipeline import AnomalyDetectionPipeline

def test_verdict_and_numeric_columns_parity():
    """
    Proves that enabling the LLM explainer rewrites wording only,
    and NEVER changes any route, week_of, cost, baseline, verdict, or matched note id.
    """
    # 1. Run pipeline without LLM
    config_no_llm = PipelineConfig(use_llm=False)
    pipeline_no_llm = AnomalyDetectionPipeline(config_no_llm)
    df_no_llm, _ = pipeline_no_llm.run()
    
    # 2. Run pipeline with LLM (mock / offline rewriter)
    config_llm = PipelineConfig(use_llm=True)
    pipeline_llm = AnomalyDetectionPipeline(config_llm)
    df_llm, _ = pipeline_llm.run()
    
    # Non-reason columns
    contract_cols = [
        'route', 'week_of', 'cost_per_tonne_km',
        'vs_own_history', 'vs_similar_routes',
        'flagged', 'matched_note_id'
    ]
    
    # Assert exact dataframe equality on all decision and numeric columns
    assert df_no_llm[contract_cols].equals(df_llm[contract_cols]), (
        "Verdict or numeric columns differed between LLM ON and LLM OFF!"
    )
    print("Verdict and numeric columns are 100% IDENTICAL with LLM on vs off.")

if __name__ == "__main__":
    test_verdict_and_numeric_columns_parity()
