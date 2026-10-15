import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from typing import List, Dict, Any
from src.config import PipelineConfig
from src.pipeline import AnomalyDetectionPipeline

# 16 Hand-labeled ground truth evaluation cases
HAND_LABELED_EVAL_SET = [
    {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N002",
        "test_category": "Genuine Note (Festival Surcharge)"
    },
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-02-24",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "test_category": "Genuine Note (Flood Disruption Week 1)"
    },
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-03",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "test_category": "Genuine Note (Flood Disruption Week 2)"
    },
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-10",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "test_category": "Window Boundary (Flood ended Mar 8, must NOT justify)"
    },
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-17",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "test_category": "Decoy Note Rejection (N009 normal conditions)"
    },
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "test_category": "Unexplained Spike (Sample Row 1, no note)"
    },
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-18",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "test_category": "Unexplained Spike (Follow-on spike)"
    },
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "test_category": "Decoy Note Rejection (Sample Row 3, N006 stable demand)"
    },
    {
        "route": "Mumbai-Delhi",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "test_category": "Genuine Note (Nationwide Diesel Hike)"
    },
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "test_category": "Genuine Note (Nationwide Diesel Hike)"
    },
    {
        "route": "Delhi-Jaipur",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "test_category": "Genuine Note (Nationwide Diesel Hike)"
    },
    {
        "route": "Delhi-Chennai",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "test_category": "Genuine Note (Nationwide Diesel Hike)"
    },
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-10-27",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "test_category": "Decoy Note Rejection (N010 costs absorbed without rate change)"
    },
    {
        "route": "Mumbai-Delhi",
        "week_of": "2024-07-29",
        "expected_flagged": "No",
        "expected_note_id": "",
        "test_category": "Decoy Note (N005 maintenance no cost effect, unflagged)"
    },
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-05-20",
        "expected_flagged": "No",
        "expected_note_id": "",
        "test_category": "Decoy Note (N007 improved roads, unflagged)"
    },
    {
        "route": "Kolkata-Bhubaneswar",
        "week_of": "2025-06-09",
        "expected_flagged": "No",
        "expected_note_id": "",
        "test_category": "Decoy Note (N008 normal conditions, unflagged)"
    }
]

def run_evaluation():
    print("="*75)
    print("REGRESSION CHECK: HAND-LABELED EVALUATION HARNESS (16 CRITICAL ROUTE-WEEKS)")
    print("="*75)
    
    # Run full pipeline with all weeks evaluated so we can check unflagged weeks as well
    config_all = PipelineConfig(output_scope="all_weeks")
    pipeline_all = AnomalyDetectionPipeline(config_all)
    all_df, _ = pipeline_all.run()
    
    # Also run flagged-only pipeline to verify production output
    config_prod = PipelineConfig(output_scope="flagged_only")
    pipeline_prod = AnomalyDetectionPipeline(config_prod)
    prod_df, _ = pipeline_prod.run()
    
    passed_tests = 0
    total_tests = len(HAND_LABELED_EVAL_SET)
    
    print(f"{'Route':<20} {'Week Of':<12} {'Expected':<16} {'Actual':<16} {'Note':<6} {'Result'}")
    print("-" * 75)
    
    for case in HAND_LABELED_EVAL_SET:
        r = case['route']
        w = case['week_of']
        exp_flag = case['expected_flagged']
        exp_note = case['expected_note_id']
        category = case['test_category']
        
        row = all_df[(all_df['route'] == r) & (all_df['week_of'] == w)]
        if len(row) == 0:
            actual_flag = "NOT_FOUND"
            actual_note = ""
        else:
            actual_flag = row.iloc[0]['flagged']
            raw_note = row.iloc[0]['matched_note_id']
            actual_note = "" if pd.isna(raw_note) else str(raw_note)
            
        flag_match = (actual_flag == exp_flag)
        note_match = (actual_note == exp_note)
        
        is_pass = flag_match and note_match
        if is_pass:
            passed_tests += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"
            
        print(f"{r:<20} {w:<12} {exp_flag:<16} {actual_flag:<16} {actual_note:<6} {status_str} [{category}]")
        
    accuracy = (passed_tests / total_tests) * 100.0
    print("-" * 75)
    print(f"EVALUATION SUMMARY: {passed_tests}/{total_tests} Tests Passed ({accuracy:.1f}% Accuracy)")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
