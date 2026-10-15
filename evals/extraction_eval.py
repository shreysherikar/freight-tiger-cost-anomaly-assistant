import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm_client import MockLLMClient, get_llm_client
from src.note_extractor import NoteExtractor
from src.temporal_parser import TemporalWindowParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_date_interval_f1(pred_start: Optional[str], pred_end: Optional[str],
                             true_start: Optional[str], true_end: Optional[str]) -> float:
    """Computes daily overlap F1 between predicted and ground-truth date windows."""
    if not true_start:
        return 1.0 if not pred_start else 0.0
    if not pred_start:
        return 0.0

    try:
        t_s = datetime.strptime(true_start, "%Y-%m-%d").date()
        t_e = datetime.strptime(true_end, "%Y-%m-%d").date() if true_end else t_s
        p_s = datetime.strptime(pred_start, "%Y-%m-%d").date()
        p_e = datetime.strptime(pred_end, "%Y-%m-%d").date() if pred_end else p_s

        true_days = set(pd.date_range(t_s, t_e))
        pred_days = set(pd.date_range(p_s, p_e))

        intersection = len(true_days.intersection(pred_days))
        if len(true_days) + len(pred_days) == 0:
            return 1.0
        f1 = (2.0 * intersection) / (len(true_days) + len(pred_days))
        return f1
    except Exception:
        return 0.0

def evaluate_extraction(labels_path: str = "labels/notes_labels.json",
                        notes_path: str = "context_notes.csv",
                        use_mock: bool = True) -> Dict[str, Any]:
    if not os.path.exists(labels_path):
        print("0 labeled notes found, skipping score")
        return {"status": "skipped", "labeled_count": 0}

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    notes_dict = labels_data.get("notes", {})
    # Check if any note has non-null labels
    labeled_notes = {}
    for nid, fields in notes_dict.items():
        if any(v is not None for v in fields.values()):
            labeled_notes[nid] = fields

    if not labeled_notes:
        print("0 labeled notes found, skipping score")
        return {"status": "skipped", "labeled_count": 0}

    notes_df = pd.read_csv(notes_path)
    if not use_mock:
        from src.llm_client import _load_env_if_present
        _load_env_if_present()
    client = MockLLMClient() if use_mock else get_llm_client(live=True)
    extractor = NoteExtractor(llm_client=client)
    rule_parser = TemporalWindowParser()

    llm_scores = {"applies_to_corr": 0, "date_f1_sum": 0.0, "cost_effect_corr": 0, "magnitude_corr": 0}
    rule_scores = {"applies_to_corr": 0, "date_f1_sum": 0.0, "cost_effect_corr": 0, "magnitude_corr": 0}
    n = len(labeled_notes)

    print(f"\n--- Evaluating Extraction against {n} labeled notes ---")

    for nid, ground_truth in labeled_notes.items():
        row = notes_df[notes_df["note_id"] == nid]
        if row.empty:
            continue
        row_data = row.iloc[0]
        note_date = pd.to_datetime(row_data["date"])
        applies_to = row_data["applies_to"]
        note_text = row_data["note"]

        # LLM Extractor
        llm_pred = extractor.extract_note(nid, note_date, applies_to, note_text)

        # Rule Extractor
        r_start, r_end, _ = rule_parser.parse_window(note_date, note_text)
        r_effect = "increase" if extractor._rule_check_cost_increase(note_text) else "no_change"

        # 1. applies_to
        if ground_truth.get("applies_to"):
            target_route = ground_truth["applies_to"].strip().lower()
            if str(llm_pred.get("applies_to", "")).strip().lower() == target_route:
                llm_scores["applies_to_corr"] += 1
            if applies_to.strip().lower() == target_route:
                rule_scores["applies_to_corr"] += 1

        # 2. date overlap F1
        llm_f1 = compute_date_interval_f1(llm_pred.get("start_date"), llm_pred.get("end_date"),
                                          ground_truth.get("start_date"), ground_truth.get("end_date"))
        r_start_str = r_start.strftime("%Y-%m-%d") if r_start else None
        r_end_str = r_end.strftime("%Y-%m-%d") if r_end else None
        rule_f1 = compute_date_interval_f1(r_start_str, r_end_str,
                                           ground_truth.get("start_date"), ground_truth.get("end_date"))
        llm_scores["date_f1_sum"] += llm_f1
        rule_scores["date_f1_sum"] += rule_f1

        # 3. cost_effect
        if ground_truth.get("cost_effect"):
            target_effect = ground_truth["cost_effect"].strip().lower()
            if str(llm_pred.get("cost_effect", "")).strip().lower() == target_effect:
                llm_scores["cost_effect_corr"] += 1
            if r_effect == target_effect:
                rule_scores["cost_effect_corr"] += 1

        # 4. magnitude coverage
        if ground_truth.get("magnitude_range"):
            target_mag = str(ground_truth["magnitude_range"]).strip().lower()
            if target_mag in str(llm_pred.get("magnitude_range", "")).strip().lower():
                llm_scores["magnitude_corr"] += 1
            # Rule parser doesn't extract magnitude by default, checks text
            if target_mag in note_text.lower():
                rule_scores["magnitude_corr"] += 1

    summary = {
        "status": "evaluated",
        "labeled_count": n,
        "llm_extractor": {
            "applies_to_accuracy": round(llm_scores["applies_to_corr"] / n, 4),
            "date_overlap_f1": round(llm_scores["date_f1_sum"] / n, 4),
            "cost_effect_accuracy": round(llm_scores["cost_effect_corr"] / n, 4),
            "magnitude_coverage": round(llm_scores["magnitude_corr"] / n, 4)
        },
        "rule_parser": {
            "applies_to_accuracy": round(rule_scores["applies_to_corr"] / n, 4),
            "date_overlap_f1": round(rule_scores["date_f1_sum"] / n, 4),
            "cost_effect_accuracy": round(rule_scores["cost_effect_corr"] / n, 4),
            "magnitude_coverage": round(rule_scores["magnitude_corr"] / n, 4)
        }
    }

    print("LLM Extractor Metrics:", summary["llm_extractor"])
    print("Rule Parser Metrics:  ", summary["rule_parser"])
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate note extraction against ground truth labels.")
    parser.add_argument("--labels", default="labels/notes_labels.json", help="Path to ground truth labels JSON")
    parser.add_argument("--notes", default="context_notes.csv", help="Path to context notes CSV")
    parser.add_argument("--use-real-llm", action="store_true", help="Use live LLM instead of mock")
    args = parser.parse_args()

    evaluate_extraction(labels_path=args.labels, notes_path=args.notes, use_mock=not args.use_real_llm)
