import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
from typing import Dict, Any, List

from src.note_matcher import DeterministicNoteMatcher

# Comprehensive suite of synthetic adversarial context notes
SYNTHETIC_ADVERSARIAL_NOTES = [
    # 1. Right Route, Wrong Week
    {
        "note_id": "ADV01",
        "date": "2024-07-15",
        "applies_to": "Ahmedabad-Mumbai",
        "note": "A regional festival week saw a temporary surcharge applied by transporters on the Ahmedabad-Mumbai corridor due to high demand and limited truck availability.",
        "category": "Right Route, Wrong Week",
        "eval_route": "Ahmedabad-Mumbai",
        "eval_week": "2025-01-20"
    },
    {
        "note_id": "ADV02",
        "date": "2024-10-14",
        "applies_to": "Chennai-Bangalore",
        "note": "Heavy flooding on the Chennai-Bangalore highway disrupted normal truck movement from Oct 14 to Oct 28, forcing longer detours and higher trip costs.",
        "category": "Right Route, Wrong Week",
        "eval_route": "Chennai-Bangalore",
        "eval_week": "2025-02-24"
    },

    # 2. Wrong Route, Right Week
    {
        "note_id": "ADV03",
        "date": "2025-01-20",
        "applies_to": "Mumbai-Delhi",
        "note": "A regional festival week saw a temporary surcharge applied by transporters on the Mumbai-Delhi corridor due to high demand.",
        "category": "Wrong Route, Right Week",
        "eval_route": "Ahmedabad-Mumbai",
        "eval_week": "2025-01-20"
    },
    {
        "note_id": "ADV04",
        "date": "2025-02-24",
        "applies_to": "Delhi-Jaipur",
        "note": "Flooding caused severe detours and higher trip costs on the Delhi-Jaipur highway from Feb 24 to Mar 8.",
        "category": "Wrong Route, Right Week",
        "eval_route": "Chennai-Bangalore",
        "eval_week": "2025-02-24"
    },

    # 3. Negated Cost Rise
    {
        "note_id": "ADV05",
        "date": "2025-05-05",
        "applies_to": "All Routes",
        "note": "Diesel prices rose nationwide, but long-term corporate freight contracts ensured transportation costs did not increase across routes.",
        "category": "Negated Cost Rise",
        "eval_route": "Mumbai-Delhi",
        "eval_week": "2025-05-05"
    },
    {
        "note_id": "ADV06",
        "date": "2024-11-11",
        "applies_to": "Delhi-Jaipur",
        "note": "Diesel taxes were reviewed, but freight transport rates did not rise on the Delhi-Jaipur corridor.",
        "category": "Negated Cost Rise",
        "eval_route": "Delhi-Jaipur",
        "eval_week": "2024-11-11"
    },

    # 4. "No Increase" / Stable Conditions
    {
        "note_id": "ADV07",
        "date": "2025-09-15",
        "applies_to": "Mumbai-Pune",
        "note": "Industry analysts confirmed no increase in freight rates occurred on the Mumbai-Pune corridor during mid-September.",
        "category": "No Increase",
        "eval_route": "Mumbai-Pune",
        "eval_week": "2025-09-15"
    },
    {
        "note_id": "ADV08",
        "date": "2025-01-20",
        "applies_to": "Ahmedabad-Mumbai",
        "note": "Shippers confirmed operational costs remained unchanged with no rate increase on the Ahmedabad-Mumbai route.",
        "category": "No Increase",
        "eval_route": "Ahmedabad-Mumbai",
        "eval_week": "2025-01-20"
    },

    # 5. Vague / Ambiguous Text
    {
        "note_id": "ADV09",
        "date": "2025-03-10",
        "applies_to": "Chennai-Bangalore",
        "note": "Weather events and logistics operations were observed across southern states during the week.",
        "category": "Vague / Ambiguous",
        "eval_route": "Chennai-Bangalore",
        "eval_week": "2025-03-10"
    },
    {
        "note_id": "ADV10",
        "date": "2024-11-11",
        "applies_to": "Delhi-Jaipur",
        "note": "Economic activity continued at normal seasonal pace along northern transit corridors.",
        "category": "Vague / Ambiguous",
        "eval_route": "Delhi-Jaipur",
        "eval_week": "2024-11-11"
    },

    # 6. Cost Decrease / Discount
    {
        "note_id": "ADV11",
        "date": "2025-10-20",
        "applies_to": "Mumbai-Pune",
        "note": "New expressway toll discounts pushed freight transport costs down by 10% on the Mumbai-Pune route.",
        "category": "Cost Decrease",
        "eval_route": "Mumbai-Pune",
        "eval_week": "2025-10-20"
    },

    # 7. Disruption Outside Active Dataset
    {
        "note_id": "ADV12",
        "date": "2025-05-05",
        "applies_to": "All Routes",
        "note": "A state-wide transporter strike caused massive disruptions, but affected routes are not part of this dataset.",
        "category": "Unincluded Corridors",
        "eval_route": "Delhi-Chennai",
        "eval_week": "2025-05-05"
    }
]

def run_adversarial_eval():
    print("=" * 80)
    print("ADVERSARIAL CONTEXT NOTES EVALUATION HARNESS")
    print("Verifying that Decoys, Negations, Mismatches, and Vague Notes Justify ZERO Cases")
    print("=" * 80)

    # Convert notes to DataFrame
    notes_df = pd.DataFrame(SYNTHETIC_ADVERSARIAL_NOTES)[['note_id', 'date', 'applies_to', 'note']]
    notes_df['date'] = pd.to_datetime(notes_df['date'])

    matcher = DeterministicNoteMatcher(notes_df)

    # Check for optional external ground truth file
    external_expected = {}
    for ext_path in ["adversarial_expected.json", "adversarial_expected.csv"]:
        if os.path.exists(ext_path):
            print(f"Loading external ground truth from {ext_path}...")
            if ext_path.endswith(".json"):
                with open(ext_path, "r", encoding="utf-8") as f:
                    external_expected = json.load(f)
            else:
                ext_df = pd.read_csv(ext_path)
                external_expected = dict(zip(ext_df['note_id'], ext_df['expected_verdict']))
            break

    passed_count = 0
    total_count = len(SYNTHETIC_ADVERSARIAL_NOTES)

    print(f"{'Note ID':<8} {'Category':<24} {'Route':<18} {'Week':<12} {'Verdict':<15} {'Result'}")
    print("-" * 80)

    for case in SYNTHETIC_ADVERSARIAL_NOTES:
        nid = case['note_id']
        cat = case['category']
        r = case['eval_route']
        w = pd.to_datetime(case['eval_week'])
        
        # Evaluate candidate note
        candidate = [{'note_id': nid}]
        res = matcher.evaluate_candidates(r, w, candidate)

        actual_verdict = res['verdict']
        actual_note_id = res['matched_note_id']

        # Ground truth: every adversarial note MUST be rejected (verdict == 'Yes', matched_note_id == '')
        expected_verdict = external_expected.get(nid, "Yes")

        is_pass = (actual_verdict == expected_verdict) and (actual_note_id == "")

        if is_pass:
            passed_count += 1
            res_str = "PASS [REJECTED]"
        else:
            res_str = "FAIL [FALSE JUSTIFICATION]"

        print(f"{nid:<8} {cat:<24} {r:<18} {case['eval_week']:<12} {actual_verdict:<15} {res_str}")

    print("-" * 80)
    print(f"EVALUATION RESULT: {passed_count}/{total_count} Adversarial Tests Passed ({(passed_count/total_count)*100:.1f}%)")
    print(f"False Justification Rate: {((total_count - passed_count)/total_count)*100:.1f}% (Must be 0.0%)")
    print("=" * 80)

    return passed_count == total_count

if __name__ == "__main__":
    success = run_adversarial_eval()
    if not success:
        sys.exit(1)
