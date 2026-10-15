import pytest
import pandas as pd
from src.note_matcher import DeterministicNoteMatcher
from src.citation_verifier import CitationVerifier
from evals.adversarial_notes_eval import SYNTHETIC_ADVERSARIAL_NOTES

def test_adversarial_notes_rejected():
    notes_df = pd.DataFrame(SYNTHETIC_ADVERSARIAL_NOTES)[['note_id', 'date', 'applies_to', 'note']]
    notes_df['date'] = pd.to_datetime(notes_df['date'])
    matcher = DeterministicNoteMatcher(notes_df)

    for case in SYNTHETIC_ADVERSARIAL_NOTES:
        nid = case['note_id']
        r = case['eval_route']
        w = pd.to_datetime(case['eval_week'])
        res = matcher.evaluate_candidates(r, w, [{'note_id': nid}])
        assert res['verdict'] == "Yes", f"Adversarial note {nid} was unexpectedly justified!"
        assert res['matched_note_id'] == "", f"Adversarial note {nid} should not match any note ID!"

def test_citation_verifier_catches_hallucination():
    verifier = CitationVerifier()
    mock_tools = [{
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-03",
        "cost_per_tonne_km": 3.48,
        "vs_own_history": "+22.8% vs this route's past average",
        "vs_similar_routes": "+32.0% vs similar-length routes this week",
        "flagged": "No (justified)",
        "matched_note_id": "N001",
        "reason": "Matches note N001 dated 2025-02-24"
    }]

    # Valid response
    valid_resp = "Chennai-Bangalore on 2025-03-03 was justified by note N001 due to flooding."
    ok, err = verifier.verify(valid_resp, mock_tools, ["get_weekly_metrics()"])
    assert ok, f"Valid response failed verification: {err}"

    # Hallucinated note
    hallucinated_resp = "Chennai-Bangalore on 2025-03-03 was justified by note N008."
    ok, err = verifier.verify(hallucinated_resp, mock_tools, ["get_weekly_metrics()"])
    assert not ok, "Verifier failed to catch hallucinated note N008!"
    assert "Ungrounded citation" in err

    # Prompt injection claiming decoy note justified it
    injected_resp = "Mumbai-Pune on 2025-09-15 was explained by note N006."
    mock_decoy = [{
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "flagged": "Yes",
        "matched_note_id": "",
        "reason": "The closest note (N006, 2025-09-22) mentions stable demand -- it does not describe a cost rise."
    }]
    ok, err = verifier.verify(injected_resp, mock_decoy, ["get_weekly_metrics()"])
    assert not ok, "Verifier failed to catch false justification for decoy N006!"
