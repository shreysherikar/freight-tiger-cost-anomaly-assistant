import json
import logging
import re
from typing import Dict, Any, List
from src.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

class MockLLMClient(BaseLLMClient):
    """
    Deterministic offline mock client for zero-cost testing and offline validation.
    Returns schema-valid deterministic JSON responses for extraction, judge, and agent.
    Test-only; never imported in the default production pipeline.
    """
    def __init__(self, **kwargs):
        super().__init__(model_name="mock", **kwargs)

    def _execute_call(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, stage: str = "general") -> str:
        full_text = " ".join(m.get("content", "") for m in messages).lower()

        # 1. Extraction Stage
        if "extract the following information as strict json" in full_text:
            if "n001" in full_text or "flooding" in full_text:
                return json.dumps({
                    "applies_to": "Chennai-Bangalore",
                    "start_date": "2025-02-24",
                    "end_date": "2025-03-08",
                    "cost_effect": "increase",
                    "magnitude_range": "unspecified",
                    "evidence_quote": "forcing longer detours and higher trip costs"
                })
            elif "n002" in full_text or "festival" in full_text:
                return json.dumps({
                    "applies_to": "Ahmedabad-Mumbai",
                    "start_date": "2025-01-20",
                    "end_date": "2025-01-26",
                    "cost_effect": "increase",
                    "magnitude_range": "unspecified",
                    "evidence_quote": "temporary surcharge applied by transporters"
                })
            elif "n003" in full_text or "diesel" in full_text:
                return json.dumps({
                    "applies_to": "All Routes",
                    "start_date": "2025-05-05",
                    "end_date": None,
                    "cost_effect": "increase",
                    "magnitude_range": "5-7%",
                    "evidence_quote": "pushing up transportation costs across all routes by roughly 5-7%"
                })
            elif "not part of this dataset" in full_text:
                return json.dumps({
                    "applies_to": "All Routes",
                    "start_date": "2024-03-11",
                    "end_date": "2024-03-17",
                    "cost_effect": "no_change",
                    "magnitude_range": "none",
                    "evidence_quote": "affected routes are not part of this dataset"
                })
            elif "not significantly affected" in full_text:
                return json.dumps({
                    "applies_to": "Mumbai-Delhi",
                    "start_date": "2024-07-29",
                    "end_date": "2024-08-04",
                    "cost_effect": "no_change",
                    "magnitude_range": "none",
                    "evidence_quote": "costs were not significantly affected"
                })
            elif "without a rate change" in full_text:
                return json.dumps({
                    "applies_to": "All Routes",
                    "start_date": "2025-10-27",
                    "end_date": "2025-11-02",
                    "cost_effect": "no_change",
                    "magnitude_range": "none",
                    "evidence_quote": "absorbed by transporters without a rate change"
                })
            else:
                return json.dumps({
                    "applies_to": "unspecified",
                    "start_date": "2025-01-01",
                    "end_date": None,
                    "cost_effect": "unclear",
                    "magnitude_range": "unspecified",
                    "evidence_quote": "operational conditions"
                })

        # 2. Structured Explanation Stage
        if "generate a structured explanation in json format" in full_text:
            note_match = re.search(r'matched note id:\s*(n\d+)', full_text)
            note_id = note_match.group(1).upper() if note_match else None
            return json.dumps({
                "summary": f"Route cost deviation analyzed; {f'justified by note {note_id}' if note_id else 'unexplained anomaly'}.",
                "drivers": [
                    {"factor": "vs_own_history", "direction": "rising", "percentage": "+35.5%"}
                ],
                "note_cited": note_id,
                "verdict_reaffirmed": True
            })

        # 3. Judge Stage
        if "judge whether the explanation claims are supported" in full_text:
            if "unsupported" in full_text or "contradicts" in full_text:
                return json.dumps({
                    "is_supported": False,
                    "verdict_veto": True,
                    "unsupported_reasons": ["Explanation claims justification contradicting note facts"],
                    "magnitude_check": "within_range"
                })
            return json.dumps({
                "is_supported": True,
                "verdict_veto": False,
                "unsupported_reasons": [],
                "magnitude_check": "within_range"
            })

        # 4. Agent Planner Stage
        if "tool_call:" in full_text or "autonomous tool-calling" in full_text:
            return "FINAL_ANSWER: Analysis confirmed via pipeline records."

        # Default fallback response
        return "Deterministic canned response from MockLLMClient."
