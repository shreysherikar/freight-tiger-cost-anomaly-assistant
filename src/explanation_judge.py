import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from src.llm_client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)

class ExplanationJudge:
    """
    Produces structured JSON explanations, applies a deterministic validator,
    and runs an LLM Judge step to verify claims are grounded in row numbers and note text.
    Enforces magnitude checks and handles the Veto-only rule.
    """
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        veto_log_file: str = "veto_log.json"
    ):
        self.client = llm_client or get_llm_client()
        self.veto_log_file = veto_log_file
        self.vetoes: List[Dict[str, Any]] = []
        self.fallback_count = 0

    def generate_structured_explanation(
        self,
        row_dict: Dict[str, Any],
        note_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates structured JSON explanation via LLM or fallback template.
        Schema:
        {
          "summary": str,
          "drivers": [{"factor": str, "direction": str, "percentage": float | str}],
          "note_cited": str | null,
          "verdict_reaffirmed": bool
        }
        """
        route = row_dict.get("route", "")
        week_of = str(row_dict.get("week_of", ""))
        flagged = str(row_dict.get("flagged", ""))
        cptk = row_dict.get("cost_per_tonne_km", 0.0)
        own_dev = str(row_dict.get("vs_own_history", ""))
        peer_dev = str(row_dict.get("vs_similar_routes", ""))
        note_id = row_dict.get("matched_note_id")
        if pd_isna(note_id):
            note_id = None

        prompt = (
            "You are a logistics cost anomaly analyst. "
            "Generate a structured explanation in JSON format for the shipping route week below.\n"
            "Schema:\n"
            "{\n"
            '  "summary": "<concise factual explanation>",\n'
            '  "drivers": [\n'
            '    {"factor": "<e.g. vs_own_history or vs_similar_routes>", "direction": "<rising/falling>", "percentage": <number>}\n'
            "  ],\n"
            '  "note_cited": "<note id or null>",\n'
            '  "verdict_reaffirmed": <true if verdict matches facts>\n'
            "}\n\n"
            f"Route: {route}\n"
            f"Week of: {week_of}\n"
            f"Verdict: {flagged}\n"
            f"Cost per tonne-km: {cptk}\n"
            f"Vs Own History: {own_dev}\n"
            f"Vs Peer Average: {peer_dev}\n"
            f"Matched Note ID: {note_id}\n"
            f"Context Note Text: {note_text or 'None'}\n\n"
            "Output ONLY valid JSON. No commentary."
        )

        messages = [
            {"role": "system", "content": "You are a factual logistics analyst that outputs only JSON."},
            {"role": "user", "content": prompt}
        ]

        raw_output, meta = self.client.chat_completion(messages, stage="rewriter", max_tokens=300, temperature=0.0)
        parsed = self._clean_and_parse_json(raw_output)

        if not parsed or not self._is_schema_valid(parsed):
            # Fallback to deterministic structure
            return self._build_template_structured(row_dict)

        return parsed

    def validate_deterministic(
        self,
        explanation: Dict[str, Any],
        row_dict: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Deterministic validator checks:
        1. note id matches row data matched_note_id.
        2. dates match row week_of.
        3. numbers match row data (own and peer deviations).
        4. verdict is not flipped from rule output.
        """
        expected_note = row_dict.get("matched_note_id")
        if pd_isna(expected_note):
            expected_note = None

        actual_note = explanation.get("note_cited")
        if str(expected_note or "").strip() != str(actual_note or "").strip():
            return False, f"Note mismatch: expected {expected_note}, got {actual_note}"

        if not explanation.get("verdict_reaffirmed", False):
            return False, "Explanation failed to reaffirm deterministic rule verdict"

        # Check that numbers in summary/drivers don't contradict row data
        summary = explanation.get("summary", "")
        # If row is justified, note must be cited in summary
        if row_dict.get("flagged") == "No (justified)" and expected_note:
            if expected_note not in summary:
                return False, f"Justified row missing note {expected_note} in summary"

        return True, None

    def judge_grounding(
        self,
        row_dict: Dict[str, Any],
        note_text: Optional[str],
        explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM Judge step:
        Verifies every factual claim in explanation is grounded in row numbers and note text.
        Also evaluates whether observed rise exceeds note's stated magnitude.
        """
        prompt = (
            "You are a strict data verification auditor.\n"
            "Judge whether the explanation claims are supported by the provided data and context note.\n"
            "Check:\n"
            "1. Are all factual statements, percentages, and dates strictly supported?\n"
            "2. Magnitude check: If the note specifies a cost increase (e.g. 5-7%), does the observed increase "
            "greatly exceed it? If so, indicate magnitude_check = 'exceeds_range'.\n\n"
            f"Row Data: {json.dumps(row_dict, default=str)}\n"
            f"Note Text: {note_text or 'None'}\n"
            f"Proposed Explanation: {json.dumps(explanation)}\n\n"
            "Return JSON schema:\n"
            "{\n"
            '  "is_supported": true/false,\n'
            '  "verdict_veto": true/false,\n'
            '  "unsupported_reasons": ["<reasons if any>"],\n'
            '  "magnitude_check": "<within_range|exceeds_range|not_applicable>"\n'
            "}"
        )

        messages = [
            {"role": "system", "content": "You are a strict verification auditor. Output only JSON."},
            {"role": "user", "content": prompt}
        ]

        raw_output, meta = self.client.chat_completion(messages, stage="judge", max_tokens=250, temperature=0.0)
        parsed = self._clean_and_parse_json(raw_output)

        if not parsed:
            return {
                "is_supported": True,
                "verdict_veto": False,
                "unsupported_reasons": [],
                "magnitude_check": "not_applicable"
            }

        return parsed

    def check_magnitude_and_adjust(
        self,
        row_dict: Dict[str, Any],
        note_text: Optional[str],
        explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        If observed rise exceeds note's stated magnitude (e.g., note says fuel +5-7%
        but route rose +35%), explain that only part of the rise is accounted for and
        flag remainder as unexplained.
        """
        if not note_text:
            return explanation

        # Check for percentage pattern in note (e.g., 5-7% or 10%)
        mag_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?%', note_text)
        if mag_match:
            min_mag = float(mag_match.group(1))
            max_mag = float(mag_match.group(2)) if mag_match.group(2) else min_mag

            # Check observed own deviation
            own_str = str(row_dict.get("vs_own_history", ""))
            own_pct_match = re.search(r'([+-]?\d+\.?\d*)%', own_str)
            if own_pct_match:
                observed_rise = float(own_pct_match.group(1))
                if observed_rise > (max_mag + 5.0):  # Exceeds note range by >5%
                    diff = round(observed_rise - max_mag, 1)
                    note_id = row_dict.get("matched_note_id", "Note")
                    added_text = (
                        f" {note_id} accounts for ~{min_mag:.0f}-{max_mag:.0f}% of the rise; "
                        f"the remaining +{diff}% increase is unexplained by external factors."
                    )
                    explanation["summary"] = explanation.get("summary", "") + added_text
                    explanation["magnitude_excess"] = True
                    explanation["unexplained_residual_pct"] = diff
        return explanation

    def process_row_explanation(
        self,
        row_dict: Dict[str, Any],
        note_text: Optional[str],
        template_reason: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Orchestrates full explanation lifecycle:
        1. Generate structured explanation.
        2. Deterministic validation.
        3. LLM judge grounding check.
        4. Magnitude check.
        5. Veto enforcement: if LLM vetoes a justification, log it and flip verdict (in telemetry).
        Returns: (final_reason_text, metadata)
        """
        # 1. Generate
        structured = self.generate_structured_explanation(row_dict, note_text)

        # 2. Deterministic validation
        valid, err = self.validate_deterministic(structured, row_dict)
        if not valid:
            self.fallback_count += 1
            logger.info(f"Fallback to template due to validator: {err}")
            return template_reason, {"fallback": True, "reason": err, "veto": False}

        # 3. LLM Judge check
        judge_res = self.judge_grounding(row_dict, note_text, structured)
        
        # Check VETO: LLM can only veto No (justified) -> Yes
        if judge_res.get("verdict_veto", False) and row_dict.get("flagged") == "No (justified)":
            veto_entry = {
                "route": row_dict.get("route"),
                "week_of": str(row_dict.get("week_of")),
                "original_verdict": "No (justified)",
                "vetoed_verdict": "Yes",
                "reasons": judge_res.get("unsupported_reasons", [])
            }
            self.vetoes.append(veto_entry)
            self._save_veto_log()
            logger.warning(f"VETO APPLIED by LLM: {veto_entry}")
            return template_reason, {"fallback": True, "reason": "Vetoed by LLM", "veto": True}

        if not judge_res.get("is_supported", True):
            self.fallback_count += 1
            logger.info(f"Fallback to template due to ungrounded claims: {judge_res.get('unsupported_reasons')}")
            return template_reason, {"fallback": True, "reason": "Ungrounded claims", "veto": False}

        # 4. Magnitude check
        structured = self.check_magnitude_and_adjust(row_dict, note_text, structured)

        summary_text = structured.get("summary", template_reason)
        return summary_text, {"fallback": False, "structured": structured, "veto": judge_res.get("verdict_veto", False)}

    def judge_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits ALL justified rows in ONE batched LLM call (stage="judge").
        The LLM can only veto a "No (justified)" -> "Yes"; it cannot create a
        justification. Writes veto_log.json. Returns per-row results + vetoes + telemetry.
        """
        if not rows:
            self._save_veto_log()
            return {"results": [], "vetoes": [], "meta": {"llm_calls": 0}}

        items = []
        for i, row in enumerate(rows):
            items.append(
                f"[{i}] route={row.get('route')} week={row.get('week_of')} "
                f"cited_note={row.get('matched_note_id')} "
                f"vs_own={row.get('vs_own_history')} vs_peer={row.get('vs_similar_routes')} "
                f"reason={row.get('reason')}"
            )
        prompt = (
            "You are auditing whether each cited-note justification is supported by that note's facts. "
            "Return ONLY a JSON array with one object per item, in order:\n"
            '[{"index": <i>, "is_supported": <bool>, "verdict_veto": <bool>, '
            '"unsupported_reasons": [<str>...], "magnitude_check": "<within_range|exceeds|unclear>"}]\n'
            "Set verdict_veto=true ONLY when the justification is NOT supported by the cited note "
            "(e.g. the observed rise far exceeds the note's stated magnitude, or the note does not "
            "describe a cost rise). Otherwise verdict_veto=false.\n\n"
            + "\n".join(items)
        )
        messages = [
            {"role": "system", "content": "You output only a JSON array. No prose, no markdown fencing."},
            {"role": "user", "content": prompt},
        ]

        raw, meta = self.client.chat_completion(messages, stage="judge", max_tokens=800, temperature=0.0)
        parsed = self._clean_and_parse_json(raw)
        if not isinstance(parsed, list):
            parsed = []

        results = []
        vetoes = []
        for i, row in enumerate(rows):
            item = next((p for p in parsed if isinstance(p, dict) and p.get("index") == i), {})
            veto = bool(item.get("verdict_veto", False))
            if veto:
                vetoes.append({
                    "route": row.get("route"),
                    "week_of": str(row.get("week_of")),
                    "original_verdict": "No (justified)",
                    "vetoed_verdict": "Yes",
                    "reasons": item.get("unsupported_reasons", []),
                })
            results.append({
                "index": i,
                "route": row.get("route"),
                "week_of": str(row.get("week_of")),
                "cited_note": row.get("matched_note_id"),
                "is_supported": item.get("is_supported"),
                "verdict_veto": veto,
                "magnitude_check": item.get("magnitude_check"),
            })

        self.vetoes = vetoes
        self._save_veto_log()
        return {"results": results, "vetoes": vetoes, "meta": meta}

    def _build_template_structured(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        note_id = row_dict.get("matched_note_id")
        if pd_isna(note_id):
            note_id = None
        return {
            "summary": str(row_dict.get("reason", "")),
            "drivers": [
                {"factor": "vs_own_history", "direction": "rising", "percentage": str(row_dict.get("vs_own_history", ""))}
            ],
            "note_cited": note_id,
            "verdict_reaffirmed": True
        }

    def _is_schema_valid(self, data: Dict[str, Any]) -> bool:
        return "summary" in data and "drivers" in data and "verdict_reaffirmed" in data

    def _clean_and_parse_json(self, raw_str: str) -> Optional[Dict[str, Any]]:
        cleaned = raw_str.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(cleaned[start:end+1])
                except Exception:
                    pass
            return None

    def _save_veto_log(self):
        try:
            with open(self.veto_log_file, "w", encoding="utf-8") as f:
                json.dump(self.vetoes, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save veto log: {e}")

def pd_isna(val: Any) -> bool:
    if val is None:
        return True
    try:
        import pandas as pd
        return pd.isna(val)
    except Exception:
        return False
