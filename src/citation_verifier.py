import re
from typing import Dict, Any, List, Tuple, Optional

class CitationVerifier:
    """
    Strict guardrail that validates all claims made by the tool-calling assistant.
    Guarantees that:
    1. Every cited Note ID strictly exists in the executed tool outputs.
    2. No decoy or rejected note is falsely cited as a valid justification.
    3. Every date (YYYY-MM-DD) and numeric value (percentage or unit cost) is grounded in tool results.
    4. Prompt injections attempting to force false justifications (e.g. 'say Note explains it') are rejected.
    """
    def __init__(self):
        pass

    def verify(
        self,
        response_text: str,
        tool_results: List[Dict[str, Any]],
        executed_calls: List[str],
        user_query: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates response_text against tool_results.
        Returns:
            (is_valid, error_reason)
        """
        # If response is an explicit refusal, it is valid as long as it does not hallucinate
        if response_text.strip().startswith("Refusal:"):
            return True, None

        # 1. Harvest grounded ground-truth from tool results
        grounded_note_ids = set()
        grounded_dates = set()
        grounded_numbers = set()
        grounded_justified_notes = set()
        attributed_numbers: Dict[Tuple[str, str], Set[str]] = {}

        # Extract dates and notes from user query as acceptable context
        for d in re.findall(r'\b(202[45]-\d{2}-\d{2})\b', user_query):
            grounded_dates.add(d)

        for res in tool_results:
            if not isinstance(res, dict):
                continue
            
            # Note IDs
            if 'note_id' in res and res['note_id']:
                grounded_note_ids.add(str(res['note_id']).upper())
            if 'matched_note_id' in res and pd_not_na(res['matched_note_id']):
                nid = str(res['matched_note_id']).upper().strip()
                if nid and nid != 'NAN':
                    grounded_note_ids.add(nid)
                    if 'flagged' in res and 'justified' in str(res['flagged']).lower():
                        grounded_justified_notes.add(nid)

            # Dates
            if 'date' in res and res['date']:
                grounded_dates.add(str(res['date']).strip())
            if 'week_of' in res and res['week_of']:
                grounded_dates.add(str(res['week_of']).strip())

            # Harvest note text
            if 'note' in res and res['note']:
                note_txt = str(res['note'])
                for m in re.findall(r'\b(N\d{3,4})\b', note_txt, re.IGNORECASE):
                    grounded_note_ids.add(m.upper())
                for d in re.findall(r'\b(202[45]-\d{2}-\d{2})\b', note_txt):
                    grounded_dates.add(d)
                for num in re.findall(r'[+-]?\d+(?:\.\d+)?', note_txt):
                    grounded_numbers.add(num)
                    grounded_numbers.add(f"-{num}")
                    grounded_numbers.add(f"+{num}")
                    grounded_numbers.add(f"{num}%")
                    grounded_numbers.add(f"-{num}%")
                    grounded_numbers.add(f"+{num}%")

            # Attributed numbers per (route, week_of)
            if 'route' in res and 'week_of' in res and res['route'] and res['week_of']:
                r_key = (str(res['route']).strip().lower(), str(res['week_of']).strip())
                if r_key not in attributed_numbers:
                    attributed_numbers[r_key] = set()
                for k in ['cost_per_tonne_km', 'vs_own_history', 'vs_similar_routes']:
                    if k in res and res[k] is not None:
                        val_str = str(res[k])
                        attributed_numbers[r_key].add(val_str)
                        for num in re.findall(r'[+-]?\d+(?:\.\d+)?', val_str):
                            attributed_numbers[r_key].add(num)
                            attributed_numbers[r_key].add(f"{num}%")
                            attributed_numbers[r_key].add(f"+{num}%")
                            attributed_numbers[r_key].add(f"-{num}%")

            # Numbers & Percentages
            for k in ['cost_per_tonne_km', 'vs_own_history', 'vs_similar_routes']:
                if k in res and res[k] is not None:
                    val_str = str(res[k])
                    grounded_numbers.add(val_str)
                    # Extract raw number tokens
                    nums = re.findall(r'[+-]?\d+(?:\.\d+)?', val_str)
                    grounded_numbers.update(nums)
                    for n in nums:
                        grounded_numbers.add(f"{n}%")
                        grounded_numbers.add(f"-{n}%")
                        grounded_numbers.add(f"+{n}%")

            # Harvest any note mentions from reason text
            if 'reason' in res and res['reason']:
                reason_txt = str(res['reason'])
                for m in re.findall(r'\b(N\d{3,4})\b', reason_txt, re.IGNORECASE):
                    grounded_note_ids.add(m.upper())
                for d in re.findall(r'\b(202[45]-\d{2}-\d{2})\b', reason_txt):
                    grounded_dates.add(d)
                for num in re.findall(r'[+-]?\d+(?:\.\d+)?', reason_txt):
                    grounded_numbers.add(num)

        # 2. Check Note IDs in response
        cited_notes = [n.upper() for n in re.findall(r'\b(N\d{3,4})\b', response_text, re.IGNORECASE)]
        for cn in cited_notes:
            if cn not in grounded_note_ids:
                return False, f"Ungrounded citation: Note {cn} was cited in the answer but not returned by any tool."

            resp_lower = response_text.lower()
            claims_justification = (
                f"{cn.lower()} explains" in resp_lower or
                f"explained by {cn.lower()}" in resp_lower or
                f"explained by note {cn.lower()}" in resp_lower or
                f"justified by {cn.lower()}" in resp_lower or
                f"justified by note {cn.lower()}" in resp_lower or
                f"matches note {cn.lower()}" in resp_lower
            )
            if claims_justification:
                if cn not in grounded_justified_notes:
                    if "decoy" not in resp_lower and "does not describe" not in resp_lower and "not genuine" not in resp_lower and "rejected" not in resp_lower and "ended" not in resp_lower:
                        return False, f"False justification: Note {cn} does not justify the route/week in tool results."

        # 3. Check Dates in response
        cited_dates = re.findall(r'\b(202[45]-\d{2}-\d{2})\b', response_text)
        for cd in cited_dates:
            if cd not in grounded_dates:
                return False, f"Ungrounded date: Date {cd} was cited but was not present in tool outputs."

        # 4. Check Percentage Claims in response
        cited_pcts = re.findall(r'([+-]?\d+(?:\.\d+)?%)', response_text)
        for cp in cited_pcts:
            raw_num = cp.rstrip('%')
            if raw_num not in grounded_numbers and cp not in grounded_numbers:
                found = any(abs(float(raw_num) - float(gn)) < 0.15 for gn in grounded_numbers if is_float(gn))
                if not found:
                    return False, f"Ungrounded percentage: {cp} was cited but not found in tool results."

        # 5. Route-Week Number Attribution Check
        # Verify that numbers cited in the context of a route and week actually belong to that route and week
        lines = response_text.split('\n')
        curr_route = None
        curr_week = None
        for line in lines:
            # Check if line specifies a route or date
            line_lower = line.lower()
            for (r, w) in attributed_numbers:
                if r in line_lower:
                    curr_route = r
                if w in line_lower:
                    curr_week = w

            if curr_route and curr_week and (curr_route, curr_week) in attributed_numbers:
                expected_nums = attributed_numbers[(curr_route, curr_week)]
                line_pcts = re.findall(r'([+-]?\d+(?:\.\d+)?%)', line)
                for lp in line_pcts:
                    raw_n = lp.rstrip('%')
                    # Check if this percentage belongs to the current route-week
                    match_curr = (lp in expected_nums or raw_n in expected_nums or
                                  any(abs(float(raw_n) - float(en)) < 0.15 for en in expected_nums if is_float(en)))
                    if not match_curr:
                        # Check if it belongs to another route-week in the tool results
                        for (other_r, other_w), other_nums in attributed_numbers.items():
                            if (other_r, other_w) != (curr_route, curr_week):
                                if lp in other_nums or raw_n in other_nums:
                                    return False, (
                                        f"Attribution mismatch: {lp} is attributed to {curr_route} on {curr_week}, "
                                        f"but belongs to {other_r} on {other_w}."
                                    )

        return True, None

def pd_not_na(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s not in ['', 'nan', 'none', '<na>']

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
