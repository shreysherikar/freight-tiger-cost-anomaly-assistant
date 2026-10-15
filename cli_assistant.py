import argparse
import os
import re
import sys
import json
import time
from datetime import datetime, timezone
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from src.agent_tools import AssistantTools
from src.citation_verifier import CitationVerifier
from src.llm_client import _load_env_if_present

# NOTE: .env is NOT auto-loaded at import. main() loads it only when --live is passed (B.1).
logger = logging.getLogger(__name__)

MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
    'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5,
    'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10,
    'november': 11, 'nov': 11, 'december': 12, 'dec': 12
}

class ToolCallingAgent:
    """
    Autonomous multi-step tool-calling assistant for Freight Tiger.
    Equipped with 4 domain tools, strict citation verification, and
    temperature-0 planning with automated fallback to the deterministic router.
    """
    def __init__(
        self,
        tools: AssistantTools,
        use_llm: bool = True,
        model_name: str = "llama-3.1-8b-instant"
    ):
        self.tools = tools
        self.use_llm = use_llm
        self.model_name = model_name
        self.verifier = CitationVerifier()
        
        # Telemetry & Execution Traces
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_llm_calls = 0
        self.total_cost_usd = 0.0
        self.execution_traces: List[Dict[str, Any]] = []

        # API keys
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def execute_tool_call(self, call_str: str) -> Tuple[Optional[Any], str]:
        """
        Parses and executes a tool call string against AssistantTools.
        Logs to execution trace: tool, args, timestamp, duration_ms, returned_keys.
        """
        call_str = call_str.strip()
        t0 = time.time()
        tool_name = "unknown"
        args_dict = {}
        res = None
        norm_call = call_str
        
        # 1. get_flagged_weeks
        if "get_flagged_weeks" in call_str:
            tool_name = "get_flagged_weeks"
            m_flag = re.search(r"get_flagged_weeks\((?:route=)?['\"]?([^'\")]*)['\"]?\)", call_str)
            route_arg = m_flag.group(1).strip() if (m_flag and m_flag.group(1) and m_flag.group(1).strip() not in ['', 'none', 'null']) else None
            args_dict = {"route": route_arg}
            res = self.tools.get_flagged_weeks(route=route_arg)
            norm_call = f"get_flagged_weeks(route='{route_arg}')" if route_arg else "get_flagged_weeks()"

        # 2. get_weekly_metrics
        elif "get_weekly_metrics" in call_str:
            tool_name = "get_weekly_metrics"
            m_metrics = re.search(r"get_weekly_metrics\((?:route=)?['\"]?([^'\",]+)['\"]?,\s*(?:week_of=)?['\"]?([^'\")]+)['\"]?\)", call_str)
            if m_metrics:
                r, w = m_metrics.group(1).strip(), m_metrics.group(2).strip()
                args_dict = {"route": r, "week_of": w}
                res = self.tools.get_weekly_metrics(r, w)
                norm_call = f"get_weekly_metrics(route='{r}', week_of='{w}')"

        # 3. get_note
        elif "get_note" in call_str:
            tool_name = "get_note"
            m_note = re.search(r"get_note\((?:note_id=)?['\"]?([A-Za-z0-9]+)['\"]?\)", call_str)
            if m_note:
                nid = m_note.group(1).upper().strip()
                args_dict = {"note_id": nid}
                res = self.tools.get_note(nid)
                norm_call = f"get_note(note_id='{nid}')"

        # 4. compare_to_peers
        elif "compare_to_peers" in call_str:
            tool_name = "compare_to_peers"
            m_peer = re.search(r"compare_to_peers\((?:route=)?['\"]?([^'\",]+)['\"]?,\s*(?:week_of=)?['\"]?([^'\")]+)['\"]?\)", call_str)
            if m_peer:
                r, w = m_peer.group(1).strip(), m_peer.group(2).strip()
                args_dict = {"route": r, "week_of": w}
                res = self.tools.compare_to_peers(r, w)
                norm_call = f"compare_to_peers(route='{r}', week_of='{w}')"

        duration_ms = round((time.time() - t0) * 1000, 2)
        
        # Extract returned keys for trace
        returned_keys = []
        if isinstance(res, dict):
            returned_keys = list(res.keys())
        elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
            returned_keys = list(res[0].keys())

        # Log trace
        self.execution_traces.append({
            "tool": tool_name,
            "args": args_dict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "returned_keys": returned_keys
        })

        return res, norm_call

    def run(self, query: str) -> Tuple[str, List[str]]:
        """
        Executes query through multi-step tool-calling with citation verification.
        Returns:
            (grounded_answer, executed_tool_calls)
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()
        tool_calls = []
        tool_results = []

        # Guardrail 1: Out-of-domain refusal
        out_of_domain_words = ['cookie', 'capital', 'president', 'weather in paris', 'movie', 'recipe', 'football']
        if any(w in query_lower for w in out_of_domain_words):
            return "Refusal: I can only answer questions related to shipping costs and route anomalies.", tool_calls

        # Guardrail 2: Unsupported route names
        unsupported_cities = ['hyderabad', 'bangalore-chennai', 'kanpur', 'patna', 'nagpur', 'bhopal']
        if any(city in query_lower for city in unsupported_cities):
            return "Refusal: The requested route is not tracked in this dataset.", tool_calls

        matched_route = None
        for r in self.tools.valid_routes:
            parts = r.lower().split('-')
            # Check exact name or directional presence (origin before destination)
            if r.lower() in query_lower:
                matched_route = r
                break
            elif parts[0] in query_lower and parts[1] in query_lower:
                idx0 = query_lower.find(parts[0])
                idx1 = query_lower.find(parts[1])
                if idx0 < idx1:
                    matched_route = r
                    break

        # Guardrail 3: Out-of-bounds years
        years = re.findall(r'\b(19\d\d|20\d\d)\b', query_clean)
        for y in years:
            yi = int(y)
            if yi < 2024 or yi > 2025:
                return f"Refusal: Data is only available for 2024 and 2025 (year {yi} is out of bounds).", tool_calls

        # Multi-Step Planning (LLM or Deterministic Router fallback)
        can_use_live_llm = self.use_llm and (bool(self.groq_key) or bool(self.openai_key))

        if can_use_live_llm:
            answer, tool_calls, tool_results = self._run_llm_planner(query_clean)
        else:
            answer, tool_calls, tool_results = self._run_deterministic_router(query_clean)

        # Citation Verification Guardrail
        is_valid, error_msg = self.verifier.verify(answer, tool_results, tool_calls, user_query=query_clean)
        if not is_valid:
            logger.warning(f"Citation verification failed: {error_msg}")
            # Enforce rejection / refusal on hallucination or injection
            return "Refusal: The generated response contains unverified claims or ungrounded note citations not supported by pipeline data.", tool_calls

        return answer, tool_calls

    def _run_deterministic_router(self, query: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        High-precision multi-step deterministic router.
        Executes domain tools and synthesizes grounded answers.
        """
        query_lower = query.lower()
        tool_calls = []
        tool_results = []

        # Find route
        matched_route = None
        for r in self.tools.valid_routes:
            parts = r.lower().split('-')
            if r.lower() in query_lower or (parts[0] in query_lower and parts[1] in query_lower):
                matched_route = r
                break

        note_match = re.search(r'\b(N00[1-9]|N010)\b', query, re.IGNORECASE)
        date_match = re.search(r'\b(2024-\d{2}-\d{2}|2025-\d{2}-\d{2})\b', query)

        # Multi-step 1: Route + Date (with peer comparison check)
        if date_match and matched_route:
            exact_date = date_match.group(1)
            
            if 'peer' in query_lower or 'similar' in query_lower:
                call_str = f"compare_to_peers(route='{matched_route}', week_of='{exact_date}')"
                peer_res, norm_call = self.execute_tool_call(call_str)
                tool_calls.append(norm_call)
                if peer_res:
                    tool_results.append(peer_res)
                    ans = (
                        f"Peer comparison for {matched_route} on week {exact_date}:\n"
                        f"- Cost: INR {peer_res['cost_per_tonne_km']:.2f}/tonne-km\n"
                        f"- Peer Deviation: {peer_res['vs_similar_routes']}\n"
                        f"- Flagged: {peer_res['flagged']}"
                    )
                    return ans, tool_calls, tool_results
                return f"No record found for route {matched_route} on week {exact_date}.", tool_calls, tool_results

            call_str = f"get_weekly_metrics(route='{matched_route}', week_of='{exact_date}')"
            metrics, norm_call = self.execute_tool_call(call_str)
            tool_calls.append(norm_call)
            if metrics:
                tool_results.append(metrics)
                nid = metrics.get('matched_note_id')
                if pd.notna(nid) and nid:
                    note_call = f"get_note(note_id='{nid}')"
                    n_info, norm_n_call = self.execute_tool_call(note_call)
                    tool_calls.append(norm_n_call)
                    if n_info:
                        tool_results.append(n_info)

                ans = (
                    f"Metrics for {matched_route} on week {exact_date}:\n"
                    f"- Cost: INR {metrics['cost_per_tonne_km']:.2f}/tonne-km\n"
                    f"- vs Own History: {metrics['vs_own_history']}\n"
                    f"- vs Similar Routes: {metrics['vs_similar_routes']}\n"
                    f"- Flagged: {metrics['flagged']}\n"
                    f"- Note Cited: {metrics['matched_note_id'] or 'None'}\n"
                    f"- Reason: {metrics['reason']}"
                )
                return ans, tool_calls, tool_results
            return f"No record found for route {matched_route} on week {exact_date}.", tool_calls, tool_results

        # Multi-step 2: Note ID Query
        if note_match and not matched_route:
            nid = note_match.group(1).upper()
            note_info, norm_n = self.execute_tool_call(f"get_note(note_id='{nid}')")
            tool_calls.append(norm_n)
            if note_info:
                tool_results.append(note_info)
                if 'route' in query_lower or 'affected' in query_lower:
                    all_flags, norm_flags = self.execute_tool_call("get_flagged_weeks()")
                    tool_calls.append(norm_flags)
                    matching = [f for f in (all_flags or []) if f.get('matched_note_id') == nid]
                    tool_results.extend(matching)
                    routes_affected = sorted(list(set(m['route'] for m in matching)))
                    ans = (
                        f"Note {nid} (Date: {note_info['date']}, Applies to: {note_info['applies_to']}):\n"
                        f"Text: \"{note_info['note']}\"\n"
                        f"Routes flagged and justified by {nid}: {', '.join(routes_affected) or 'None in output.csv'}."
                    )
                    return ans, tool_calls, tool_results

                ans = (
                    f"Note {nid} Details:\n"
                    f"- Date: {note_info['date']}\n"
                    f"- Applies To: {note_info['applies_to']}\n"
                    f"- Content: \"{note_info['note']}\""
                )
                return ans, tool_calls, tool_results
            return f"Note {nid} was not found.", tool_calls, tool_results

        # Multi-step 3: Route over Month / Multi-Week
        if matched_route:
            weeks, norm_w = self.execute_tool_call(f"get_flagged_weeks(route='{matched_route}')")
            tool_calls.append(norm_w)
            tool_results.extend(weeks or [])

            target_month = None
            for m_name, m_num in MONTH_MAP.items():
                if re.search(r'\b' + m_name + r'\b', query_lower):
                    target_month = m_num
                    break

            target_year = None
            if '2024' in query:
                target_year = 2024
            elif '2025' in query:
                target_year = 2025

            filtered = []
            for w in (weeks or []):
                dt = pd.to_datetime(w['week_of'])
                if target_year and dt.year != target_year:
                    continue
                if target_month and dt.month != target_month:
                    continue
                filtered.append(w)

            if not filtered:
                return f"No flagged cost increases found for {matched_route} in the requested period.", tool_calls, tool_results

            # Check if any notes need to be fetched via get_note
            for f in filtered:
                nid = f.get('matched_note_id')
                if pd.notna(nid) and nid:
                    call_n = f"get_note(note_id='{nid}')"
                    if call_n not in tool_calls:
                        n_res, norm_n = self.execute_tool_call(call_n)
                        tool_calls.append(norm_n)
                        if n_res:
                            tool_results.append(n_res)

            lines = [f"Analysis for {matched_route}:"]
            for f in filtered:
                nid = f['matched_note_id']
                status = "JUSTIFIED" if "justified" in str(f['flagged']).lower() else "UNJUSTIFIED (FLAGGED YES)"
                note_str = f" [Cited Note: {nid}]" if pd.notna(nid) and nid else ""
                lines.append(f"- Week of {f['week_of']} ({status}{note_str}): {f['reason']}")

            return "\n".join(lines), tool_calls, tool_results

        return "Refusal: Please specify a valid tracked route (e.g. Chennai-Bangalore) or note ID (e.g. N001).", tool_calls, tool_results

    def _run_llm_planner(self, query: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """Runs the multi-step planning loop using temperature 0 LLM."""
        tool_calls = []
        tool_results = []
        
        system_prompt = (
            "You are Freight Tiger's AI Shipping Cost Assistant. "
            "You have 4 domain tools to inspect shipping records and context notes:\n"
            "1. get_flagged_weeks(route=None)\n"
            "2. get_weekly_metrics(route, week_of)\n"
            "3. get_note(note_id)\n"
            "4. compare_to_peers(route, week_of)\n\n"
            "To execute a tool call, output exactly: TOOL_CALL: tool_name(args...)\n"
            "When ready to answer, output: FINAL_ANSWER: your answer.\n"
            "Rules:\n"
            "- Only cite facts, note IDs, dates, and numbers that were returned by your tool calls.\n"
            "- Never guess or claim a note justifies an anomaly unless the tool output confirms it.\n"
            "- If the query is out of domain or unsupported, output: FINAL_ANSWER: Refusal: [reason]"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Multi-step loop (max 4 steps)
        for _ in range(4):
            try:
                raw_response = self._call_llm(messages)
            except Exception as e:
                logger.warning(f"LLM planning failed ({e}); falling back to deterministic router.")
                return self._run_deterministic_router(query)

            if "FINAL_ANSWER:" in raw_response:
                answer = raw_response.split("FINAL_ANSWER:", 1)[1].strip()
                return answer, tool_calls, tool_results

            if "TOOL_CALL:" in raw_response:
                call_text = raw_response.split("TOOL_CALL:", 1)[1].strip().split("\n")[0]
                res, norm_call = self.execute_tool_call(call_text)
                tool_calls.append(norm_call)
                if isinstance(res, list):
                    tool_results.extend(res)
                elif isinstance(res, dict):
                    tool_results.append(res)

                # Feed result back to model
                messages.append({"role": "assistant", "content": f"TOOL_CALL: {norm_call}"})
                messages.append({"role": "user", "content": f"TOOL_RESULT: {json.dumps(res, default=str)}"})
            else:
                # If neither format is emitted, treat response as final answer
                return raw_response.strip(), tool_calls, tool_results

        # Fallback if loop ends without final answer
        return self._run_deterministic_router(query)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Invokes LLM API at temperature 0."""
        self.total_llm_calls += 1
        prompt_len = sum(len(m["content"].split()) for m in messages) * 2
        self.total_input_tokens += prompt_len

        if self.groq_key:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            model_to_use = self.model_name  # pinned; no silent fallback, no models.list() probe (B.2)
            resp = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.0
            )
            out_text = resp.choices[0].message.content.strip()
        elif self.openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            out_text = resp.choices[0].message.content.strip()
        else:
            raise RuntimeError("No LLM API key present.")

        out_len = len(out_text.split()) * 2
        self.total_output_tokens += out_len
        self.total_cost_usd += (prompt_len / 1_000_000) * 0.05 + (out_len / 1_000_000) * 0.08
        return out_text

def main():
    parser = argparse.ArgumentParser(description="Freight Tiger Autonomous Tool-Calling Assistant")
    parser.add_argument("--question", type=str, help="Natural language question to ask")
    parser.add_argument("--output-csv", type=str, default="output.csv", help="Path to pipeline output CSV")
    parser.add_argument("--notes-csv", type=str, default="context_notes.csv", help="Path to context notes CSV")
    parser.add_argument("--use-llm", action="store_true", default=True, help="Enable LLM planner (default: True, with rule fallback)")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Enforce deterministic rule-based router only")
    parser.add_argument("--live", action="store_true", default=False, help="Load .env and allow LIVE LLM planning calls. Without this the agent runs fully offline (deterministic router); the API key is never read.")
    args = parser.parse_args()

    if args.live:
        _load_env_if_present()

    tools = AssistantTools(output_csv_path=args.output_csv, notes_csv_path=args.notes_csv)
    agent = ToolCallingAgent(tools, use_llm=args.use_llm)

    if args.question:
        ans, calls = agent.run(args.question)
        print(f"Executed Tool Calls: {calls}")
        print("\n" + ans)
    else:
        print("="*65)
        print("FREIGHT TIGER AUTONOMOUS TOOL-CALLING ASSISTANT")
        print("Type 'exit' to quit.")
        print("="*65)
        while True:
            try:
                q = input("\nYour question: ").strip()
                if not q or q.lower() in ['exit', 'quit', 'q']:
                    break
                ans, calls = agent.run(q)
                print(f"[Tools Used: {calls}]")
                print(ans)
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
