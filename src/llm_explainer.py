import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional

from src.llm_client import get_llm_client, BaseLLMClient, _load_env_if_present

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens for common free/low-cost models
PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mock-llm": {"input": 0.0, "output": 0.0}
}

class LLMExplainer:
    """
    Optional LLM rewriter (temperature 0) that strictly rewrites explanation wording.
    Never decides or modifies the verdict, cited note ID, or numeric calculations.
    Includes strict validation and prompt-hash caching.
    """
    def __init__(
        self,
        use_llm: bool = False,
        model_name: str = "llama-3.1-8b-instant",
        cache_file: str = "llm_cache.json",
        cost_log_file: str = "cost_log.json",
        llm_client: Optional[BaseLLMClient] = None,
        live: bool = False
    ):
        self.use_llm = use_llm
        self.model_name = model_name
        self.cache_file = cache_file
        self.cost_log_file = cost_log_file
        self.live = live
        self.mode = "live" if live else "offline"
        self.client = llm_client if llm_client is not None else (get_llm_client(live=live) if use_llm else None)
        self.model_name = self.client.model_name if self.client else "template"
        self.cache = self.client.cache if self.client else {}
        
        # Usage tracking
        self.call_count = 0
        self.fallback_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load LLM cache: {e}")
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save LLM cache: {e}")

    def save_cost_log(self):
        """Writes usage metrics to a per-run file cost_logs/<timestamp>_<mode>.json (never overwrites cost_log.json)."""
        import os
        import datetime
        summary = {
            "use_llm": self.use_llm,
            "mode": self.mode,
            "model_name": self.model_name,
            "llm_calls": self.call_count,
            "validator_fallbacks": self.fallback_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        os.makedirs("cost_logs", exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        run_path = os.path.join("cost_logs", f"{ts}_{self.mode}.json")
        try:
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            self._last_cost_log_path = run_path
        except Exception as e:
            logger.warning(f"Could not save cost log: {e}")

    def validate_explanation(
        self,
        text: str,
        verdict: str,
        matched_note_id: str,
        note_date: Optional[str]
    ) -> bool:
        """
        Enforces strict guardrails:
        1. If verdict is 'No (justified)', matched_note_id and note_date must appear.
        2. Verdict direction must never be contradicted.
        """
        text_lower = text.lower()
        if verdict == "No (justified)":
            if matched_note_id not in text:
                logger.warning(f"Validation failed: {matched_note_id} missing in LLM response.")
                return False
            if note_date and note_date not in text:
                logger.warning(f"Validation failed: Date {note_date} missing in LLM response.")
                return False
            if "unexplained" in text_lower or "unjustified" in text_lower:
                logger.warning("Validation failed: Justified case labeled as unexplained.")
                return False
        elif verdict == "Yes":
            if "has a clear explanation" in text_lower or "justified by note" in text_lower:
                logger.warning("Validation failed: Unjustified case claimed to have valid justification.")
                return False
        return True

    def rewrite_explanation(
        self,
        route: str,
        week_of: str,
        verdict: str,
        matched_note_id: str,
        template_reason: str,
        note_date: Optional[str] = None,
        note_text: Optional[str] = None
    ) -> str:
        """
        Rewrites explanation wording if use_llm is enabled; otherwise returns template.
        """
        if not self.use_llm:
            return template_reason

        prompt = (
            f"You are a logistics assistant explaining shipping cost anomaly flags. "
            f"Do not alter the facts, the verdict, or the cited note ID.\n"
            f"Route: {route}\n"
            f"Week of: {week_of}\n"
            f"Verdict: {verdict}\n"
            f"Cited Note ID: {matched_note_id or 'None'}\n"
            f"Note Date: {note_date or 'None'}\n"
            f"Note Details: {note_text or 'None'}\n"
            f"Standard Baseline Text: {template_reason}\n\n"
            f"Task: Rewrite the baseline text in 1-2 clear, professional sentences. "
            f"You MUST cite the note ID '{matched_note_id}' and date '{note_date}' if justified."
        )

        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()

        # Check Cache
        if prompt_hash in self.cache:
            cached_text = self.cache[prompt_hash]
            if self.validate_explanation(cached_text, verdict, matched_note_id, note_date):
                return cached_text

        # Call LLM via central client with telemetry and stage caps
        try:
            messages = [{"role": "user", "content": prompt}]
            generated_text, meta = self.client.chat_completion(
                messages,
                stage="rewriter",
                max_tokens=250,
                temperature=0.0
            )
            self.call_count += 1
            self.input_tokens += meta.get("input_tokens", 0)
            self.output_tokens += meta.get("output_tokens", 0)
            self.estimated_cost_usd += meta.get("cost_usd", 0.0)
        except Exception as e:
            logger.warning(f"LLM call failed ({e}). Falling back to template reason.")
            self.fallback_count += 1
            return template_reason

        # Validate Guardrails
        if generated_text and self.validate_explanation(generated_text, verdict, matched_note_id, note_date):
            self.cache[prompt_hash] = generated_text
            self._save_cache()
            return generated_text
        else:
            logger.warning("LLM response failed validation guardrail. Falling back to template.")
            self.fallback_count += 1
            return template_reason
