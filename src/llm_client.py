import os
import json
import time
import hashlib
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _load_env_if_present():
    """Safely loads variables from local .env if present without overwriting existing env."""
    for env_path in [".env", os.path.join(os.path.dirname(__file__), "..", ".env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'").strip('"')
                            if k and not os.environ.get(k):
                                os.environ[k] = v
            except Exception:
                pass

# NOTE: .env is NOT auto-loaded at import time. Entry points call _load_env_if_present()
# only when an explicit --live flag is passed. Default = no key, mock/cache only (B.1).

# Token pricing per 1M tokens (Groq llama-3.1-8b-instant: free tier, commercial equivalent pricing for tracking)
PRICING_PER_1M = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mock": {"input": 0.0, "output": 0.0}
}

class BaseLLMClient:
    """
    Base client enforcing cost controls, stage caps, retries, and prompt-hash caching.
    """
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        max_total_calls: int = 120,
        stage_caps: Optional[Dict[str, int]] = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 1,  # B.3: at most 1 retry, and only on HTTP 429
        cache_file: str = "llm_cache.json",
        cost_log_file: str = "cost_log.json",
        dry_run: bool = False,
        refresh_cache: bool = False
    ):
        self.model_name = model_name
        self.max_total_calls = max_total_calls
        self.stage_caps = stage_caps or {
            "extraction": 15,
            "judge": 60,
            "agent_eval": 60,
            "rewriter": 60,
            "general": 60
        }
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_file = cache_file
        self.cost_log_file = cost_log_file
        self.dry_run = dry_run
        self.refresh_cache = refresh_cache
        self.mode = "offline"
        self.ledger_file = "llm_ledger.jsonl"
        self.stage_consec_failures: Dict[str, int] = {}
        self._counts_attempts = False
        self._last_cost_log_path = None

        # Cache
        self.cache: Dict[str, str] = self._load_cache()

        # Telemetry tracking per stage
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.stage_metrics: Dict[str, Dict[str, Any]] = {}
        for stage in self.stage_caps:
            self.stage_metrics[stage] = {
                "calls": 0,
                "cache_hits": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0
            }

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache {self.cache_file}: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    def save_cost_log(self):
        """Writes telemetry to a per-run file cost_logs/<timestamp>_<mode>.json (never overwrites cost_log.json)."""
        summary = {
            "model_name": self.model_name,
            "mode": self.mode,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_estimated_cost_usd": round(self.total_cost_usd, 6),
            "stages": self.stage_metrics,
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

    def _log_attempt(self, stage, attempt, http_status, success, in_tokens, out_tokens, cost, error):
        """Append one line per HTTP attempt to llm_ledger.jsonl (B.3)."""
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": self.mode,
            "stage": stage,
            "model": self.model_name,
            "attempt": attempt,
            "http_status": http_status,
            "success": success,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(cost, 8),
            "error": error,
        }
        try:
            with open(self.ledger_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Could not write ledger: {e}")

    def compute_prompt_hash(self, messages: List[Dict[str, str]]) -> str:
        serialized = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()) * 2)

    def dry_run_estimate(self, messages: List[Dict[str, str]], stage: str = "general") -> Dict[str, Any]:
        """Calculates tokens and cost without making any network calls."""
        full_prompt = " ".join(m.get("content", "") for m in messages)
        in_tokens = self.estimate_tokens(full_prompt)
        est_out_tokens = 80
        rates = PRICING_PER_1M.get("llama-3.1-8b-instant", {"input": 0.05, "output": 0.08})
        cost = (in_tokens / 1_000_000) * rates["input"] + (est_out_tokens / 1_000_000) * rates["output"]
        return {
            "stage": stage,
            "input_tokens": in_tokens,
            "estimated_output_tokens": est_out_tokens,
            "estimated_cost_usd": round(cost, 6)
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stage: str = "general",
        max_tokens: int = 500,
        temperature: float = 0.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes a chat completion with strict cap checks and prompt caching.
        """
        prompt_hash = self.compute_prompt_hash(messages)
        full_prompt = " ".join(m.get("content", "") for m in messages)
        in_tokens = self.estimate_tokens(full_prompt)

        # Check Cache
        if not self.refresh_cache and prompt_hash in self.cache:
            cached_text = self.cache[prompt_hash]
            out_tokens = self.estimate_tokens(cached_text)
            if stage in self.stage_metrics:
                self.stage_metrics[stage]["cache_hits"] += 1
            return cached_text, {
                "cache_hit": True,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cost_usd": 0.0
            }

        # Dry run check
        if self.dry_run:
            est = self.dry_run_estimate(messages, stage=stage)
            return "DRY_RUN_RESPONSE", {
                "cache_hit": False,
                "dry_run": True,
                "input_tokens": est["input_tokens"],
                "output_tokens": est["estimated_output_tokens"],
                "cost_usd": est["estimated_cost_usd"]
            }

        # Enforce Stage & Total Call Caps
        stage_cap = self.stage_caps.get(stage, 60)
        curr_stage_calls = self.stage_metrics.get(stage, {}).get("calls", 0)
        if curr_stage_calls >= stage_cap:
            raise RuntimeError(
                f"Aborting: Exceeded maximum allowed calls ({stage_cap}) for stage '{stage}'."
            )
        if self.total_calls >= self.max_total_calls:
            raise RuntimeError(
                f"Aborting: Exceeded maximum total allowed calls ({self.max_total_calls})."
            )

        # Execute call with retries and exponential backoff
        content = self._execute_call(messages, max_tokens=max_tokens, temperature=temperature, stage=stage)
        out_tokens = self.estimate_tokens(content)

        rates = PRICING_PER_1M.get(self.model_name, PRICING_PER_1M["llama-3.1-8b-instant"])
        call_cost = (in_tokens / 1_000_000) * rates["input"] + (out_tokens / 1_000_000) * rates["output"]

        # Update metrics. For the live Groq client every HTTP attempt is already
        # counted inside _execute_call (B.3); only count a logical call here for
        # clients that do not self-count (the mock).
        if not self._counts_attempts:
            self.total_calls += 1
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            self.total_cost_usd += call_cost
            if stage in self.stage_metrics:
                self.stage_metrics[stage]["calls"] += 1
                self.stage_metrics[stage]["input_tokens"] += in_tokens
                self.stage_metrics[stage]["output_tokens"] += out_tokens
                self.stage_metrics[stage]["estimated_cost_usd"] += call_cost

        # Update and save cache (per-run cost summary is written explicitly at end of run)
        self.cache[prompt_hash] = content
        self._save_cache()

        return content, {
            "cache_hit": False,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(call_cost, 6)
        }

    def _execute_call(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class GroqLLMClient(BaseLLMClient):
    """Client for Groq API using free tier llama-3.1-8b-instant or available open model."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self._counts_attempts = True
        self.mode = "live"
        env_model = os.getenv("GROQ_MODEL")
        if env_model:
            self.model_name = env_model
        # Model is pinned (default llama-3.1-8b-instant). No silent fallback and no
        # models.list() capability probe: if the model is unavailable the API call
        # fails and the client stops with an error (B.2).

    def _execute_call(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, stage: str = "general") -> str:
        from groq import Groq
        client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
        full_prompt = " ".join(m.get("content", "") for m in messages)
        in_tokens = self.estimate_tokens(full_prompt)
        rates = PRICING_PER_1M.get(self.model_name, PRICING_PER_1M["llama-3.1-8b-instant"])

        last_error = None
        for attempt in range(self.max_retries + 1):  # max_retries == 1 -> at most 2 attempts
            # Count every HTTP attempt toward the caps (B.3)
            self.total_calls += 1
            if stage in self.stage_metrics:
                self.stage_metrics[stage]["calls"] += 1
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0
                )
                content = resp.choices[0].message.content.strip()
                out_tokens = self.estimate_tokens(content)
                call_cost = (in_tokens / 1_000_000) * rates["input"] + (out_tokens / 1_000_000) * rates["output"]
                self.total_input_tokens += in_tokens
                self.total_output_tokens += out_tokens
                self.total_cost_usd += call_cost
                if stage in self.stage_metrics:
                    self.stage_metrics[stage]["input_tokens"] += in_tokens
                    self.stage_metrics[stage]["output_tokens"] += out_tokens
                    self.stage_metrics[stage]["estimated_cost_usd"] += call_cost
                self._log_attempt(stage, attempt + 1, 200, True, in_tokens, out_tokens, call_cost, None)
                self.stage_consec_failures[stage] = 0
                return content
            except Exception as e:
                last_error = e
                status = _extract_http_status(e)
                self._log_attempt(stage, attempt + 1, status, False, in_tokens, 0, 0.0, str(e))
                self.stage_consec_failures[stage] = self.stage_consec_failures.get(stage, 0) + 1
                logger.warning(f"Groq API call attempt {attempt + 1} failed (status {status}): {e}")
                if self.stage_consec_failures[stage] >= 2:
                    raise RuntimeError(f"Aborting stage '{stage}' after 2 consecutive failures: {e}")
                if status == 429 and attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise RuntimeError(f"Groq API call failed (status {status}): {e}")

        raise RuntimeError(f"Groq API call failed after {self.max_retries + 1} attempts: {last_error}")


def _extract_http_status(exc):
    for attr in ("status_code", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def get_llm_client(
    use_mock: bool = False,
    dry_run: bool = False,
    refresh_cache: bool = False,
    live: bool = False
) -> BaseLLMClient:
    """
    Factory function. Default (live=False) returns MockLLMClient (offline,
    cache/mock only, never reads the API key). A live GroqLLMClient is returned
    only when live=True AND GROQ_API_KEY is set (entry points pass live only with --live).
    """
    if not live or use_mock:
        from tests.mock_llm import MockLLMClient
        return MockLLMClient(dry_run=dry_run, refresh_cache=refresh_cache, cache_file="llm_cache_mock.json")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Live mode requested but GROQ_API_KEY is not set. Pass --live and load .env in the entry point.")
    return GroqLLMClient(dry_run=dry_run, refresh_cache=refresh_cache)


def __getattr__(name: str):
    if name == "MockLLMClient":
        from tests.mock_llm import MockLLMClient
        return MockLLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

