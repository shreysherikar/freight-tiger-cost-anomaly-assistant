import os
import json
import hashlib
import datetime
from typing import Dict, Any

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file if it exists."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_and_save_run_manifest(
    config: Any,
    stats: Dict[str, Any],
    output_filepath: str,
    manifest_filepath: str = "run_manifest.json"
) -> Dict[str, Any]:
    """
    Generates and persists an immutable audit manifest for each pipeline run.
    Captures input data hashes, hyperparameter configuration, code version,
    row counts, verdict distributions, output SHA-256, and token/cost telemetry.
    """
    manifest = {
        "manifest_version": "1.0.0",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_version": "1.2.0",
        "data_hashes": {
            "shipment_records_sha256": compute_file_sha256(config.shipment_records_path),
            "context_notes_sha256": compute_file_sha256(config.context_notes_path)
        },
        "config": {
            "threshold_own_pct": config.threshold_own_pct,
            "threshold_peer_pct": config.threshold_peer_pct,
            "rolling_history_weeks": config.rolling_history_weeks,
            "fuel_surcharge_window_weeks": getattr(config, "fuel_surcharge_window_weeks", 2),
            "output_scope": config.output_scope,
            "use_llm": config.use_llm
        },
        "row_counts": {
            "shipments_processed": stats.get("total_records_processed", 0),
            "total_route_weeks_evaluated": stats.get("total_route_weeks_evaluated", 0),
            "anomalies_flagged": stats.get("anomalies_flagged", 0)
        },
        "verdict_counts": {
            "justified_count": stats.get("justified_count", 0),
            "unjustified_count": stats.get("unjustified_count", 0)
        },
        "output_file": {
            "path": output_filepath,
            "sha256": compute_file_sha256(output_filepath) if os.path.exists(output_filepath) else None
        },
        "token_cost_log": stats.get("token_cost_log", {})
    }

    with open(manifest_filepath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
