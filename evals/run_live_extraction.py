import os
import sys
import json
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm_client import get_llm_client, GroqLLMClient
from src.note_extractor import NoteExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("="*70)
    print("PHASE C1: RUNNING LIVE NOTE EXTRACTION (10 NOTES)")
    print("="*70)
    
    import argparse
    ap = argparse.ArgumentParser(description="Note extraction (offline mock by default)")
    ap.add_argument("--live", action="store_true", default=False, help="Load .env and use the LIVE Groq client. Default: MockLLMClient (offline).")
    a = ap.parse_args()
    if a.live:
        from src.llm_client import _load_env_if_present
        _load_env_if_present()

    notes_df = pd.read_csv("context_notes.csv")
    client = get_llm_client(live=a.live)
    print(f"Using Client: {type(client).__name__} (Model: {client.model_name})")
    
    extractor = NoteExtractor(llm_client=client)
    extracted_results = []
    disagreements = []

    for _, row in notes_df.iterrows():
        nid = row["note_id"]
        date = pd.to_datetime(row["date"])
        applies_to = row["applies_to"]
        note_text = row["note"]
        
        print(f"Extracting {nid} ({date.strftime('%Y-%m-%d')}, {applies_to})...")
        res = extractor.extract_note(nid, date, applies_to, note_text)
        extracted_results.append(res)
        
        if not res["agreement_with_rules"]:
            disagreements.append({
                "note_id": nid,
                "disagreement": res.get("disagreement_details"),
                "llm_result": res
            })
            print(f"  [DISAGREEMENT] {nid}: {res.get('disagreement_details')}")
        else:
            print(f"  [OK] {nid}: effect={res.get('cost_effect')}, applies_to={res.get('applies_to')}")

    # Save outputs
    with open("extracted_notes.json", "w", encoding="utf-8") as f:
        json.dump(extracted_results, f, indent=2)

    client.save_cost_log()

    print("\n" + "="*70)
    print("EXTRACTION SUMMARY:")
    print(f"Total Notes Extracted:       {len(extracted_results)}")
    print(f"Valid Schema Count:          {sum(1 for r in extracted_results if r['is_valid'])}")
    print(f"Rule Agreements:             {sum(1 for r in extracted_results if r['agreement_with_rules'])}")
    print(f"Disagreements Flagged:       {len(disagreements)}")
    print(f"Telemetry Logged to:         cost_log.json, llm_cache.json, extracted_notes.json")
    print("="*70)

if __name__ == "__main__":
    main()
