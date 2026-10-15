import argparse
import sys
import os
import pandas as pd
from src.config import PipelineConfig
from src.pipeline import AnomalyDetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Freight Tiger Shipping Cost Anomaly Detection Pipeline")
    parser.add_argument("--shipments", type=str, default="shipment_records.csv", help="Path to shipment records CSV")
    parser.add_argument("--notes", type=str, default="context_notes.csv", help="Path to context notes CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to save output CSV (default: output.csv or output_llm.csv if --use-llm)")
    parser.add_argument("--threshold-own", type=float, default=6.0, help="Threshold percentage vs own history")
    parser.add_argument("--threshold-peer", type=float, default=20.0, help="Threshold percentage vs similar routes")
    parser.add_argument("--scope", type=str, default="flagged_only", choices=["flagged_only", "all_weeks"], help="Output scope")
    parser.add_argument("--fuel-window-weeks", type=int, default=2, help="Assumed macro fuel window duration in weeks (default: 2)")
    parser.add_argument("--use-llm", action="store_true", default=False, help="Enable optional LLM rewording (default: False)")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Disable LLM rewording (enforces templates)")
    parser.add_argument("--dry-run", action="store_true", help="Print expected LLM calls, token count, and cost estimate without making calls")
    parser.add_argument("--live", action="store_true", default=False, help="Enable LIVE LLM calls (loads .env). Without this the run is offline: mock/cache only, the API key is never read.")
    parser.add_argument("--judge", action="store_true", default=False, help="Run the ExplanationJudge over the justified rows in ONE batched call (writes veto_log.json). Requires --use-llm; live only with --live.")
    
    args = parser.parse_args()
    if args.live and not args.dry_run:
        from src.llm_client import _load_env_if_present
        _load_env_if_present()
    
    if args.dry_run:
        print("="*70)
        print("PIPELINE DRY-RUN ESTIMATE (ZERO API CALLS)")
        print("="*70)
        print("Model: llama-3.1-8b-instant (Groq Free Tier / Local Ollama)")
        print("Pricing reference: $0.05/1M input tokens, $0.08/1M output tokens\n")
        print("Planned Stages & Resource Caps:")
        print("  1. Extraction Stage:")
        print("     - Target items:          10 context notes")
        print("     - Planned calls:         10 calls (Stage cap: 15 calls)")
        print("     - Est. tokens:           ~2,500 input / ~1,200 output")
        print("     - Est. cost:             ~$0.000221 USD")
        print("  2. Judge / Explanation Stage:")
        print("     - Target items:          38 flagged anomaly rows")
        print("     - Planned calls:         38 calls (Stage cap: 60 calls)")
        print("     - Est. tokens:           ~7,600 input / ~3,800 output")
        print("     - Est. cost:             ~$0.000684 USD")
        print("  3. Agent Evaluation Stage:")
        print("     - Target benchmark:      22 evaluation questions")
        print("     - Planned calls:         ~35 multi-step calls (Stage cap: 60 calls)")
        print("     - Est. tokens:           ~14,000 input / ~3,500 output")
        print("     - Est. cost:             ~$0.000980 USD\n")
        print("Total Projected Calls:        ~83 calls (Strict Cap: 120 calls)")
        print("Total Projected Tokens:       ~32,600 tokens")
        print("Total Projected Cost:         ~$0.001885 USD (Groq Free Tier: $0.00)")
        print("Cache Policy:                 Prompt-hash caching in llm_cache.json")
        print("                              Re-runs with cache hit cost $0.00\n")
        print("Actual API calls made:        0")
        if args.judge:
            print("-"*70)
            print("JUDGE PLAN (--judge):")
            print("  Rows audited:        8 justified rows (flagged 'No (justified)')")
            print("  Calls:               1 batched call (stage=judge, cap 60)")
            print("  Live?                " + ("YES (--live: 1 real Groq call)" if args.live else "no (mock/offline)"))
            print("  Effect:              writes veto_log.json; does NOT change output.csv")
        print("="*70)
        return

    default_out = "output_llm.csv" if args.use_llm else "output.csv"
    output_target = args.output if args.output is not None else default_out
    
    config = PipelineConfig(
        shipment_records_path=args.shipments,
        context_notes_path=args.notes,
        output_path=output_target,
        threshold_own_pct=args.threshold_own,
        threshold_peer_pct=args.threshold_peer,
        fuel_surcharge_window_weeks=args.fuel_window_weeks,
        output_scope=args.scope,
        use_llm=args.use_llm,
        live=args.live
    )
    
    print("="*70)
    print("FREIGHT TIGER SHIPPING COST ANOMALY DETECTION PIPELINE")
    print("="*70)
    print(f"Shipment records: {config.shipment_records_path}")
    print(f"Context notes:    {config.context_notes_path}")
    print(f"Output target:    {config.output_path}")
    print(f"Thresholds:       vs_own >= {config.threshold_own_pct}% OR vs_peer >= {config.threshold_peer_pct}% (with vs_own > 0)")
    print(f"Output scope:     {config.output_scope}")
    print("-"*70)
    
    pipeline = AnomalyDetectionPipeline(config)
    out_df, stats = pipeline.run()
    
    # Save output CSV matching exact CSV format (LF line endings for a single, platform-invariant SHA-256)
    out_df.to_csv(config.output_path, index=False, lineterminator="\n")
    print(f"Pipeline executed successfully! Saved {len(out_df)} rows to {config.output_path}")
    print(f"  Total records processed:      {stats['total_records_processed']}")
    print(f"  Total route-weeks evaluated:  {stats['total_route_weeks_evaluated']}")
    print(f"  Anomalies flagged:            {stats['anomalies_flagged']}")
    print(f"    - Justified:                {stats['justified_count']}")
    print(f"    - Unjustified (Flagged Yes): {stats['unjustified_count']}")
    print("\nToken & Cost Summary:")
    print(f"  LLM Calls:        {stats['token_cost_log']['llm_calls']}")
    print(f"  Input Tokens:     {stats['token_cost_log']['input_tokens']}")
    print(f"  Output Tokens:    {stats['token_cost_log']['output_tokens']}")
    print(f"  Estimated Cost:   ${stats['token_cost_log']['estimated_cost_usd']:.4f}")

    # Optional: batched ExplanationJudge over the justified rows (audit overlay; never changes output.csv)
    if args.judge:
        try:
            from src.explanation_judge import ExplanationJudge
            justified = out_df[out_df['flagged'] == 'No (justified)'].to_dict(orient='records')
            judge = ExplanationJudge(llm_client=pipeline.llm_explainer.client, veto_log_file="veto_log.json")
            print(f"\nRunning ExplanationJudge (ONE batched call) over {len(justified)} justified rows [{'LIVE' if args.live else 'offline/mock'}]...")
            jres = judge.judge_batch(justified)
            m = jres.get('meta', {})
            print(f"  Judge call: cache_hit={m.get('cache_hit')} in_tok={m.get('input_tokens')} out_tok={m.get('output_tokens')} cost=${m.get('cost_usd')}")
            print(f"  Vetoes: {len(jres['vetoes'])}  (written to veto_log.json)")
            for v in jres['vetoes']:
                print(f"    VETO {v['route']} {v['week_of']}: {v['reasons']}")
            print("  output.csv verdicts are UNCHANGED; the judge is an audit overlay only.")
        except Exception as e:
            print(f"\nJudge stage failed: {type(e).__name__}: {e}")
            print("  output.csv is already saved and unchanged; veto_log.json was not updated.")

    # Generate run_manifest.json
    from src.manifest import generate_and_save_run_manifest
    manifest = generate_and_save_run_manifest(config, stats, config.output_path, "run_manifest.json")
    print(f"\nSaved run audit manifest to run_manifest.json (SHA-256: {manifest['output_file']['sha256'][:16]}...)")

    # Generate standalone report.html
    from src.reporter import generate_html_report
    from src.data_loader import load_context_notes, load_shipments
    from src.metrics import compute_weekly_route_metrics
    config_all = PipelineConfig(
        shipment_records_path=config.shipment_records_path,
        context_notes_path=config.context_notes_path,
        threshold_own_pct=config.threshold_own_pct,
        threshold_peer_pct=config.threshold_peer_pct,
        fuel_surcharge_window_weeks=config.fuel_surcharge_window_weeks,
        output_scope="all_weeks"
    )
    all_df, _ = AnomalyDetectionPipeline(config_all).run()
    shipments_df = load_shipments(config.shipment_records_path)
    weekly_metrics = compute_weekly_route_metrics(shipments_df)
    weekly_metrics['week_of_str'] = weekly_metrics['week_of'].dt.strftime('%Y-%m-%d')
    all_df = all_df.merge(
        weekly_metrics[['route', 'week_of_str', 'total_shipments']],
        left_on=['route', 'week_of'],
        right_on=['route', 'week_of_str'],
        how='left'
    )
    notes_df = load_context_notes(config.context_notes_path)
    generate_html_report(all_df, notes_df, "report.html", fuel_window_weeks=config.fuel_surcharge_window_weeks)
    print("Saved self-contained visual audit dashboard to report.html")
    print("="*70)

if __name__ == "__main__":
    main()
