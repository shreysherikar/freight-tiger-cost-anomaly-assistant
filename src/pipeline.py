import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.config import PipelineConfig
from src.data_loader import load_shipments, load_context_notes
from src.metrics import compute_weekly_route_metrics
from src.baselines import compute_baselines, format_deviation_own, format_deviation_peer
from src.flagging import identify_candidate_spikes
from src.retriever import NoteRetriever
from src.note_matcher import DeterministicNoteMatcher
from src.explainer import ReasonExplainer
from src.llm_explainer import LLMExplainer

class AnomalyDetectionPipeline:
    """
    End-to-end pipeline that flags shipping routes with unjustified cost increases.
    """
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.explainer = ReasonExplainer(use_llm=False)
        self.llm_explainer = LLMExplainer(
            use_llm=self.config.use_llm,
            cost_log_file=self.config.cost_log_path,
            live=self.config.live
        )

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes the full pipeline and returns the output DataFrame
        along with run metadata and token/cost logs.
        """
        # 1. Load data
        shipments_df = load_shipments(self.config.shipment_records_path)
        notes_df = load_context_notes(self.config.context_notes_path)
        
        # 2. Compute weekly route metrics (ratio-of-sums)
        weekly_df = compute_weekly_route_metrics(shipments_df)
        
        # 3. Compute own history and peer baselines
        baselines_df = compute_baselines(
            weekly_df,
            rolling_window=self.config.rolling_history_weeks
        )
        
        # 4. Identify candidate spikes
        flagged_df = identify_candidate_spikes(
            baselines_df,
            threshold_own_pct=self.config.threshold_own_pct,
            threshold_peer_pct=self.config.threshold_peer_pct
        )
        
        # 5. Initialize retrieval and matcher
        retriever = NoteRetriever(notes_df)
        matcher = DeterministicNoteMatcher(notes_df, fuel_window_weeks=self.config.fuel_surcharge_window_weeks)
        
        # 6. Evaluate flagged weeks against notes
        output_rows = []
        
        # Determine rows to process based on output_scope
        if self.config.output_scope == "flagged_only":
            candidates_to_process = flagged_df[flagged_df['is_candidate_spike']].copy()
        else:
            candidates_to_process = flagged_df.copy()
            
        for _, row in candidates_to_process.iterrows():
            route = row['route']
            week_of = row['week_of']
            cptk = row['cost_per_tonne_km']
            pct_own = row['pct_vs_own_history']
            pct_peer = row['pct_vs_similar_routes']
            is_spike = row['is_candidate_spike']
            
            if is_spike:
                # Retrieve candidate notes
                candidates = retriever.retrieve_candidates(route, week_of, top_k=5)
                # Deterministic rule evaluation
                verdict_res = matcher.evaluate_candidates(route, week_of, candidates)
                flagged_status = verdict_res['verdict']
                matched_id = verdict_res['matched_note_id']
                reason_text = self.explainer.generate_explanation(
                    route, week_of, verdict_res,
                    pct_vs_own_history=pct_own,
                    fuel_window_weeks=self.config.fuel_surcharge_window_weeks
                )
                
                # Optional LLM rewording if enabled
                if self.config.use_llm:
                    matched_note = verdict_res.get('matched_note')
                    note_date_str = matched_note.date.strftime('%Y-%m-%d') if matched_note else None
                    note_text_str = matched_note.note_text if matched_note else None
                    reason_text = self.llm_explainer.rewrite_explanation(
                        route=route,
                        week_of=week_of.strftime('%Y-%m-%d'),
                        verdict=flagged_status,
                        matched_note_id=matched_id,
                        template_reason=reason_text,
                        note_date=note_date_str,
                        note_text=note_text_str
                    )
            else:
                flagged_status = "No"
                matched_id = ""
                reason_text = "Cost is within normal historical and peer bounds."
                
            output_rows.append({
                'route': route,
                'week_of': week_of.strftime('%Y-%m-%d'),
                'cost_per_tonne_km': round(cptk, 2),
                'vs_own_history': format_deviation_own(pct_own),
                'vs_similar_routes': format_deviation_peer(pct_peer),
                'flagged': flagged_status,
                'matched_note_id': matched_id,
                'reason': reason_text
            })
            
        out_df = pd.DataFrame(output_rows)
        # Contract column order
        cols = [
            'route', 'week_of', 'cost_per_tonne_km',
            'vs_own_history', 'vs_similar_routes',
            'flagged', 'matched_note_id', 'reason'
        ]
        out_df = out_df[cols]
        
        # Sort by week_of, then route
        out_df = out_df.sort_values(by=['week_of', 'route']).reset_index(drop=True)
        
        # Save cost log to disk
        self.llm_explainer.save_cost_log()
        
        run_stats = {
            "total_records_processed": len(shipments_df),
            "total_route_weeks_evaluated": len(flagged_df),
            "anomalies_flagged": len(output_rows),
            "justified_count": sum(1 for r in output_rows if r['flagged'] == 'No (justified)'),
            "unjustified_count": sum(1 for r in output_rows if r['flagged'] == 'Yes'),
            "token_cost_log": {
                "use_llm": self.llm_explainer.use_llm,
                "model_name": self.llm_explainer.model_name,
                "llm_calls": self.llm_explainer.call_count,
                "input_tokens": self.llm_explainer.input_tokens,
                "output_tokens": self.llm_explainer.output_tokens,
                "estimated_cost_usd": self.llm_explainer.estimated_cost_usd
            }
        }
        
        return out_df, run_stats
