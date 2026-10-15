import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib
import tempfile
import pandas as pd
from src.config import PipelineConfig
from src.pipeline import AnomalyDetectionPipeline

def compute_file_hash(filepath: str) -> str:
    """Computes raw SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_normalized_hash(filepath: str) -> str:
    """Computes normalized LF SHA-256 (invariant to CRLF vs LF line endings)."""
    with open(filepath, 'rb') as f:
        content = f.read().replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()

def main():
    print("="*70)
    print("REPRODUCIBILITY VERIFICATION (3 UNTOUCHED RUNS)")
    print("="*70)
    
    config = PipelineConfig(
        shipment_records_path="shipment_records.csv",
        context_notes_path="context_notes.csv",
        output_scope="flagged_only"
    )
    
    hashes = []
    norm_hashes = []
    dataframes = []
    temp_files = []
    
    try:
        for run_idx in range(1, 4):
            temp_csv = f"output_run_{run_idx}.csv"
            temp_files.append(temp_csv)
            
            pipeline = AnomalyDetectionPipeline(config)
            df, stats = pipeline.run()
            df.to_csv(temp_csv, index=False, lineterminator="\n")
            
            fhash = compute_file_hash(temp_csv)
            nhash = compute_normalized_hash(temp_csv)
            hashes.append(fhash)
            norm_hashes.append(nhash)
            dataframes.append(df)
            print(f"Run {run_idx}: Raw SHA-256 = {fhash} | Normalized LF SHA-256 = {nhash}")
            
        # Verify hash equality
        all_hashes_identical = (len(set(hashes)) == 1)
        all_norm_identical = (len(set(norm_hashes)) == 1)
        
        # Verify pandas dataframe exact equality
        df_match_1_2 = dataframes[0].equals(dataframes[1])
        df_match_2_3 = dataframes[1].equals(dataframes[2])
        all_dfs_identical = df_match_1_2 and df_match_2_3
        
        print("-"*70)
        print(f"Raw Hash Match:        {'PASSED (All 3 hashes identical)' if all_hashes_identical else 'FAILED'}")
        print(f"Normalized LF Match:   {'PASSED (Identical across runs)' if all_norm_identical else 'FAILED'}")
        print(f"DataFrame Value Match: {'PASSED (All values and types identical)' if all_dfs_identical else 'FAILED'}")

        if all_hashes_identical and all_dfs_identical:
            print("\nRESULT: 100% DETERMINISTIC AND FULLY REPRODUCIBLE ACROSS RUNS.")
        else:
            print("\nRESULT: MISMATCH DETECTED!")
            sys.exit(1)
            
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
    print("="*70)

if __name__ == "__main__":
    main()
