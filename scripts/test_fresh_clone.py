import os
import shutil
import zipfile
import subprocess
import hashlib
import pandas as pd

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def build_zip(zip_filename: str = "submission.zip") -> list:
    """
    Walks the repo and packs submission.zip, excluding Part E items:
    EXPLAIN_ME.md, the company brief PDF, the nested duplicate folder, and
    labels/notes_labels.json (empty template). Also never packs secrets,
    caches, __pycache__, or per-run logs. Returns the packed relative paths.
    """
    EXCLUDE_DIRS = {
        "__pycache__", ".pytest_cache", ".cache", "cost_logs",
        "fresh_clone_test", ".git", "AI Intern Case Study",
    }
    EXCLUDE_FILES = {
        ".env", "submission.zip", "AI Intern Case Study.zip", "llm_ledger.jsonl", "veto_log.json",
        "EXPLAIN_ME.md", "FreightTiger_Intern_CaseStudy.pdf",
        "labels\\notes_labels.json", "labels/notes_labels.json",
    }

    target_zip = os.path.join(REPO_ROOT, zip_filename) if not os.path.isabs(zip_filename) else zip_filename
    print(f"1. Packing repository into {target_zip}...")
    packed = []
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                full_path = os.path.join(root, file)
                rel = os.path.relpath(full_path, REPO_ROOT).replace("\\", "/")
                if file.endswith(".pyc") or file in EXCLUDE_FILES or rel in EXCLUDE_FILES or any(p in EXCLUDE_DIRS for p in rel.split('/')):
                    continue
                zipf.write(full_path, arcname=rel)
                packed.append(rel)
    print(f"   Successfully packed {zip_filename} ({os.path.getsize(target_zip)} bytes, {len(packed)} files).")
    for bad in (".env", "EXPLAIN_ME.md", "FreightTiger_Intern_CaseStudy.pdf", "labels/notes_labels.json"):
        assert bad not in packed, f"FORBIDDEN file packed: {bad}"
    assert not any("__pycache__" in p for p in packed), "__pycache__ leaked into zip"
    print("   Verified: no .env, no EXPLAIN_ME.md, no PDF, no labels/notes_labels.json, no __pycache__.")
    return packed


def main():
    print("="*70)
    print("FRESH-CLONE END-TO-END EXECUTION TEST")
    print("="*70)
    
    zip_filename = "submission.zip"
    fresh_dir = os.path.join(REPO_ROOT, "fresh_clone_test")

    # 1. Create zip (shared builder with Part E exclusions)
    build_zip(zip_filename)

    
    # 2. Extract to fresh_dir
    print(f"\n2. Unpacking to clean directory: {fresh_dir}...")
    if os.path.exists(fresh_dir):
        shutil.rmtree(fresh_dir)
    os.makedirs(fresh_dir, exist_ok=True)
    
    with zipfile.ZipFile(os.path.join(REPO_ROOT, zip_filename), 'r') as zipf:
        zipf.extractall(fresh_dir)
    print("   Unpacked successfully.")
    
    # 3. Run pipeline from scratch inside fresh_dir
    print("\n3. Executing pipeline inside fresh_clone_test directory...")
    res = subprocess.run(
        ["python", "run_pipeline.py"],
        cwd=fresh_dir,
        capture_output=True,
        text=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR running pipeline:", res.stderr)
        return False
        
    # Check output hash against original output.csv
    orig_hash = compute_file_hash(os.path.join(REPO_ROOT, "output.csv"))
    fresh_hash = compute_file_hash(os.path.join(fresh_dir, "output.csv"))
    print(f"Original output.csv SHA-256: {orig_hash}")
    print(f"Fresh run output.csv SHA-256: {fresh_hash}")
    assert orig_hash == fresh_hash, "Hash mismatch between original and fresh run!"
    print("-> Fresh run output.csv matches original output.csv EXACTLY!")
    
    # 4. Run reproducibility verification inside fresh_dir
    print("\n4. Running verify_reproducibility.py in fresh_clone_test...")
    res_rep = subprocess.run(
        ["python", "scripts/verify_reproducibility.py"],
        cwd=fresh_dir,
        capture_output=True,
        text=True
    )
    print(res_rep.stdout)
    assert res_rep.returncode == 0, "Reproducibility verification failed in fresh clone!"
    
    # 5. Run evaluation suite in fresh_dir
    print("5. Running eval_suite.py in fresh_clone_test...")
    res_eval = subprocess.run(
        ["python", "evals/eval_suite.py"],
        cwd=fresh_dir,
        capture_output=True,
        text=True
    )
    print(res_eval.stdout)
    assert res_eval.returncode == 0, "Evaluation suite failed in fresh clone!"
    
    # 6. Run CLI assistant in fresh_dir
    print("6. Testing cli_assistant.py in fresh_clone_test...")
    res_cli = subprocess.run(
        ["python", "cli_assistant.py", "--question", "why did Chennai-Bangalore get pricier in March?"],
        cwd=fresh_dir,
        capture_output=True,
        text=True
    )
    print(res_cli.stdout)
    assert res_cli.returncode == 0, "CLI assistant failed in fresh clone!"

    # 7. Run agent_eval.py in fresh_dir
    print("7. Running agent_eval.py in fresh_clone_test...")
    res_agent_eval = subprocess.run(
        ["python", "agent_eval.py"],
        cwd=fresh_dir,
        capture_output=True,
        text=True
    )
    print(res_agent_eval.stdout)
    assert res_agent_eval.returncode == 0, "Agent evaluation failed in fresh clone!"

    # 8. Run adversarial_notes_eval.py in fresh_dir
    print("8. Running adversarial_notes_eval.py in fresh_clone_test...")
    res_adv = subprocess.run(
        ["python", "evals/adversarial_notes_eval.py"],
        cwd=fresh_dir,
        capture_output=True,
        text=True
    )
    print(res_adv.stdout)
    assert res_adv.returncode == 0, "Adversarial notes evaluation failed in fresh clone!"

    print("="*70)
    print("ALL FRESH-CLONE TESTS PASSED WITH 100% REPRODUCIBILITY!")
    print("="*70)
    
    # Clean up fresh_dir
    shutil.rmtree(fresh_dir)
    print("Cleaned up fresh_clone_test directory.")
    return True

if __name__ == "__main__":
    main()
