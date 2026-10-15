.PHONY: run test eval verify clean help

help:
	@echo "Available commands:"
	@echo "  make run       - Execute anomaly detection pipeline (generates output.csv)"
	@echo "  make test      - Run all pytest unit and integration tests"
	@echo "  make eval      - Run regression eval suite and synthetic spike injection test"
	@echo "  make verify    - Run 3-run reproducibility verification"
	@echo "  make benchmark - Run TF-IDF vs FAISS retrieval benchmark"

run:
	python run_pipeline.py

test:
	python -m pytest tests/

eval:
	python evals/eval_suite.py
	python evals/synthetic_eval.py
	python agent_eval.py
	python evals/adversarial_notes_eval.py

verify:
	python scripts/verify_reproducibility.py

benchmark:
	python evals/evaluate_retrieval.py

clean:
	rm -rf __pycache__ .pytest_cache .cache fresh_clone_test
