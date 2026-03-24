"""
evaluate_token_hallucination.py

학습된 Token Hallucination Detection 모델을 test set에서 평가합니다. (모듈화 버전)

Usage:
    python evaluate_token_hallucination.py
    python evaluate_token_hallucination.py --model-dir ./token_hal_model_v1_restored
    python evaluate_token_hallucination.py --show-examples 10
"""
import os
import argparse
from evaluator_module import TokenEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "psiloqa_data")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "token_hal_model_v1_restored")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-data", default=os.path.join(DATA_DIR, "test.jsonl"))
    parser.add_argument("--show-examples", type=int, default=5)
    parser.add_argument("--output-result", default=os.path.join(SCRIPT_DIR, "eval_results.json"))
    args = parser.parse_args()
    
    evaluator = TokenEvaluator(args.model_dir)
    evaluator.evaluate(
        test_data_path=args.test_data, 
        show_examples=args.show_examples, 
        output_result_path=args.output_result
    )

if __name__ == "__main__":
    main()
