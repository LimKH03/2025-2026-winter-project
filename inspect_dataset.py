import os
import json
import sys
import torch
from transformers import AutoTokenizer

# train_token_hallucination.py에서 전처리 함수를 가져오기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_token_hallucination import align_labels_to_tokens_sliding_window, LABEL_LIST, ID2LABEL

def inspect_data(num_samples=3):
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "psiloqa_data", "train.jsonl")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Reading samples from {data_path}...\n")
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    for i, sample in enumerate(samples):
        print(f"{'='*100}")
        print(f" Sample {i+1}")
        print(f"{'='*100}")
        
        passage = sample.get("wiki_passage", "").strip()
        question = sample.get("question", "").strip()
        context = f"{passage}\n\nQuestion: {question}" if passage and question else (passage or question)
        answer = sample["llm_answer"]
        char_labels = sample["labels"]
        
        print(f"[Context (split)]: {context[:100]}...")
        print(f"[Answer]: {answer}")
        print(f"[Original Labels (char-level)]: {char_labels}")
        
        # 전처리 실행 (Sliding Window)
        max_length = 512 # 가독성을 위해 짧게 설정
        stride = 128
        chunks = align_labels_to_tokens_sliding_window(context, answer, char_labels, tokenizer, max_length, stride)
        
        print(f"\nPreprocessing produced {len(chunks)} chunks.")
        
        for c_idx, chunk in enumerate(chunks):
            print(f"\n--- Chunk {c_idx+1} ---")
            input_ids = chunk["input_ids"]
            labels = chunk["labels"]
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # 토큰별 라벨 시각화
            # -100인 경우(Context/Padding)는 출력을 최소화하거나 표시
            rows = []
            for j, (tok, lbl_id) in enumerate(zip(tokens, labels)):
                if tok == "[PAD]":
                    continue
                
                label_name = ID2LABEL.get(lbl_id, "IGNORE") if lbl_id != -100 else "CTX/SP"
                
                # 환각 토큰은 강조
                if label_name in ["B-HAL", "I-HAL"]:
                    rows.append(f"[{label_name}] {tok}")
                else:
                    # 너무 많으면 생략하기 위해 간단히 표시
                    rows.append(tok)
            
            print(" | ".join(rows[:100])) # 너무 길면 잘라서 출력
            if len(rows) > 100:
                print("... (truncated)")

if __name__ == "__main__":
    inspect_data()
