import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, classification_report

from token_config import LABEL_LIST
from token_data import load_jsonl_lazy, align_labels_to_tokens_sliding_window
from eval_metrics import extract_spans_from_bio, compute_span_metrics

class TokenEvaluator:
    def __init__(self, model_dir, device=None):
        self.model_dir = model_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def evaluate(self, test_data_path, show_examples=5, output_result_path="eval_results.json"):
        print("=" * 70)
        print(f"  Token Hallucination Model Evaluation")
        print(f"  Model: {self.model_dir}")
        print(f"  Device: {self.device}")
        print("=" * 70)
        
        # 메타데이터 로드
        meta_path = os.path.join(self.model_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            max_length = meta["config"]["max_length"]
            best_threshold = meta["config"].get("best_threshold", 0.5)
            print(f"  Training F1: {meta['best_f1']:.4f} (epoch {meta['epoch']})")
            print(f"  Best Threshold: {best_threshold:.2f}")
        else:
            max_length = 512
            best_threshold = 0.5
        
        # 모델 & 토크나이저 로드
        print("\n[1/3] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.model = self.model.half().to(self.device)  # FP16 추론
        self.model.eval()
        
        if torch.cuda.is_available():
            print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # 데이터 로드
        print("[2/3] Loading test data...")
        test_samples = load_jsonl_lazy(test_data_path)
        print(f"  Test samples: {len(test_samples)}")
        
        # 추론
        print("[3/3] Running inference...")
        all_preds = []
        all_labels = []
        pred_spans_list = []
        gold_spans_list = []
        example_outputs = []
        
        for idx, sample in enumerate(test_samples):
            text = sample["llm_answer"]
            char_labels = sample["labels"]
            
            # 토큰화
            encoding = self.tokenizer(
                text, truncation=True, max_length=max_length,
                padding=False, return_offsets_mapping=True, return_tensors="pt",
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast("cuda") if self.device == "cuda" else torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Softmax & Threshold application
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy() # (L, 3)
            hal_probs = probs[:, 1] + probs[:, 2]
            
            preds = np.argmax(probs, axis=-1)
            preds = np.where((preds == 0) & (hal_probs > best_threshold), 1, preds)
            
            chunked_features = align_labels_to_tokens_sliding_window(text, char_labels, self.tokenizer, max_length, stride=0)
            gold_labels = chunked_features[0]["labels"][:len(preds)]
            
            # padding 제외
            for p, g in zip(preds, gold_labels):
                if g != -100:
                    all_preds.append(p)
                    all_labels.append(g)
            
            # 스팬 추출
            pred_spans = extract_spans_from_bio(
                self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]),
                preds, offset_mapping, text
            )
            gold_spans = [{"text": text[s:e], "start": s, "end": e} for s, e in char_labels]
            
            pred_spans_list.append(pred_spans)
            gold_spans_list.append(gold_spans)
            
            if idx < show_examples:
                example_outputs.append({
                    "text": text[:300],
                    "gold_spans": gold_spans,
                    "pred_spans": pred_spans,
                })
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx+1}/{len(test_samples)}...")
        
        # 결과 계산
        print("\n" + "=" * 70)
        print("  [평가 결과 요약]")
        print("=" * 70)
        
        print("\n[1. 토큰 단위 분류 성능 (Token-Level)]")
        print("- 각 토큰(단어 조각)이 환각인지 아닌지를 개별적으로 판별한 결과입니다.")
        print(classification_report(
            all_labels, all_preds,
            labels=[0, 1, 2],
            target_names=LABEL_LIST,
            digits=4,
            zero_division=0,
        ))
        
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[1, 2], average="micro", zero_division=0
        )
        
        binary_labels = [1 if l > 0 else 0 for l in all_labels]
        binary_preds = [1 if p > 0 else 0 for p in all_preds]
        bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, pos_label=1, average="binary", zero_division=0
        )
        
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        print(f"[HAL 토큰 통합 지표]  정밀도(P): {bin_p:.4f}  재현율(R): {bin_r:.4f}  Binary F1: {bin_f1:.4f}  BIO micro F1: {f1:.4f}  정확도(ACC): {accuracy*100:.2f}%")
        print("  * 정확도(ACC): 전체 토큰 중 맞춘 토큰의 비율 (%)")
        print("  * 정밀도(P): 환각이라고 예측한 것 중 실제 환각인 비율 (O vs HAL)")
        print("  * 재현율(R): 실제 환각 중 모델이 찾아낸 비율 (O vs HAL)")
        print("  * Binary F1: 환각 구간 자체를 맞춘 조화 평균 (수정된 주 지표)")
        print("  * BIO micro F1: B-HAL/I-HAL 구분까지 포함한 엄격한 지표")
        
        span_metrics = compute_span_metrics(pred_spans_list, gold_spans_list)
        print(f"\n[2. 스팬 단위 지표 (Span-Level)]")
        print("- 단어의 시작과 끝이 실제 환각 위치와 얼마나 겹치는지 평가한 결과(최종 유저 체감 성능)입니다.")
        print(f"  정밀도: {span_metrics['span_precision']:.4f}  "
              f"재현율: {span_metrics['span_recall']:.4f}  "
              f"F1-Score: {span_metrics['span_f1']:.4f}")
        print(f"  * 탐지된 총 환각 개수: {span_metrics['total_pred_spans']}")
        print(f"  * 실제 존재한 환각 개수: {span_metrics['total_gold_spans']}")
        print(f"  * 정확히 맞춘 개수: {span_metrics['total_correct_spans']}")
        
        if show_examples > 0:
            print(f"\n{'='*70}")
            print(f"  [탐지 예시 ({show_examples}개)]")
            print(f"{'='*70}")
            
            for i, ex in enumerate(example_outputs):
                print(f"\n--- 예시 {i+1} ---")
                try:
                    # 인코딩 에러 방지
                    safe_text = ex['text'].encode('utf-8', 'ignore').decode('utf-8')
                    print(f"내용: {safe_text}...")
                except Exception:
                    print(f"내용: [인코딩 문제로 생략됨]")
                
                if ex["gold_spans"]:
                    gold_strs = [f"'{s['text'][:50]}' ({s['start']}-{s['end']})" for s in ex["gold_spans"]]
                    print(f"실제 정답 (Gold): {', '.join(gold_strs)}")
                else:
                    print("실제 정답 (Gold): (없음)")
                
                if ex["pred_spans"]:
                    pred_strs = [f"'{s['text'][:50]}' ({s['start']}-{s['end']})" for s in ex["pred_spans"]]
                    print(f"모델 예측 (Pred): {', '.join(pred_strs)}")
                else:
                    print("모델 예측 (Pred): (없음)")
        
        results = {
            "token_level": {
                "hal_precision": float(p),
                "hal_recall": float(r),
                "hal_f1": float(f1),
            },
            "span_level": {k: float(v) if isinstance(v, (float, np.floating)) else v 
                           for k, v in span_metrics.items()},
            "model_dir": self.model_dir,
            "test_samples": len(test_samples),
        }
        
        with open(output_result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_result_path}")
