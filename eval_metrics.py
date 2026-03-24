import numpy as np
from token_config import LABEL2ID

def extract_spans_from_bio(tokens, labels, offset_mapping, text):
    """BIO 라벨 시퀀스에서 환각 스팬을 추출한다.
    
    Returns: [{"text": "...", "start": int, "end": int}, ...]
    """
    spans = []
    current_start = None
    current_end = None
    
    for i, (lbl, (tok_start, tok_end)) in enumerate(zip(labels, offset_mapping)):
        if lbl == LABEL2ID["B-HAL"]:
            # 이전 스팬 저장
            if current_start is not None:
                spans.append({
                    "text": text[current_start:current_end],
                    "start": current_start,
                    "end": current_end,
                })
            current_start = tok_start
            current_end = tok_end
        elif lbl == LABEL2ID["I-HAL"] and current_start is not None:
            current_end = tok_end
        else:
            if current_start is not None:
                spans.append({
                    "text": text[current_start:current_end],
                    "start": current_start,
                    "end": current_end,
                })
                current_start = None
                current_end = None
    
    # 마지막 스팬
    if current_start is not None:
        spans.append({
            "text": text[current_start:current_end],
            "start": current_start,
            "end": current_end,
        })
    
    return spans

def compute_span_metrics(pred_spans_list, gold_spans_list):
    """스팬 수준 P/R/F1 (char-level overlap 기반)."""
    total_pred = 0
    total_gold = 0
    total_correct = 0
    
    for pred_spans, gold_spans in zip(pred_spans_list, gold_spans_list):
        total_pred += len(pred_spans)
        total_gold += len(gold_spans)
        
        for ps in pred_spans:
            for gs in gold_spans:
                # overlap 비율 확인
                overlap_start = max(ps["start"], gs["start"])
                overlap_end = min(ps["end"], gs["end"])
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    pred_len = ps["end"] - ps["start"]
                    gold_len = gs["end"] - gs["start"]
                    # 50% 이상 겹치면 correct
                    if overlap_len / max(pred_len, 1) > 0.5 or overlap_len / max(gold_len, 1) > 0.5:
                        total_correct += 1
                        break
    
    precision = total_correct / max(total_pred, 1)
    recall = total_correct / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "total_pred_spans": total_pred,
        "total_gold_spans": total_gold,
        "total_correct_spans": total_correct,
    }
