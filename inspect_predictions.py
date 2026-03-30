"""
=============================================================
  Token Hallucination 예측 결과 시각적 검사 도구
  
  저장된 모델을 로드하여 테스트 데이터셋에 대한 예측을 수행하고,
  각 토큰별로 정답(GT) vs 예측(Pred) 라벨을 색상 코드로 출력합니다.
=============================================================

색상 코드 규칙:
  [OK]  초록: 정상 토큰을 정상으로 올바르게 예측 (TN)
  [FN]  빨강: 환각 토큰을 정상으로 잘못 예측 (FN - 놓친 환각!)
  [FP]  노랑: 정상 토큰을 환각으로 잘못 예측 (FP - 오탐!)
  [TP]  보라: 환각 토큰을 환각으로 올바르게 예측 (TP)
  [--]  회색: 평가 제외 토큰 (질문/지문/특수토큰, label=-100)

사용법:
  python inspect_predictions.py                          # 기본 설정 (model_save에서 로드)
  python inspect_predictions.py --model_dir model_save   # 모델 경로 지정
  python inspect_predictions.py --num_samples 5          # 검사할 샘플 수
  python inspect_predictions.py --only_errors            # 오류가 있는 샘플만 표시
  python inspect_predictions.py --sample_idx 42          # 특정 인덱스 1개 검사
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# Windows 터미널 UTF-8 출력 강제
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")  # ANSI escape code 활성화
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_cosine_schedule_with_warmup, DataCollatorForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix

# 프로젝트 모듈 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from InteractionLayer import ModernBertWithTokenMatch
from PsiloQA_Dataset import load_dataset, align_labels_to_tokens


# =============================================
# ANSI 색상 코드 (터미널 출력용)
# =============================================
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    
    # 배경색
    BG_GREEN   = "\033[42m"    # TN: 올바른 정상 예측
    BG_RED     = "\033[41m"    # FN: 놓친 환각
    BG_YELLOW  = "\033[43m"    # FP: 오탐
    BG_MAGENTA = "\033[45m"    # TP: 올바른 환각 탐지
    BG_GRAY    = "\033[100m"   # 평가 제외 (-100)
    
    # 글자색
    FG_WHITE  = "\033[97m"
    FG_BLACK  = "\033[30m"
    FG_RED    = "\033[91m"
    FG_GREEN  = "\033[92m"
    FG_YELLOW = "\033[93m"
    FG_CYAN   = "\033[96m"


def clean_subword_token(token_str):
    """서브워드 토큰의 특수 접두사 문자를 정리합니다.
    
    ModernBERT(GPT 계열 토크나이저)는 'Ġ'(U+0120)를 공백 접두사로 사용합니다.
    이 유니코드 문자는 Windows 터미널에서 제대로 렌더링되지 않아 깨져 보입니다.
    """
    display = token_str
    # GPT 계열: Ġ -> 공백
    if display.startswith("\u0120"):
        display = " " + display[1:]
    display = display.replace("\u0120", " ")
    # BERT 계열: ## 제거
    if display.startswith("##"):
        display = display[2:]
    # SentencePiece: ▁ -> 공백
    if display.startswith("\u2581"):
        display = " " + display[1:]
    display = display.replace("\u2581", " ")
    return display


def colorize_token(token_str, gt_label, pred_label):
    """토큰에 색상을 입혀 반환합니다.
    
    색상 규칙:
      TN (정상→정상): 색상 없음 (기본 터미널 글자색 = 검정/흰색)
      FN (환각→정상): 빨간 배경 (흰 글자) - 놓친 환각!
      FP (정상→환각): 노란 배경 (검정 글자) - 오탐!
      TP (환각→환각): 초록 배경 (흰 글자) - 올바른 탐지
      -100:          회색 배경 (흐린 글자) - 평가 제외
    """
    if gt_label == -100:
        # 평가 제외 토큰 (질문/지문/특수토큰)
        return f"{Colors.DIM}{Colors.BG_GRAY}{Colors.FG_WHITE}{token_str}{Colors.RESET}"
    
    if gt_label == 0 and pred_label == 0:
        # TN: 정상 -> 정상 (올바름) -- 색상 없이 기본 텍스트
        return token_str
    
    elif gt_label == 1 and pred_label == 0:
        # FN: 환각 -> 정상 (놓침!) -- 가장 위험, 빨간 배경
        return f"{Colors.BOLD}{Colors.BG_RED}{Colors.FG_WHITE} {token_str} {Colors.RESET}"
    
    elif gt_label == 0 and pred_label == 1:
        # FP: 정상 -> 환각 (오탐) -- 노란 배경
        return f"{Colors.BOLD}{Colors.BG_YELLOW}{Colors.FG_BLACK} {token_str} {Colors.RESET}"
    
    elif gt_label == 1 and pred_label == 1:
        # TP: 환각 -> 환각 (올바르게 탐지!) -- 초록 배경
        return f"{Colors.BOLD}{Colors.BG_GREEN}{Colors.FG_WHITE} {token_str} {Colors.RESET}"
    
    return token_str


def print_legend():
    """색상 범례를 출력합니다."""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}  [범례] 색상 코드 (Color Legend){Colors.RESET}")
    print("=" * 70)
    print(f"  일반텍스트              = [OK] TN: 정상->정상 (올바른 예측, 색상 없음)")
    print(f"  {Colors.BOLD}{Colors.BG_RED}{Colors.FG_WHITE} 빨간배경 {Colors.RESET}        = [FN] 환각->정상 (놓친 환각! 가장 위험)")
    print(f"  {Colors.BOLD}{Colors.BG_YELLOW}{Colors.FG_BLACK} 노란배경 {Colors.RESET}        = [FP] 정상->환각 (오탐)")
    print(f"  {Colors.BOLD}{Colors.BG_GREEN}{Colors.FG_WHITE} 초록배경 {Colors.RESET}        = [TP] 환각->환각 (올바른 탐지)")
    print(f"  {Colors.DIM}{Colors.BG_GRAY}{Colors.FG_WHITE}회색배경{Colors.RESET}          = [--] 평가 제외 (질문/지문/특수토큰)")
    print("=" * 70 + "\n")


def load_model(model_dir, device):
    """저장된 모델을 로드합니다."""
    print(f"\n[*] 모델 로딩 중: {model_dir}")
    
    # config에서 sep_token_id 확인
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_name = config.get("_name_or_path", "answerdotai/ModernBERT-base")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 로드 (__init__에서 커스텀 레이어가 자동 생성되므로 가중치 정상 복원)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=2)
    model.to(device)
    model.eval()
    
    # 메타 정보 출력
    meta_path = os.path.join(model_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        epoch = meta.get('epoch', '?')
        print(f"  [INFO] 학습 Epoch: {epoch}")
        train_f1 = meta.get('train_f1')
        if isinstance(train_f1, float):
            print(f"  [INFO] Train F1:   {train_f1:.4f}")
        val_f1 = meta.get('curr_val_f1')
        if isinstance(val_f1, float):
            print(f"  [INFO] Val F1:     {val_f1:.4f}")
    
    print(f"  [OK] 모델 로드 완료 (device: {device})\n")
    return model, tokenizer


def inspect_sample(model, tokenizer, sample, sample_idx, device, max_length=8192, show_context=False):
    """단일 샘플에 대해 추론 후 결과를 시각적으로 출력합니다."""
    words = sample.get("words", [])
    word_labels = sample.get("labels", [])
    wiki_passage = sample.get("wiki_passage", "")
    question = sample.get("question", "")
    
    if len(words) == 0 or len(words) != len(word_labels):
        print(f"  [WARN] 샘플 {sample_idx}: 데이터 형식 오류 (건너뜀)")
        return None
    
    # 토크나이즈 & 라벨 정렬
    tokenized = align_labels_to_tokens(wiki_passage, question, words, word_labels, tokenizer, max_length)
    
    input_ids = torch.tensor([tokenized["input_ids"]], device=device)
    attention_mask = torch.tensor([tokenized["attention_mask"]], device=device)
    labels = tokenized["labels"]
    
    # 추론
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, num_labels]
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [seq_len]
    
    # 토큰 문자열 복원
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    
    # 통계 계산
    stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for gt, pr in zip(labels, preds):
        if gt == -100:
            continue
        if gt == 1 and pr == 1:
            stats["TP"] += 1
        elif gt == 0 and pr == 0:
            stats["TN"] += 1
        elif gt == 0 and pr == 1:
            stats["FP"] += 1
        elif gt == 1 and pr == 0:
            stats["FN"] += 1
    
    total_errors = stats["FP"] + stats["FN"]
    total_halluc_gt = stats["TP"] + stats["FN"]
    
    # -- 헤더 출력 --
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}  [Sample #{sample_idx}]{Colors.RESET}")
    print("=" * 80)
    print(f"  {Colors.FG_CYAN}질문:{Colors.RESET} {question[:120]}{'...' if len(question) > 120 else ''}")
    print(f"  {Colors.FG_CYAN}정답 환각 토큰 수:{Colors.RESET} {total_halluc_gt}개")
    print(f"  {Colors.FG_CYAN}총 토큰 수:{Colors.RESET} {len(tokens)}개")
    
    # 에러 요약
    if total_errors == 0:
        print(f"  {Colors.FG_GREEN}[OK] 결과: 완벽한 예측! (오류 0개){Colors.RESET}")
    else:
        print(f"  {Colors.FG_RED}[ERR] 결과: 오류 {total_errors}개{Colors.RESET}"
              f"  (FN: {stats['FN']}개  FP: {stats['FP']}개)")
    
    print(f"  {Colors.DIM}통계: TP={stats['TP']}  TN={stats['TN']}  FP={stats['FP']}  FN={stats['FN']}{Colors.RESET}")
    print()
    
    # -- 요약문/답변 구간의 토큰만 출력 --
    print(f"  {Colors.BOLD}>> 답변 토큰 (요약문 구간만 표시):{Colors.RESET}")
    print("  " + "-" * 70)
    
    line_tokens = []
    line_length = 0
    
    for i, (tok, gt, pr) in enumerate(zip(tokens, labels, preds)):
        if gt == -100 and not show_context:
            continue  # 질문/지문 구간 건너뛰기
        
        # 서브워드 표시 정리 (Ġ, ##, ▁ 등 특수 접두사 제거)
        display_tok = clean_subword_token(tok)
        
        colored = colorize_token(display_tok, gt, pr)
        
        # 줄바꿈 처리 (터미널 폭 초과 방지)
        tok_len = len(display_tok) + 2  # 색상 코드 제외 실제 글자 길이
        if line_length + tok_len > 90:
            print("  " + "".join(line_tokens))
            line_tokens = []
            line_length = 0
        
        line_tokens.append(colored)
        line_length += tok_len
    
    if line_tokens:
        print("  " + "".join(line_tokens))
    
    print()
    
    # -- 문제 토큰 목록 (FN/FP 상세) --
    if total_errors > 0:
        print(f"  {Colors.BOLD}>> 문제 토큰 상세:{Colors.RESET}")
        print("  " + "-" * 70)
        
        for i, (tok, gt, pr) in enumerate(zip(tokens, labels, preds)):
            if gt == -100:
                continue
            
            display_tok = clean_subword_token(tok)
            
            if gt == 1 and pr == 0:
                print(f"    [FN] 위치 {i:4d}: '{display_tok.strip()}'  (정답=환각, 예측=정상) <- 놓침!")
            elif gt == 0 and pr == 1:
                print(f"    [FP] 위치 {i:4d}: '{display_tok.strip()}'  (정답=정상, 예측=환각) <- 오탐!")
        print()
    
    # -- 원문 단어-라벨 비교 (GT 기준) --
    print(f"  {Colors.BOLD}>> 원본 단어 라벨 (정답 기준):{Colors.RESET}")
    print("  " + "-" * 70)
    halluc_words = [(w, l) for w, l in zip(words, word_labels) if l == 1]
    if halluc_words:
        word_strs = [f"'{w}'" for w, _ in halluc_words[:20]]
        print(f"    환각 단어: {', '.join(word_strs)}")
        if len(halluc_words) > 20:
            print(f"    ... 외 {len(halluc_words) - 20}개")
    else:
        print(f"    (환각 단어 없음 - 정상 샘플)")
    
    return stats


def run_full_evaluation(model, tokenizer, samples, device, max_length=8192):
    """전체 테스트셋에 대한 종합 평가를 수행합니다."""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}  [EVAL] 전체 테스트셋 종합 평가{Colors.RESET}")
    print("=" * 70)
    
    all_gt = []
    all_pred = []
    
    from tqdm import tqdm
    for sample in tqdm(samples, desc="전체 평가 진행"):
        words = sample.get("words", [])
        word_labels = sample.get("labels", [])
        wiki_passage = sample.get("wiki_passage", "")
        question = sample.get("question", "")
        
        if len(words) == 0 or len(words) != len(word_labels):
            continue
        
        tokenized = align_labels_to_tokens(wiki_passage, question, words, word_labels, tokenizer, max_length)
        
        input_ids = torch.tensor([tokenized["input_ids"]], device=device)
        attention_mask = torch.tensor([tokenized["attention_mask"]], device=device)
        labels = tokenized["labels"]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
        
        for gt, pr in zip(labels, preds):
            if gt == -100:
                continue
            all_gt.append(gt)
            all_pred.append(int(pr))
    
    # Classification Report
    print("\n" + classification_report(
        all_gt, all_pred,
        target_names=["Normal (0)", "Halluc (1)"],
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(all_gt, all_pred)
    print(f"  Confusion Matrix:")
    print(f"                Pred=Normal  Pred=Halluc")
    print(f"  GT=Normal    {cm[0][0]:>10d}  {cm[0][1]:>10d}")
    print(f"  GT=Halluc    {cm[1][0]:>10d}  {cm[1][1]:>10d}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Token Hallucination Prediction Inspector")
    parser.add_argument("--model_dir", type=str, default="model_save",
                        help="저장된 모델 디렉토리 경로 (기본값: model_save)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="검사할 샘플 수 (기본값: 10)")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="특정 인덱스의 샘플만 검사")
    parser.add_argument("--only_errors", action="store_true",
                        help="오류가 있는 샘플만 표시")
    parser.add_argument("--show_context", action="store_true",
                        help="질문/지문 구간도 함께 표시 (회색)")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="최대 토큰 길이 (기본값: 8192)")
    parser.add_argument("--full_eval", action="store_true",
                        help="전체 테스트셋 종합 평가 실행")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                        help="사용할 데이터 split (기본값: test)")
    parser.add_argument("--device", type=str, default=None,
                        help="사용할 디바이스 (기본: 자동 감지)")
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"  Token Hallucination Prediction Inspector")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Model Dir:  {args.model_dir}")
    print(f"  Data Split: {args.split}")
    
    # 모델 로드
    model, tokenizer = load_model(args.model_dir, device)
    
    # 테스트 데이터 로드
    print(f"[*] {args.split} 데이터 로딩 중...")
    samples = load_dataset(args.split)
    print(f"  총 {len(samples)}개 샘플 로드 완료\n")
    
    # 범례 출력
    print_legend()
    
    # 전체 평가 모드
    if args.full_eval:
        run_full_evaluation(model, tokenizer, samples, device, args.max_length)
    
    # 개별 샘플 검사
    if args.sample_idx is not None:
        # 특정 인덱스 검사
        if args.sample_idx >= len(samples):
            print(f"[ERR] 인덱스 {args.sample_idx}는 범위를 초과합니다 (최대: {len(samples)-1})")
            return
        inspect_sample(model, tokenizer, samples[args.sample_idx], args.sample_idx, 
                       device, args.max_length, args.show_context)
    else:
        # 여러 샘플 순회
        shown = 0
        total_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        
        for idx, sample in enumerate(samples):
            if shown >= args.num_samples:
                break
            
            # 미리보기: 오류가 있는 샘플만 보려면 먼저 확인
            if args.only_errors:
                words = sample.get("words", [])
                word_labels = sample.get("labels", [])
                wiki_passage = sample.get("wiki_passage", "")
                question = sample.get("question", "")
                
                if len(words) == 0 or len(words) != len(word_labels):
                    continue
                
                tokenized = align_labels_to_tokens(wiki_passage, question, words, word_labels, tokenizer, args.max_length)
                input_ids = torch.tensor([tokenized["input_ids"]], device=device)
                attention_mask = torch.tensor([tokenized["attention_mask"]], device=device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
                
                has_error = False
                for gt, pr in zip(tokenized["labels"], preds):
                    if gt == -100:
                        continue
                    if gt != pr:
                        has_error = True
                        break
                
                if not has_error:
                    continue
            
            stats = inspect_sample(model, tokenizer, sample, idx, device, 
                                   args.max_length, args.show_context)
            if stats:
                for k in total_stats:
                    total_stats[k] += stats[k]
                shown += 1
        
        # 종합 요약
        total_tokens = sum(total_stats.values())
        if total_tokens > 0:
            print("\n" + "=" * 70)
            print(f"{Colors.BOLD}  [SUMMARY] 검사 종합 요약 ({shown}개 샘플){Colors.RESET}")
            print("=" * 70)
            print(f"  총 평가 토큰: {total_tokens:,}개")
            print(f"  TP (환각 올바르게 탐지): {total_stats['TP']:>6,}개")
            print(f"  TN (정상 올바르게 유지): {total_stats['TN']:>6,}개")
            print(f"  FP (오탐):              {total_stats['FP']:>6,}개")
            print(f"  FN (놓친 환각):         {total_stats['FN']:>6,}개")
            
            precision = total_stats['TP'] / (total_stats['TP'] + total_stats['FP']) if (total_stats['TP'] + total_stats['FP']) > 0 else 0
            recall = total_stats['TP'] / (total_stats['TP'] + total_stats['FN']) if (total_stats['TP'] + total_stats['FN']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (total_stats['TP'] + total_stats['TN']) / total_tokens
            
            print(f"\n  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
