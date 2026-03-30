"""
토크나이저의 special token 확인 및 데이터셋 1개 샘플의 토큰 구조 분석
"""
from transformers import AutoTokenizer
import json

MODEL_NAME = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── 1. 토크나이저 Special Token 정보 ──
print("=" * 60)
print("토크나이저 Special Token 정보")
print("=" * 60)
print(f"  sep_token     : {tokenizer.sep_token!r}  (id: {tokenizer.sep_token_id})")
print(f"  eos_token     : {tokenizer.eos_token!r}  (id: {tokenizer.eos_token_id})")
print(f"  cls_token     : {tokenizer.cls_token!r}  (id: {tokenizer.cls_token_id})")
print(f"  bos_token     : {tokenizer.bos_token!r}  (id: {tokenizer.bos_token_id})")
print(f"  pad_token     : {tokenizer.pad_token!r}  (id: {tokenizer.pad_token_id})")
print(f"  unk_token     : {tokenizer.unk_token!r}  (id: {tokenizer.unk_token_id})")
print(f"  all_special_tokens: {tokenizer.all_special_tokens}")
print(f"  all_special_ids   : {tokenizer.all_special_ids}")
print()

# ── 2. 데이터셋 1개 샘플 로드 ──
with open("token_hallucination/v2/dataset/token_data_train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sample = data[0]
print("=" * 60)
print("첫 번째 샘플 정보")
print("=" * 60)
print(f"  question      : {sample.get('question', '')[:80]}...")
print(f"  wiki_passage  : {sample.get('wiki_passage', '')[:80]}...")
print(f"  words (처음5개): {sample.get('words', [])[:5]}")
print(f"  labels(처음5개): {sample.get('labels', [])[:5]}")
print()

# ── 3. 토크나이즈 결과 분석 ──
words = sample.get("words", [])
word_labels = sample.get("labels", [])
wiki_passage = sample.get("wiki_passage", "")
question = sample.get("question", "")

context_text = f"{question} {wiki_passage}"
context_words = context_text.split()

tokenized = tokenizer(
    text=context_words,
    text_pair=words,
    is_split_into_words=True,
    truncation="only_first",
    max_length=8192
)

input_ids = tokenized["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
word_ids = tokenized.word_ids()
sequence_ids = tokenized.sequence_ids()

print("=" * 60)
print("토크나이즈 결과")
print("=" * 60)
print(f"  총 토큰 수: {len(input_ids)}")
print()

# special token 위치 찾기
print("  Special Token 위치:")
for i, (tok, tid, sid) in enumerate(zip(tokens, input_ids, sequence_ids)):
    if sid is None:  # special token
        print(f"    위치 {i:4d}: token={tok!r:20s}  id={tid:6d}  sequence_id={sid}")

print()

# 구간별 토큰 수 세기
seq0_count = sum(1 for s in sequence_ids if s == 0)
seq1_count = sum(1 for s in sequence_ids if s == 1)
special_count = sum(1 for s in sequence_ids if s is None)
print(f"  구간 0 (질문+지문) 토큰 수: {seq0_count}")
print(f"  구간 1 (요약문/답변) 토큰 수: {seq1_count}")
print(f"  Special 토큰 수: {special_count}")
print()

# ── 4. SEP 역할 토큰 확인 ──
print("=" * 60)
print("구간 구분자(SEP 역할) 토큰 분석")
print("=" * 60)
# sequence_id가 바뀌는 경계점 찾기
for i in range(1, len(sequence_ids)):
    if sequence_ids[i] != sequence_ids[i-1]:
        start = max(0, i-2)
        end = min(len(tokens), i+3)
        print(f"\n  경계점 (위치 {i}): sequence_id {sequence_ids[i-1]} → {sequence_ids[i]}")
        for j in range(start, end):
            marker = " <<<" if j == i or j == i-1 else ""
            print(f"    [{j:4d}] token={tokens[j]!r:20s}  id={input_ids[j]:6d}  seq_id={sequence_ids[j]}{marker}")
