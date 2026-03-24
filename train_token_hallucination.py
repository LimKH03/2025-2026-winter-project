import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 분리된 모듈 임포트
from token_config import LABEL_LIST
from token_data import load_jsonl_lazy, TokenHalDataset
from token_trainer_module import TokenTrainer

def main():
    print("Restore Mode: Running modularized token hallucination training script")
    MODEL_NAME = "answerdotai/ModernBERT-base"
    EPOCHS = 15
    LR = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "psiloqa_data", "train.jsonl")
    
    # 1. 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. 데이터 불러오기
    train_samples = load_jsonl_lazy(train_path)
    if not train_samples:
        print("Data not found. Exiting.")
        return
        
    # 3. 데이터셋 및 로더 구성
    dataset = TokenHalDataset(train_samples, tokenizer, max_length=512, stride=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 4. 분류용 모델 생성 - num_labels에 config의 길이를 세팅
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST))
    
    # 5. 옵티마이저 선언
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 6. 트레이너 초기화 후 훈련 시작
    output_dir = os.path.join(script_dir, "token_hal_model_v1_restored")
    trainer = TokenTrainer(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
        output_dir=output_dir
    )
    
    trainer.train(train_loader=loader, epochs=EPOCHS)

if __name__ == "__main__":
    main()
