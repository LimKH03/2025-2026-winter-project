import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TokenTrainer:
    def __init__(self, model, optimizer, device, output_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir

    def train(self, train_loader, epochs):
        self.model.to(self.device)
        print("Training Started...")
        
        for ep in range(epochs):
            self.model.train()
            total_loss = 0
            
            # 평가지표용 리스트
            all_preds = []
            all_labels = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {ep+1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 예측값 및 레이블 추출 (성능 평가용)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                labels = batch["labels"]
                
                # 배치 데이터를 numpy로 변환 및 평탄화 (GPU -> CPU)
                preds = preds.view(-1).cpu().numpy()
                labels = labels.view(-1).cpu().numpy()
                
                # 패딩 토큰(-100)을 제외하고 유효한 라벨만 필터링
                valid_mask = labels != -100
                all_preds.extend(preds[valid_mask])
                all_labels.extend(labels[valid_mask])
            
            avg_loss = total_loss / len(train_loader)
            
            # Epoch 단위의 정확도(Acc) 및 F1 스코어 계산
            acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
            
            # 결과 출력란에 각종 메트릭 추가
            print(f"Epoch {ep+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
            
            # 모델 저장
            os.makedirs(self.output_dir, exist_ok=True)
            self.model.save_pretrained(self.output_dir)
            
        print("Training complete.")
