from dotenv import load_dotenv
load_dotenv()

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForTokenClassification

from dataset_interface import get_document_data_loader, BIO_LABELS
from logger_utils import calculate_model_size_mb, print_model_training_passport

PATH_BERT = os.path.normpath(os.getenv("MODEL_PATH_BERT", "./saved_bert_model"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = os.getenv("REPORT_DIR", "./report")
os.makedirs(REPORT_DIR, exist_ok=True)

class TrainingHistoryTracker:
    """Класс для аккумуляции метрик эпох, сохранения JSON и генерации графиков."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
        
    def log_step(self, epoch, train_loss, val_loss, val_acc):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_acc)
        print(f"Эпоха {epoch} залогирована | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    def save_and_plot(self):
        # Сохранение истории в JSON
        json_path = os.path.join(REPORT_DIR, "bert_training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
            
        # Построение графиков потерь
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["epoch"], self.history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
        plt.plot(self.history["epoch"], self.history["val_loss"], 'r-s', label='Val Loss', linewidth=2)
        plt.title(f"Кривые потерь (Loss Curves) — {self.model_name}", fontsize=12, fontweight='bold')
        plt.xlabel("Эпоха", fontsize=10)
        plt.ylabel("Значение Loss", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        
        plot_path = os.path.join(REPORT_DIR, "loss_curves_bert.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"График кривых обучения сохранен в: {plot_path}")

def calculate_metrics(preds, labels, idx2label):
    """Вычисляет базовую точность токенов (Accuracy) без учета паддингов (-100)."""
    flat_preds = preds.flatten()
    flat_labels = labels.flatten()
    
    # Фильтруем технические токены (Ignored index в PyTorch CrossEntropy по умолчанию -100)
    mask = flat_labels != -100
    filtered_preds = flat_preds[mask]
    filtered_labels = flat_labels[mask]
    
    if len(filtered_labels) == 0:
        return 0.0
    return np.mean(filtered_preds == filtered_labels)

def main():
    print("\nДообучение Pure BERT...")
    
    epochs = int(os.getenv("BERT_EPOCHS", 5))
    batch_size = int(os.getenv("BERT_BATCH_SIZE", 4))
    lr = float(os.getenv("BERT_LR", 5e-5))
    max_length = int(os.getenv("BERT_MAX_LENGTH", 512))
    early_stopping_patience = int(os.getenv("BERT_PATIENCE", 3)) # Защита от оверфиттинга
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(BIO_LABELS))
    
    id2label = {i: l for i, l in enumerate(BIO_LABELS)}
    model.config.id2label = id2label
    model.config.label2id = {l: i for i, l in id2label.items()}
    model.to(DEVICE)
    
    train_loader = get_document_data_loader(
        mode="train", model_type="bert", tokenizer_or_processor=tokenizer, 
        batch_size=batch_size, max_length=max_length, augment=True
    )
    val_loader = get_document_data_loader(
        mode="val", model_type="bert", tokenizer_or_processor=tokenizer, 
        batch_size=batch_size, max_length=max_length, augment=False
    )
    
    if len(train_loader.dataset) == 0:
        print("❌ Ошибка: Обучающая выборка для BERT пуста!")
        return
        
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tracker = TrainingHistoryTracker("Pure BERT Token Classifier")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nЗапуск эпохи {epoch+1}/{epochs}...")
        
        # --- ФАЗА ОБУЧЕНИЯ ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Защита от взрыва градиентов
            opt.step()
            train_loss += loss.item()
            
        # --- ФАЗА ВАЛИДАЦИИ ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**inputs)
                
                val_loss += outputs.loss.item()
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = inputs['labels'].cpu().numpy()
                
                preds = np.argmax(logits, axis=-1)
                all_preds.append(preds)
                all_labels.append(label_ids)
                
        # Расчет средних эпохальных метрик
        mean_train_loss = train_loss / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else all_preds[0]
        all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 1 else all_labels[0]
        val_accuracy = calculate_metrics(all_preds, all_labels, id2label)
        
        # Передаем метрики в аккумулирующий трекер
        tracker.log_step(epoch + 1, mean_train_loss, mean_val_loss, val_accuracy)
        
        # Логика сохранения лучшей контрольной точки (Best Checkpoint) и Early Stopping
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            # Сохраняем только веса лучшей эпохи
            os.makedirs(PATH_BERT, exist_ok=True)
            model.save_pretrained(PATH_BERT)
            tokenizer.save_pretrained(PATH_BERT)
            print("🌟 Найдена лучшая эпоха! Веса модели перезаписаны.")
        else:
            patience_counter += 1
            print(f"⚠️ Val Loss не уменьшился. Страйк ранней остановки: {patience_counter}/{early_stopping_patience}")
            
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Сработал Early Stopping на эпохе {epoch+1}! Обучение прервано во избежание оверфиттинга.")
            break
            
    # Генерация графиков по итогам обучения
    tracker.save_and_plot()
    
    print_model_training_passport({
        "name": "Pure BERT (Validation & Auto-Plot Mode)",
        "scheme": "Fine-Tuning (Token Classification HEAD)",
        "img_size": f"Неприменимо (Вход: Текст, max_length={max_length})",
        "epochs": f"{epoch+1} (Контроль ранней остановки)",
        "hyperparams": f"Optimizer=AdamW, lr={lr}, batch_size={batch_size}, best_val_loss={best_val_loss:.4f}",
        "size_mb": calculate_model_size_mb(PATH_BERT),
        "saved_to": PATH_BERT
    })

if __name__ == "__main__":
    main()
