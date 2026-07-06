import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from dotenv import load_dotenv

from dataset_interface import get_document_data_loader, BIO_LABELS
from logger_utils import calculate_model_size_mb, print_model_training_passport

load_dotenv()

PATH_LAYOUTLM = os.path.normpath(os.getenv("MODEL_PATH_LAYOUTLMV3", "./saved_layoutlmv3_model"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = os.getenv("REPORT_DIR", "./report")
os.makedirs(REPORT_DIR, exist_ok=True)

class LayoutLMTrainingTracker:
    """Класс для аккумуляции метрик эпох LayoutLMv3, сохранения JSON и генерации графиков."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
        
    def log_step(self, epoch, train_loss, val_loss, val_acc):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_acc)
        print(f"📊 Эпоха {epoch} залогирована | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    def save_and_plot(self):
        # Сохранение логов в файл для возможности перестроения
        json_path = os.path.join(REPORT_DIR, "layoutlmv3_training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
            
        # Построение графиков обучения
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["epoch"], self.history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
        plt.plot(self.history["epoch"], self.history["val_loss"], 'r-s', label='Val Loss', linewidth=2)
        plt.title(f"Кривые потерь (Loss Curves) — {self.model_name}", fontsize=12, fontweight='bold')
        plt.xlabel("Эпоха", fontsize=10)
        plt.ylabel("Значение Loss", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        
        plot_path = os.path.join(REPORT_DIR, "loss_curves_layoutlmv3.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"График кривых обучения LayoutLMv3 успешно сохранен в: {plot_path}")

def calculate_token_accuracy(preds, labels):
    """Вычисляет базовую точность токенов, полностью игнорируя паддинги (-100)."""
    flat_preds = preds.flatten()
    flat_labels = labels.flatten()
    
    mask = flat_labels != -100
    filtered_preds = flat_preds[mask]
    filtered_labels = flat_labels[mask]
    
    if len(filtered_labels) == 0:
        return 0.0
    return np.mean(filtered_preds == filtered_labels)

def main():
    print("\nДообучение LayoutLMv3...")
    
    epochs = int(os.getenv("LAYOUTLM_EPOCHS", 5))
    batch_size = int(os.getenv("LAYOUTLM_BATCH_SIZE", 2))
    lr = float(os.getenv("LAYOUTLM_LR", 3e-5))
    max_length = int(os.getenv("LAYOUTLM_MAX_LENGTH", 512))
    early_stopping_patience = int(os.getenv("LAYOUTLM_PATIENCE", 3)) # Защита от переобучения мультимодального эмбеддинга
    
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(BIO_LABELS))
    
    id2label = {i: l for i, l in enumerate(BIO_LABELS)}
    model.config.id2label = id2label
    model.config.label2id = {l: i for i, l in id2label.items()}
    model.to(DEVICE)
    
    train_loader = get_document_data_loader(
        mode="train", model_type="layoutlmv3", tokenizer_or_processor=processor,
        batch_size=batch_size, max_length=max_length, augment=True
    )
    val_loader = get_document_data_loader(
        mode="val", model_type="layoutlmv3", tokenizer_or_processor=processor,
        batch_size=batch_size, max_length=max_length, augment=False
    )
    
    if len(train_loader.dataset) == 0:
        print("❌ Ошибка: Обучающая выборка для LayoutLMv3 пуста! Проверьте переменные окружения.")
        return
        
    # Использование стабильного оптимизатора с затуханием весов
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tracker = LayoutLMTrainingTracker("LayoutLMv3 (Text + Layout + Vision)")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nЗапуск эпохи {epoch+1}/{epochs}...")
        
        # --- ФАЗА ОБУЧЕНИЯ (TRAIN) ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Клиппинг для мультимодальных весов
            opt.step()
            train_loss += loss.item()
            
        # --- ФАЗА ВАЛИДАЦИИ (VALIDATION) ---
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
                
        # Расчет средних метрик за эпоху
        mean_train_loss = train_loss / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else all_preds
        all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 1 else all_labels
        val_accuracy = calculate_token_accuracy(all_preds, all_labels)
        
        # Аккумулируем данные в трекер
        tracker.log_step(epoch + 1, mean_train_loss, mean_val_loss, val_accuracy)
        
        # Контроль сохранения лучшего чекпоинта и ранней остановки
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            os.makedirs(PATH_LAYOUTLM, exist_ok=True)
            model.save_pretrained(PATH_LAYOUTLM)
            processor.save_pretrained(PATH_LAYOUTLM)
            print("🌟 Найдена лучшая эпоха! Веса мультимодальной модели перезаписаны.")
        else:
            patience_counter += 1
            print(f"⚠️ Ошибка на валидации не снизилась. Триггер ранней остановки: {patience_counter}/{early_stopping_patience}")
            
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Сработал Early Stopping на эпохе {epoch+1}! Сеть начала переобучаться.")
            break
            
    # Сохраняем финальные файлы отчета (JSON + PNG)
    tracker.save_and_plot()
    
    print_model_training_passport({
        "name": "LayoutLMv3 (Multimodal Transformer in Auto-Plot Mode)",
        "scheme": "Fine-Tuning (Early-Fusion Text + 2D Bounding Boxes + Image Patches)",
        "img_size": "224x224 пикселей для визуальных патчей текста (Visual Embedding HEAD)",
        "epochs": f"{epoch+1} (Под контролем ранней остановки)",
        "hyperparams": f"Optimizer=AdamW, lr={lr}, batch_size={batch_size}, best_val_loss={best_val_loss:.4f}",
        "size_mb": calculate_model_size_mb(PATH_LAYOUTLM),
        "saved_to": PATH_LAYOUTLM
    })

if __name__ == "__main__":
    main()
