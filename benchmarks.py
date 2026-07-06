from dotenv import load_dotenv
load_dotenv()

import os
import time
import json
import torch
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

from document_pipeline import run_easyocr
from predictors import BertPredictor, LayoutLMv3Predictor, RandomForestPredictor, YoloPredictor, LiltRelationPredictor

TEST_IMG_DIR = os.path.normpath(os.getenv("FUNSD_TEST_IMG", "./FUNSD_SPLIT/test/images"))
TEST_JSON_DIR = os.path.normpath(os.getenv("FUNSD_TEST_JSN", "./FUNSD_SPLIT/test/annotations"))
YOLO_MODEL_PATH = os.path.normpath(os.getenv("YOLO_MODEL_PATH", "models/yolo/best.pt"))
OUTPUT_REPORT_DIR = os.path.normpath(os.getenv("REPORT_DIR", "./report"))
os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

CLASSES = ["question", "answer", "header", "other"]
LABEL_MAP = {"question": 0, "answer": 1, "header": 2, "other": 3}

def calculate_model_size_mb(folder_or_file_path):
    if not folder_or_file_path or not os.path.exists(folder_or_file_path):
        return 0.0
    if os.path.isfile(folder_or_file_path):
        return os.path.getsize(folder_or_file_path) / (1024 * 1024)
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_or_file_path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    return total_size / (1024 * 1024)

def get_ground_truth_labels(json_path, ocr_form_items):
    with open(json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_labels = []
    for ocr_item in ocr_form_items:
        bx1, by1, bx2, by2 = ocr_item["box"]
        best_label = "other"
        max_intersection = 0
        for gt_item in gt_data.get("form", []):
            gx1, gy1, gx2, gy2 = gt_item["box"]
            ix1, iy1 = max(bx1, gx1), max(by1, gy1)
            ix2, iy2 = min(bx2, gx2), min(by2, gy2)
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if intersection > max_intersection:
                    max_intersection = intersection
                    best_label = gt_item["label"].lower().strip()
        if best_label not in CLASSES: 
            best_label = "other"
        gt_labels.append(best_label)
    return gt_labels

def evaluate_yolo_geometric_metrics():
    """Функция расчета метрик для YOLOv8"""
    if not os.path.exists(YOLO_MODEL_PATH):
        return {"mAP50": "N/A", "mAP50-95": "N/A", "Ложные срабатывания (FP)": "N/A", "Пропуски (FN)": "N/A"}
    
    VAL_DATA_DIR = os.path.normpath(os.path.join(OUTPUT_REPORT_DIR, "./yolo_temp"))
    VAL_IMG_DIR = os.path.join(VAL_DATA_DIR, "images")
    VAL_LBL_DIR = os.path.join(VAL_DATA_DIR, "labels")
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)
    
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg'))]
    
    # Цикл генерации TXT-разметки формата YOLO для каждого тестового изображения
    for img_file in image_files:
        shutil.copy(os.path.join(TEST_IMG_DIR, img_file), os.path.join(VAL_IMG_DIR, img_file))
        with Image.open(os.path.join(TEST_IMG_DIR, img_file)) as img:
            W, H = img.size
        
        json_file_name = os.path.splitext(img_file)[0] + ".json"
        json_path = os.path.join(TEST_JSON_DIR, json_file_name)
        
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        yolo_lines = []
        for item in data.get("form", []):
            lbl = item["label"].lower().strip()
            if lbl in LABEL_MAP:
                x1, y1, x2, y2 = item["box"]
                yolo_lines.append(f"{LABEL_MAP[lbl]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        
        txt_file_name = os.path.splitext(img_file)[0] + ".txt"
        with open(os.path.join(VAL_LBL_DIR, txt_file_name), "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            
    # Создание конфигурационного файла
    with open(os.path.join(VAL_DATA_DIR, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(
            f"train: {os.path.abspath(VAL_IMG_DIR)}\n"
            f"val: {os.path.abspath(VAL_IMG_DIR)}\n"
            f"nc: 4\n"
            f"names: ['question', 'answer', 'header', 'other']\n"
        )
        
    # Запуск валидатора Ultralytics
    model = YOLO(YOLO_MODEL_PATH)
    yolo_project_dir = os.path.abspath(os.getenv("YOLO_RESULTS_DIR", "./models/yolo_runs"))
    metrics = model.val(
        data=os.path.join(VAL_DATA_DIR, "dataset.yaml"), 
        verbose=False, plots=False,
        project=yolo_project_dir, name="funsd_val", exist_ok=True
    )
    
    fp, fn = 0, 0
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        matrix = metrics.confusion_matrix.matrix
        fp = int(matrix.sum(axis=0)[-1] - matrix[-1, -1])
        fn = int(matrix.sum(axis=1)[-1] - matrix[-1, -1])
        
    shutil.rmtree(VAL_DATA_DIR, ignore_errors=True)
    return {
        "mAP50": round(metrics.results_dict["metrics/mAP50(B)"], 4),
        "mAP50-95": round(metrics.results_dict["metrics/mAP50-95(B)"], 4),
        "Ложные срабатывания (FP)": fp,
        "Пропуски (FN)": fn
    }

def run_benchmark_for_predictor(model_name, predictor_class, model_path_env_key):
    print(f"\n Начат тест архитектуры: {model_name}")
    model_path = os.getenv(model_path_env_key, "")
    model_size = calculate_model_size_mb(model_path)
    
    try: 
        predictor = predictor_class()
    except Exception as e:
        print(f"❌ Ошибка инициализации {model_name}: {e}")
        return None

    all_gt, all_preds, inference_times = [], [], []
    image_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg'))])
    
    for img_file in image_files:
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        json_path = os.path.join(TEST_JSON_DIR, os.path.splitext(img_file)[0] + ".json")
        if not os.path.exists(json_path): 
            continue
            
        ocr_results = run_easyocr(img_path)
        if not ocr_results: 
            continue
            
        words_all, boxes_all, word_to_block_map, form_items = [], [], [], []
        for block_idx, ocr_item in enumerate(ocr_results):
            x1, y1, x2, y2 = ocr_item["box"]
            text = ocr_item["text"]
            split_words = text.split()
            if not split_words: 
                continue
            word_w = (x2 - x1) / len(split_words)
            for i, w_text in enumerate(split_words):
                words_all.append(w_text)
                boxes_all.append([int(x1 + i * word_w), y1, int(x1 + (i + 1) * word_w), y2])
                word_to_block_map.append(block_idx)
            form_items.append({"id": block_idx, "box": [x1, y1, x2, y2], "text": text, "label": "other"})
            
        img_pil = Image.open(img_path).convert("RGB")
        start_time = time.perf_counter()
        predicted_labels = predictor.predict_labels(img_pil, words_all, boxes_all, word_to_block_map, len(form_items))
        inference_times.append((time.perf_counter() - start_time) * 1000)
        
        all_gt.extend(get_ground_truth_labels(json_path, form_items))
        all_preds.extend(predicted_labels)
        
    predictor.unload()
    del predictor
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()

    report_dict = classification_report(all_gt, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_gt, all_preds, labels=CLASSES)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Безопасное имя файла без запрещенных символов Windows/Linux
    safe_name = "".join([c if c.isalnum() or c in ('_', '-') else '_' for c in model_name])
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, f"cm_{safe_name.lower()}.png"), dpi=200)
    plt.close()

    base_stats = {
        "Архитектура": model_name,
        "Accuracy": round(report_dict["accuracy"], 4),
        "Precision (Weighted)": round(report_dict["weighted avg"]["precision"], 4),
        "Recall (Weighted)": round(report_dict["weighted avg"]["recall"], 4),
        "F1-Score (Weighted)": round(report_dict["weighted avg"]["f1-score"], 4),
        "mAP50": "—", 
        "mAP50-95": "—", 
        "Ложные срабатывания (FP)": "—", 
        "Пропуски (FN)": "—",
        "Инференс (средний, мс)": round(np.mean(inference_times), 2),
        "Размер на диске (МБ)": round(model_size, 2)
    }
    
    if model_name == "YOLOv8 (Vision-Only Detection)":
        yolo_det_metrics = evaluate_yolo_geometric_metrics()
        base_stats.update(yolo_det_metrics)
        
    return base_stats

def main():
    print("==================================================")
    print("         Начало автотеста всех моделей            ")
    print("==================================================")
    
    # Список тестируемых
    models_to_test = [
        ("Random Forest", RandomForestPredictor, "RANDOM_FOREST_MODEL_PATH"),
        ("BERT", BertPredictor, "MODEL_PATH_BERT"),
        ("YOLOv8 (Vision-Only Detection)", YoloPredictor, "YOLO_MODEL_PATH"),
        ("LayoutLMv3", LayoutLMv3Predictor, "MODEL_PATH_LAYOUTLMV3"),
        ("LiLT (Parallel Layout-Text Transformer)", LiltRelationPredictor, "MODEL_PATH_LILT")
    ]
    
    results_history = []
    for name, pred_class, env_key in models_to_test:
        res = run_benchmark_for_predictor(name, pred_class, env_key)
        if res: 
            results_history.append(res)
            
    # Сохранение промежуточной истории в формате JSON
    with open(os.path.join(OUTPUT_REPORT_DIR, "benchmark_history.json"), "w", encoding="utf-8") as f:
        json.dump(results_history, f, ensure_ascii=False, indent=4)
        
    from openpyxl.utils import get_column_letter

    # Создание финального датафрейма для генерации отчетов
    df = pd.DataFrame(results_history)
    excel_path = os.path.join(OUTPUT_REPORT_DIR, "document_ai_final_report.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Сравнение Архитектур")
        worksheet = writer.sheets["Сравнение Архитектур"]
        
        # Автоматическое выравнивание ширины столбцов под размер контента
        for col_idx, col in enumerate(worksheet.columns, start=1):
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col_idx)
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
    # Вывод результата в консоль
    print("\n" + "="*90)
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ АРХИТЕКТУР")
    print("="*90)
    print(df.to_markdown(index=False))
    print("="*90)
    
if __name__ == "__main__":
    main()
