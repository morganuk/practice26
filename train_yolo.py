import os
import json
import shutil
import torch
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO, settings

from augmentations import get_yolo_augmentation_params
from logger_utils import calculate_model_size_mb, print_model_training_passport

load_dotenv()

TRAIN_JSON_DIR = os.path.normpath(os.getenv("FUNSD_TRAIN_JSN", "./FUNSD_SPLIT/train/annotations"))
TRAIN_IMG_DIR = os.path.normpath(os.getenv("FUNSD_TRAIN_IMG", "./FUNSD_SPLIT/train/images"))
VAL_JSON_DIR = os.path.normpath(os.getenv("FUNSD_VAL_JSN", "./FUNSD_SPLIT/annotations"))
VAL_IMG_DIR = os.path.normpath(os.getenv("FUNSD_VAL_IMG", "./FUNSD_SPLIT/images"))

CLASSES = ["question", "answer", "header", "other"]
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = os.getenv("REPORT_DIR", "./report")
os.makedirs(REPORT_DIR, exist_ok=True)

def prepare_yolo_subset(json_dir, img_dir, target_base_dir, subset_name):
    """Конвертирует JSON-аннотации FUNSD в TXT-формат YOLO для конкретной выборки"""
    lbl_map = {"question": 0, "answer": 1, "header": 2, "other": 3}
    dst_img_dir = os.path.join(target_base_dir, subset_name, "images")
    dst_lbl_dir = os.path.join(target_base_dir, subset_name, "labels")
    
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    if not os.path.exists(json_dir):
        return
        
    for f in os.listdir(json_dir):
        if not f.endswith(".json"): continue
        img_name = f.replace(".json", ".png")
        src_img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(src_img_path): continue
            
        shutil.copy(src_img_path, os.path.join(dst_img_dir, img_name))
        
        with Image.open(src_img_path) as img: 
            W, H = img.size
            
        with open(os.path.join(json_dir, f), "r", encoding="utf-8") as file: 
            data = json.load(file)
            
        lines = []
        for item in data.get("form", []):
            lbl = item["label"].lower().strip()
            if lbl in lbl_map:
                x1, y1, x2, y2 = item["box"]
                x_center = (x1 + x2) / 2 / W
                y_center = (y1 + y2) / 2 / H
                width = (x2 - x1) / W
                height = (y2 - y1) / H
                lines.append(f"{lbl_map[lbl]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
        txt_name = f.replace(".json", ".txt")
        with open(os.path.join(dst_lbl_dir, txt_name), "w", encoding="utf-8") as txt: 
            txt.write("\n".join(lines))

def print_pretty_yolo_metrics(results_csv_path):
    """Вычитывает CSV-логи YOLOv8 и строит красивую сводную таблицу обучения в консоли"""
    if not os.path.exists(results_csv_path):
        return
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        print("\n📈 ХРОНОЛОГИЯ ОБУЧЕНИЯ YOLOv8 (ВЫБОРКА ЭПОХ):")
        print(f"{'Эпоха':<6} | {'Train Box Loss':<14} | {'Train Cls Loss':<14} | {'Val Box Loss':<12} | {'mAP50':<8} | {'mAP50-95':<8}")
        print("-" * 75)
        
        # Выводим каждую 3-ю эпоху и обязательно самую последнюю
        epochs_to_show = list(range(2, len(df), 3))
        if (len(df) - 1) not in epochs_to_show:
            epochs_to_show.append(len(df) - 1)
            
        for idx in epochs_to_show:
            row = df.iloc[idx]
            print(f"{int(row['epoch']):<6} | {row['train/box_loss']:<14.4f} | {row['train/cls_loss']:<14.4f} | {row['val/box_loss']:<12.4f} | {row['metrics/mAP50(B)']:<8.4f} | {row['metrics/mAP50-95(B)']:<8.4f}")
        print("-" * 75)
    except Exception as e:
        print(f"⚠️ Не удалось распарсить текстовый лог результатов YOLO: {e}")

def main():
    print("\nДообучение детектора YOLOv8...")
    
    # 0. Инициализация базовой директории и принудительных абсолютных путей в самом начале
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    epochs = int(os.getenv("YOLO_EPOCHS", 15))
    batch_size = int(os.getenv("YOLO_BATCH_SIZE", 16))
    img_size = int(os.getenv("YOLO_IMG_SIZE", 640))
    
    # Строим строгие абсолютные пути от корня проекта
    yolo_model_env = os.getenv("YOLO_MODEL_PATH", "models/yolo/best.pt")
    final_target_path = os.path.normpath(os.path.join(BASE_DIR, yolo_model_env))
    
    yolo_cache_env = os.getenv("YOLO_PRETRAINED_CACHE", "./models/yolo_cache/yolov8n.pt")
    yolo_cache_file = os.path.normpath(os.path.join(BASE_DIR, yolo_cache_env))
    
    yolo_project_env = os.getenv("YOLO_RESULTS_DIR", "./models/yolo_runs")
    yolo_project_dir = os.path.normpath(os.path.join(BASE_DIR, yolo_project_env))
    
    # Настройка окружения Ultralytics и создание папок
    settings.update({"runs_dir": yolo_project_dir, "weights_dir": os.path.dirname(yolo_cache_file)})
    os.makedirs(os.path.dirname(yolo_cache_file), exist_ok=True)
    os.makedirs(yolo_project_dir, exist_ok=True)
    
    # Подготовка временного датасета
    yolo_temp_dir = os.path.abspath(os.path.join(BASE_DIR, "temp_yolo_train_dataset"))
    shutil.rmtree(yolo_temp_dir, ignore_errors=True)
    
    prepare_yolo_subset(TRAIN_JSON_DIR, TRAIN_IMG_DIR, yolo_temp_dir, "train")
    prepare_yolo_subset(VAL_JSON_DIR, VAL_IMG_DIR, yolo_temp_dir, "val")
            
    with open(os.path.join(yolo_temp_dir, "data.yaml"), "w", encoding="utf-8") as yml:
        yml.write(f"train: {os.path.join(yolo_temp_dir, 'train', 'images')}\n")
        yml.write(f"val: {os.path.join(yolo_temp_dir, 'val', 'images')}\n")
        yml.write(f"nc: 4\n")
        yml.write(f"names: {CLASSES}\n")
        
    # Инициализация модели
    model = YOLO(yolo_cache_file if os.path.exists(yolo_cache_file) else "yolov8n.pt")
    if not os.path.exists(yolo_cache_file): 
        model.save(yolo_cache_file)
    
    yolo_aug = get_yolo_augmentation_params()
    run_name = "funsd_train"
    
    # Запуск обучения
    model.train(
        data=os.path.join(yolo_temp_dir, "data.yaml"), 
        epochs=epochs, 
        batch=batch_size,
        imgsz=img_size, 
        device=DEVICE, 
        verbose=False,
        project=yolo_project_dir, 
        name=run_name, 
        exist_ok=True,
        hsv_h=yolo_aug["hsv_h"], 
        degrees=yolo_aug["degrees"], 
        translate=yolo_aug["translate"], 
        scale=yolo_aug["scale"]
    )
    
    # Определение путей к артефактам на основе yolo_project_dir и run_name
    run_output_dir = os.path.join(yolo_project_dir, run_name)
    trained_best_weights = os.path.join(run_output_dir, "weights", "best.pt")
    yolo_plots_source = os.path.join(run_output_dir, "results.png")
    yolo_csv_source = os.path.join(run_output_dir, "results.csv")
    
    # Пост-обработка: Копирование графиков обучения в общую папку отчетов
    if os.path.exists(yolo_plots_source):
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_plot_path = os.path.join(REPORT_DIR, "loss_curves_yolov8.png")
        shutil.copy(yolo_plots_source, report_plot_path)
        print(f"Сводный график обучения YOLOv8 скопирован в: {report_plot_path}")
        
    # Вывод красивой текстовой таблицы прогресса в консоль
    if os.path.exists(yolo_csv_source):
        print_pretty_yolo_metrics(yolo_csv_source)
    else:
        print("⚠️ Файл результатов обучения results.csv не найден.")
    
    # 3. Валидация лучшей модели для получения точных финальных метрик по классам
    final_metrics_str = "Метрики недоступны"
    if os.path.exists(trained_best_weights):
        os.makedirs(os.path.dirname(final_target_path), exist_ok=True)
        shutil.copy(trained_best_weights, final_target_path)
        print(f"Веса успешно сохранены в целевую директорию: {final_target_path}")
        
        print("\nЗапуск финальной валидации лучших весов...")
        best_model = YOLO(final_target_path)
        val_results = best_model.val(data=os.path.join(yolo_temp_dir, "data.yaml"), imgsz=img_size, device=DEVICE, verbose=False)
        final_metrics_str = f"mAP50={val_results.box.map50:.4f}, mAP50-95={val_results.box.map:.4f}"
    else:
        print(f"❌ Ошибка: Итоговые веса не найдены по пути: {trained_best_weights}")
        
    # Очистка временных файлов
    shutil.rmtree(yolo_temp_dir, ignore_errors=True)
    
    print_model_training_passport({
        "name": "YOLOv8 (Vision-Only Object Detection with Validation)",
        "scheme": "Fine-Tuning Anchor-Free Object Detector (Ultralytics Engine)",
        "img_size": f"{img_size}x{img_size} пикселей (Интерполяция RGB)",
        "epochs": epochs,
        "hyperparams": f"Optimizer=AdamW/SGD(Auto), batch_size={batch_size}, {final_metrics_str}",
        "size_mb": calculate_model_size_mb(final_target_path) if os.path.exists(final_target_path) else 0,
        "saved_to": final_target_path
    })


if __name__ == "__main__":
    main()
