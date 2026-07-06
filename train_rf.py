import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from logger_utils import calculate_model_size_mb, print_model_training_passport

load_dotenv()

TRAIN_JSON_DIR = os.path.normpath(os.getenv("FUNSD_TRAIN_JSN", "./FUNSD_SPLIT/train/annotations"))
TRAIN_IMG_DIR = os.path.normpath(os.getenv("FUNSD_TRAIN_IMG", "./FUNSD_SPLIT/train/images"))
VAL_JSON_DIR = os.path.normpath(os.getenv("FUNSD_VAL_JSN", "./FUNSD_SPLIT/annotations"))
VAL_IMG_DIR = os.path.normpath(os.getenv("FUNSD_VAL_IMG", "./FUNSD_SPLIT/images"))
PATH_RF = os.path.normpath(os.getenv("RANDOM_FOREST_MODEL_PATH", "./saved_rf_baseline.pkl"))
REPORT_DIR = os.getenv("REPORT_DIR", "./report")
os.makedirs(REPORT_DIR, exist_ok=True)

def extract_rf_features(json_dir, img_dir, augment=True):
    X, y = [], []
    lbl_map = {"question": 0, "answer": 1, "header": 2, "other": 3}
    if not os.path.exists(json_dir):
        return X, y
        
    for f in os.listdir(json_dir):
        if not f.endswith(".json"): continue
        with open(os.path.join(json_dir, f), "r", encoding="utf-8") as file:
            data = json.load(file)
        img_name = f.replace(".json", ".png")
        if not os.path.exists(os.path.join(img_dir, img_name)): continue
        with Image.open(os.path.join(img_dir, img_name)) as img:
            W, H = img.size
            
        for item in data.get("form", []):
            lbl = item["label"].lower().strip()
            if lbl not in lbl_map: continue
            
            box = item["box"]
            if augment:
                from augmentations import augment_bounding_box
                box = augment_bounding_box(box, W, H)
                
            bx_w = box[2] - box[0]
            bx_h = box[3] - box[1]
            text = item.get("text", "")
            
            features = [
                bx_w / W, bx_h / H, (bx_w * bx_h) / (W * H), bx_w / (bx_h + 1e-6),
                (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H,
                float(len(text)), float(len(text.split())),
                1.0 if text.strip().endswith(":") else 0.0,
                1.0 if any(c.isdigit() for c in text) else 0.0
            ]
            X.append(features)
            y.append(lbl_map[lbl])
    return X, y

def generate_validation_curve(X_train, y_train, X_val, y_val, n_estimators, target_depth):
    """
    Аккумулирует метрики качества при изменении глубины дерева 
    и строит кривую валидации для отчета.
    """
    print("\nАккумуляция процесса обучения (построение Validation Curve)...")
    depths = list(range(2, target_depth + 3, 2))
    if target_depth not in depths:
        depths.append(target_depth)
    depths = sorted(depths)
    
    train_accs, val_accs = [], []
    
    for d in depths:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=d, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        val_accs.append(clf.score(X_val, y_val))
        print(f"  └─ Глубина max_depth={d:02d} | Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")
        
    # Построение и сохранение графика
    plt.figure(figsize=(9, 5))
    plt.plot(depths, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    plt.plot(depths, val_accs, 'r-s', label='Validation Accuracy', linewidth=2)
    plt.axvline(x=target_depth, color='g', linestyle='--', label=f'Выбранная глубина ({target_depth})')
    
    plt.title("Кривая валидации Random Forest (Обоснование глубины)", fontsize=12, fontweight='bold')
    plt.xlabel("Максимальная глубина дерева (max_depth)", fontsize=10)
    plt.ylabel("Точность классификации (Accuracy)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    plot_path = os.path.join(REPORT_DIR, "validation_curve_rf.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✔️ График кривой валидации успешно сохранен в: {plot_path}")

def print_pretty_confusion_matrix(cm, class_names):
    """Выводит матрицу ошибок в виде структурированной текстовой таблицы."""
    print("\n🧩 МАТРИЦА ОШИБОК (CONFUSION MATRIX):")
    print(f"{'Предсказано ->':<15} | " + " | ".join(f"{name:<10}" for name in class_names))
    print("-" * 65)
    for i, row in enumerate(cm):
        row_str = " | ".join(f"{val:<10}" for val in row)
        print(f"Истинно: {class_names[i]:<9} | {row_str}")
    print("-" * 65)

def main():
    print("\nОбучение Random Forest")
    
    n_estimators = int(os.getenv("RF_N_ESTIMATORS", 150))
    max_depth = int(os.getenv("RF_MAX_DEPTH", 12))
    
    X_train, y_train = extract_rf_features(TRAIN_JSON_DIR, TRAIN_IMG_DIR, augment=True)
    X_val, y_val = extract_rf_features(VAL_JSON_DIR, VAL_IMG_DIR, augment=False)
    
    if not X_train:
        print("❌ Ошибка: Обучающая выборка признаков пуста. Проверьте пути!")
        return
        
    # Генерируем данные для отчета и строим график, если доступна валидация
    if X_val:
        generate_validation_curve(X_train, y_train, X_val, y_val, n_estimators, max_depth)
    
    # Финальное обучение целевой модели
    print("\nОбучение финальной архитектуры...")
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    val_accuracy = 0.0
    class_names = ["question", "answer", "header", "other"]
    
    if X_val:
        val_accuracy = rf.score(X_val, y_val)
        y_pred = rf.predict(X_val)
        
        # Расширенный текстовый отчет по классам
        print("\nОтчет по обучению")
        print(classification_report(y_val, y_pred, target_names=class_names))
        
        # Структурированная матрица ошибок
        cm = confusion_matrix(y_val, y_pred)
        print_pretty_confusion_matrix(cm, class_names)
    else:
        print("⚠️ Предупреждение: Валидационная выборка признаков пуста!")
        
    os.makedirs(os.path.dirname(PATH_RF), exist_ok=True)
    joblib.dump(rf, PATH_RF)
    
    print_model_training_passport({
        "name": "Random Forest (Classic ML Baseline with Validation)",
        "scheme": "Supervised Learning (Scikit-Learn Classifier)",
        "img_size": "Неприменимо (Вход: 10 геометрических и текстовых признаков)",
        "epochs": "1 (Прямой расчет критериев информативности Джини)",
        "hyperparams": f"n_estimators={n_estimators}, max_depth={max_depth}, targets={len(class_names)}",
        "size_mb": calculate_model_size_mb(PATH_RF),
        "saved_to": PATH_RF
    })

if __name__ == "__main__":
    main()
