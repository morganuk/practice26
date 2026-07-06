import os
import random
import shutil
from dotenv import load_dotenv

load_dotenv()

def make_split():
    src_train_dir = os.getenv("DATASET_TRAIN_DIR", "FUNSD/training_data")
    src_test_dir = os.getenv("DATASET_TEST_DIR", "FUNSD/testing_data")
    
    src_train_img = os.path.join(src_train_dir, "images")
    src_train_json = os.path.join(src_train_dir, "annotations")
    src_test_img = os.path.join(src_test_dir, "images")
    src_test_json = os.path.join(src_test_dir, "annotations")
    
    dst_dirs = {
        "train": {
            "img": os.getenv("FUNSD_TRAIN_IMG", "./FUNSD_SPLIT/train/images"),
            "json": os.getenv("FUNSD_TRAIN_JSN", "./FUNSD_SPLIT/train/annotations")
        },
        "val": {
            "img": os.getenv("FUNSD_VAL_IMG", "./FUNSD_SPLIT/images"),
            "json": os.getenv("FUNSD_VAL_JSN", "./FUNSD_SPLIT/annotations")
        },
        "test": {
            "img": os.getenv("FUNSD_TEST_IMG", "./FUNSD_SPLIT/test/images"),
            "json": os.getenv("FUNSD_TEST_JSN", "./FUNSD_SPLIT/test/annotations")
        }
    }
    
    # Создаем новые директории
    for mode in dst_dirs.values():
        os.makedirs(mode["img"], exist_ok=True)
        os.makedirs(mode["json"], exist_ok=True)
        
    # Читаем файлы исходного train для разделения на train/val (80/20)
    train_files = [os.path.splitext(f)[0] for f in os.listdir(src_train_json) if f.endswith('.json')]
    random.seed(42)
    random.shuffle(train_files)
    
    split_idx = int(len(train_files) * 0.8)
    split_mapping = {
        "train": train_files[:split_idx],
        "val": train_files[split_idx:]
    }
    
    # Распределяем Train и Val
    for mode in ["train", "val"]:
        for f in split_mapping[mode]:
            shutil.copy(os.path.join(src_train_img, f"{f}.png"), os.path.join(dst_dirs[mode]["img"], f"{f}.png"))
            shutil.copy(os.path.join(src_train_json, f"{f}.json"), os.path.join(dst_dirs[mode]["json"], f"{f}.json"))
            
    # Переносим Test без изменений
    test_files = [os.path.splitext(f)[0] for f in os.listdir(src_test_json) if f.endswith('.json')]
    for f in test_files:
        shutil.copy(os.path.join(src_test_img, f"{f}.png"), os.path.join(dst_dirs["test"]["img"], f"{f}.png"))
        shutil.copy(os.path.join(src_test_json, f"{f}.json"), os.path.join(dst_dirs["test"]["json"], f"{f}.json"))
        
    print(f"✔️ Датасет успешно реорганизован!")
    print(f"  Разбиение: Train={len(split_mapping['train'])}, Val={len(split_mapping['val'])}, Test={len(test_files)}")

if __name__ == "__main__":
    make_split()
