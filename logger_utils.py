import os

def calculate_model_size_mb(path):
    """Размер модели на диске в мегабайтах"""
    if not path or not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    return total_size / (1024 * 1024)

def print_model_training_passport(specs):
    """Отчет обучения архитектуры"""
    print("\n" + "="*70)
    print(f"АРХИТЕКТУРА: {specs['name']}")
    print("="*70)
    print(f" Схема обучения:       {specs['scheme']}")
    print(f" Размер входного изображения: {specs['img_size']}")
    print(f" Число эпох обучения:        {specs['epochs']}")
    print(f" Параметры:             {specs['hyperparams']}")
    print(f" Размер модели:     {specs['size_mb']:.2f} МБ")
    print(f" Путь сохранения:   {specs['saved_to']}")
    print("="*70 + "\n")
