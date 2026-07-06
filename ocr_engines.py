import os
# import pytesseract
import easyocr
import numpy as np
from PIL import Image

_EASYOCR_READER = None

def run_easyocr(img_path):
    """Реализация EasyOCR"""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import torch
        _EASYOCR_READER = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
        
    results = _EASYOCR_READER.readtext(img_path, paragraph=False, width_ths=1.5, height_ths=0.5, x_ths=0.8)
    
    parsed_items = []
    for bbox, text, _ in results:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
        
        if text.strip():
            parsed_items.append({
                "box": [x1, y1, x2, y2],
                "text": text
            })
    return parsed_items

def run_tesseract(img_path):
    """Заглушка для Tesseract OCR (pip install pytesseract)."""
    # TODO: Добавить в PATH путь к exe-драйверу
    """
    image = Image.open(img_path)
    data = pytesseract.image_to_data(image, lang='rus+eng', output_type=pytesseract.Output.DICT)
    
    parsed_items = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 40 and data['text'][i].strip():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            parsed_items.append({
                "box": [x, y, x + w, y + h],
                "text": data['text'][i]
            })
    """
    print("Tesseract нереализован. Зарезервированно на будущее")
    return []

def run_paddleocr(img_path):
    """Заглушка для PaddleOCR (pip install paddlepaddle paddleocr)."""
    return []

ENGINES = {
    "easyocr": run_easyocr,
    # "tesseract": run_tesseract,
    # "paddleocr": run_paddleocr
}

def extract_text_with_engine(engine_name, img_path):
    """Универсальный вызов движка по названию"""
    name_lower = engine_name.lower()
    if name_lower not in ENGINES:
        raise ValueError(f"Неизвестный OCR движок: {engine_name}. Доступные: {list(ENGINES.keys())}")
        
    return ENGINES[name_lower](img_path)
