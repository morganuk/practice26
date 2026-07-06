import os
import easyocr

# Кэшируем инициализацию движка, чтобы он не загружал веса в память при каждом вызове
_READER_INSTANCE = None

def _get_ocr_reader():
    global _READER_INSTANCE
    if _READER_INSTANCE is None:
        import torch
        # Автоматически подхватываем GPU, если он доступен
        _READER_INSTANCE = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
    return _READER_INSTANCE

def extract_document_structure(img_path):
    """
    Сквозная функция оцифровки документа.
    Принимает: путь к изображению (или PIL.Image / numpy array).
    Возвращает: Стандартизированный список словарей в формате FUNSD-блок.
    """
    reader = _get_ocr_reader() #  Исправлено
    
    # Тонкая настройка параметров EasyOCR для документов (разделение строк и полей)
    ocr_results = reader.readtext(
        img_path, 
        paragraph=False, 
        width_ths=1.5, 
        height_ths=0.5, 
        x_ths=0.8
    )
    
    standardized_form = []
    
    for block_idx, (bbox, text, confidence) in enumerate(ocr_results):
        # Безопасный разбор вложенной геометрии точек EasyOCR [[x0,y0], [x1,y1]...]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
        
        split_words = text.split()
        if not split_words:
            continue
            
        words_list = []
        word_w = (x2 - x1) / len(split_words)
        
        # Генерируем пословные подэлементы (требование для LayoutLMv3)
        for i, w_text in enumerate(split_words):
            w_x1 = int(x1 + i * word_w)
            w_x2 = int(w_x1 + word_w)
            words_list.append({
                "box": [w_x1, y1, w_x2, y2],
                "text": w_text
            })
            
        # Формируем эталонный объект блока, независимый от типа OCR
        standardized_form.append({
            "id": block_idx,
            "box": [x1, y1, x2, y2],
            "text": text,
            "label": "other",  # Заполняется вызывающей нейросетью
            "words": words_list,
            "linking": []      # Заполняется вызывающим алгоритмом линкинга
        })
        
    return standardized_form
