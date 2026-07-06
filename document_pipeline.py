from dotenv import load_dotenv
load_dotenv()

import os
import json
import cv2
import numpy as np
import easyocr
import torch
from PIL import Image

COLOR_MAP = {
    "question": (0, 165, 255),      # Оранжевый
    "answer": (0, 200, 0),          # Зеленый
    "header": (255, 0, 0),          # Синий
    "other": (128, 128, 128)        # Серый
}

_EASYOCR_READER = None

def run_easyocr(img_path):
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        # Читаем устройство из .env, если оно там прописано, иначе автодерект GPU
        env_device = os.getenv("EASYOCR_DEVICE", "cpu")
        use_gpu = True if env_device == "gpu" else (torch.cuda.is_available() if env_device == "cpu" else False)
        _EASYOCR_READER = easyocr.Reader(['ru', 'en'], gpu=use_gpu)
        
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return []
        
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
        
    # Передаем в EasyOCR матрицу numpy
    results = _EASYOCR_READER.readtext(img_cv, paragraph=False, width_ths=1.5, height_ths=0.5, x_ths=0.8)
    
    parsed_items = []
    for bbox, text, _ in results:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        if text.strip():
            parsed_items.append({
                "box": [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))],
                "text": text
            })
    return parsed_items

def get_center(box):
    return [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]

def build_links(form_items):
    questions = [item for item in form_items if item["label"] == "question"]
    answers = [item for item in form_items if item["label"] == "answer"]

    for q in questions:
        q_center = get_center(q["box"])
        best_ans, min_dist = None, float("inf")
        
        for a in answers:
            a_center = get_center(a["box"])
            dx, dy = a_center[0] - q_center[0], a_center[1] - q_center[1]
            q_h = q["box"][3] - q["box"][1]
            
            if (dx > 0 and abs(dy) < q_h * 1.2) or (dy > 0 and dx >= -q_h):
                dist = np.linalg.norm(np.array(q_center) - np.array(a_center)) * (0.7 if dx > 0 and abs(dy) < q_h * 1.2 else 1.0)
                if dist < min_dist and dist < 300:
                    min_dist, best_ans = dist, a
                    
        if best_ans:
            q["linking"].append([q["id"], best_ans["id"]])
            best_ans["linking"].append([q["id"], best_ans["id"]])
            
    return form_items

def draw_results(img_path, form_items):
    """Отрисовка предсказаний и эвристических связей"""
    img = cv2.imread(img_path)
    questions = [item for item in form_items if item["label"] == "question"]
    answers = [item for item in form_items if item["label"] == "answer"]
    
    for q in questions:
        for pair in q["linking"]:
            target_ans = next((a for a in answers if a["id"] == pair[1]), None)
            if target_ans:
                cv2.line(img, tuple(get_center(q["box"])), tuple(get_center(target_ans["box"])), (255, 0, 255), 2)
    
    for item in form_items:
        b, lbl = item["box"], item["label"]
        color = COLOR_MAP.get(lbl, (128, 128, 128))
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        cv2.putText(img, lbl.upper(), (int(b[0]), int(b[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_ground_truth(img_path, json_path):
    """Отрисовка оригинальной разметки поверх исходного документа"""
    img = cv2.imread(img_path)
    if not os.path.exists(json_path):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data.get("form", []):
        b, lbl = item["box"], item["label"].lower()
        color = COLOR_MAP.get(lbl, (128, 128, 128))
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        cv2.putText(img, f"GT_{lbl.upper()}", (int(b[0]), int(b[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_document(img_path, model_predictor):
    ocr_results = run_easyocr(img_path)
    if not ocr_results: return []

    form_items, words_all, word_to_block_map, boxes_all = [], [], [], []
    
    for block_idx, ocr_item in enumerate(ocr_results):
        x1, y1, x2, y2 = ocr_item["box"]
        text = ocr_item["text"]
        split_words = text.split()
        if not split_words: continue
            
        word_w = (x2 - x1) / len(split_words)
        words_list = []
        
        for i, w_text in enumerate(split_words):
            w_box = [int(x1 + i * word_w), y1, int(x1 + (i + 1) * word_w), y2]
            words_list.append({"box": w_box, "text": w_text})
            words_all.append(w_text)
            boxes_all.append(w_box)
            word_to_block_map.append(block_idx)
            
        form_items.append({
            "id": block_idx, "box": [x1, y1, x2, y2], "text": text,
            "label": "other", "words": words_list, "linking": []
        })
        
    img_pil = Image.open(img_path).convert("RGB")
    block_labels = model_predictor.predict_labels(img_pil, words_all, boxes_all, word_to_block_map, len(form_items))
    
    for idx, label in enumerate(block_labels):
        form_items[idx]["label"] = label
        
    return build_links(form_items)
