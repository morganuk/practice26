from dotenv import load_dotenv
load_dotenv()

import os
import gc
import json
import torch
import joblib
import numpy as np
from ultralytics import YOLO
from transformers import BertTokenizerFast, BertForTokenClassification, AutoProcessor, LayoutLMv3ForTokenClassification, LiltForTokenClassification, AutoTokenizer
# from transformers import Qwen2_5_VLForConditionalGeneration

class BertPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("MODEL_PATH_BERT", "./saved_bert_model")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_labels(self, img_pil, words, boxes, word_to_block_map, num_blocks):
        if not words:
            return ["other"] * num_blocks
            
        encoding = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = np.atleast_1d(outputs.logits.argmax(-1).squeeze(0).cpu().numpy()).tolist()
        word_ids = encoding.word_ids()
        
        block_votes = {i: [] for i in range(num_blocks)}
        for pred, word_idx in zip(predictions, word_ids):
            if word_idx is not None and word_idx < len(word_to_block_map):
                block_votes[word_to_block_map[word_idx]].append(self.model.config.id2label[pred].replace("B-", "").replace("I-", ""))
                
        return [max(set(votes), key=votes.count).lower() if votes else "other" for i, votes in block_votes.items()]

    def unload(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'tokenizer'): del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class LayoutLMv3Predictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("MODEL_PATH_LAYOUTLMV3", "./saved_layoutlmv3_model")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_labels(self, img_pil, words, boxes, word_to_block_map, num_blocks):
        if not words:
            return ["other"] * num_blocks
            
        w, h = img_pil.size
        norm_boxes = [
            [int(1000*(x1/w)), int(1000*(y1/h)), int(1000*(x2/w)), int(1000*(y2/h))] 
            for (x1, y1, x2, y2) in boxes
        ]
        
        encoding = self.processor(img_pil, words, boxes=norm_boxes, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = np.atleast_1d(outputs.logits.argmax(-1).squeeze(0).cpu().numpy()).tolist()
        word_ids = encoding.word_ids()
        
        block_votes = {i: [] for i in range(num_blocks)}
        for pred, word_idx in zip(predictions, word_ids):
            if word_idx is not None and word_idx < len(word_to_block_map):
                block_votes[word_to_block_map[word_idx]].append(self.model.config.id2label[pred].replace("B-", "").replace("I-", ""))
                
        return [max(set(votes), key=votes.count).lower() if votes else "other" for i, votes in block_votes.items()]

    def unload(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'processor'): del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class RandomForestPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("RANDOM_FOREST_MODEL_PATH", "./saved_rf_baseline.pkl")
            
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.classes_list = ["question", "answer", "header", "other"]
        else:
            self.model = None

    def predict_labels(self, img_pil, words, boxes, word_to_block_map, num_blocks):
        if self.model is None:
            return ["other"] * num_blocks
            
        width, height = img_pil.size
        
        # Группируем слова EasyOCR обратно по уникальным ID исходных блоков
        blocks_data = {i: {"boxes": [], "text_pieces": []} for i in range(num_blocks)}
        for w_text, w_box, b_idx in zip(words, boxes, word_to_block_map):
            if b_idx < num_blocks:
                blocks_data[b_idx]["boxes"].append(w_box)
                blocks_data[b_idx]["text_pieces"].append(w_text)
                
        features_list = []
        for i in range(num_blocks):
            b_boxes = blocks_data[i]["boxes"]
            b_texts = blocks_data[i]["text_pieces"]
            
            if not b_boxes:
                features_list.append([0.0] * 10)
                continue
                
            # Восстанавливаем границы блока EasyOCR
            x1 = min([b[0] for b in b_boxes])
            y1 = min([b[1] for b in b_boxes])
            x2 = max([b[2] for b in b_boxes])
            y2 = max([b[3] for b in b_boxes])
            
            text = " ".join(b_texts)
            
            b_width = x2 - x1
            b_height = y2 - y1
            area = b_width * b_height
            aspect_ratio = b_width / (b_height + 1e-6)
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            text_len = len(text)
            word_count = len(b_texts)
            has_colon = 1.0 if text.strip().endswith(":") else 0.0
            has_digits = 1.0 if any(char.isdigit() for char in text) else 0.0
            
            features = [
                b_width / width, b_height / height, area / (width * height), aspect_ratio,
                x_center, y_center, float(text_len), float(word_count), has_colon, has_digits
            ]
            features_list.append(features)
            
        preds_encoded = self.model.predict(features_list)
        return [self.classes_list[idx] for idx in preds_encoded]

    def unload(self):
        if hasattr(self, 'model'): 
            del self.model
        gc.collect()
        
class YoloPredictor:
    """YOLOv8 - детектор объектов на основе изображения."""
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("YOLO_MODEL_PATH", "models/yolo/best.pt")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Файл обученных весов YOLOv8 не найден: {os.path.abspath(model_path)}. "
                f"Проверьте путь в .env или запустите train_all.py."
            )
            
        self.model = YOLO(model_path)
        self.classes_map = {0: "question", 1: "answer", 2: "header", 3: "other"}
        
        if hasattr(self.model, "names") and self.model.names:
            self.classes_map = {idx: str(name).lower().strip() for idx, name in self.model.names.items()}

    def predict_labels(self, img_pil, words, boxes, word_to_block_map, num_blocks):
        if self.model is None or not boxes: 
            return ["other"] * num_blocks
            
        results = self.model(img_pil, verbose=False)
        
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        yolo_cls = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []
        
        block_labels = ["other"] * num_blocks
        blocks_geom = {i: [] for i in range(num_blocks)}
        for w_box, b_idx in zip(boxes, word_to_block_map):
            if b_idx < num_blocks: 
                blocks_geom[b_idx].append(w_box)
                
        for i in range(num_blocks):
            b_boxes = blocks_geom[i]
            if not b_boxes: 
                continue
            
            bx1 = min([b[0] for b in b_boxes])
            by1 = min([b[1] for b in b_boxes])
            bx2 = max([b[2] for b in b_boxes])
            by2 = max([b[3] for b in b_boxes])
            
            best_cls, max_intersection = "other", 0
            for y_box, cls_id in zip(yolo_boxes, yolo_cls):
                yx1, yy1, yx2, yy2 = y_box
                ix1, iy1 = max(bx1, yx1), max(by1, yy1)
                ix2, iy2 = min(bx2, yx2), min(by2, yy2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    if intersection > max_intersection:
                        max_intersection = intersection
                        best_cls = self.classes_map.get(int(cls_id), "other")
            block_labels[i] = best_cls
        return block_labels

    def unload(self):
        if hasattr(self, 'model'): 
            del self.model
        gc.collect()

"""
class QwenVLPredictor:
    # Qwen2.5-VL (End-to-End Мультимодальная Генерация)
    def __init__(self, model_id=None):
        if model_id is None:
            model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

    def predict_labels(self, img_pil, words, boxes, word_to_block_map, num_blocks):
        if not words: 
            return ["other"] * num_blocks
            
        # Группируем слова EasyOCR по блокам текста
        blocks_text = []
        blocks_data = {i: [] for i in range(num_blocks)}
        for w_text, b_idx in zip(words, word_to_block_map):
            if b_idx < num_blocks: 
                blocks_data[b_idx].append(w_text)
            
        for i in range(num_blocks):
            blocks_text.append(f"#{i}: {' '.join(blocks_data[i])}")
            
        blocks_payload = "\n".join(blocks_text)
        
        # Новый ультра-компактный промпт (экономит токены, защищает от обрыва генерации)
        prompt = (
            f"Analyze this document image. Classify each numbered block below into 'question', 'answer', 'header', or 'other'.\n"
            f"Respond ONLY with the labels separated by commas in a single line, like this: question, answer, other, header.\n"
            f"Do not write any intro, explanations or markdown blocks. The number of labels MUST be exactly {num_blocks}.\n"
            f"Blocks to classify:\n{blocks_payload}"
        )
        
        messages = [{"role": "user", "content": [{"type": "image", "image": img_pil}, {"type": "text", "text": prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text_input], images=img_pil, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Увеличиваем лимит токенов генерации до 1024 для очень больших документов
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
        # ИСПРАВЛЕНО: добавляем [0] перед вызовом .strip()
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        # --- СВЕРХНАДЕЖНЫЙ ПАРСИНГ СТРОКИ С ЗАПЯТЫМИ ---
        clean_text = output_text.replace("\n", "").replace("`", "").replace("json", "").strip()
        
        # Удаляем квадратные скобки, если модель всё-таки попыталась вывести массив
        if clean_text.startswith("[") and clean_text.endswith("]"):
            clean_text = clean_text[1:-1]
            
        # Разбиваем строку по запятым на чистые токены классов
        raw_labels = [lbl.strip().lower().replace('"', '').replace("'", "") for lbl in clean_text.split(",")]
        
        # Фильтруем и валидируем классы по нашей схеме разметки
        valid_classes = {"question", "answer", "header", "other"}
        final_labels = [lbl if lbl in valid_classes else "other" for lbl in raw_labels]
        
        # Выравниваем длину ответа строго под количество блоков EasyOCR
        if len(final_labels) == num_blocks:
            return final_labels
        elif len(final_labels) > num_blocks:
            return final_labels[:num_blocks]
        else:
            # Если токенов не хватило, заполняем оставшиеся как 'other'
            while len(final_labels) < num_blocks:
                final_labels.append("other")
            return final_labels

    def unload(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'processor'): del self.processor
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
"""
     
class LiltRelationPredictor:
    """Подход №5: LiLT (Parallel Layout-Text Transformer)"""
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("MODEL_PATH_LILT", "./models/lilt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fallback_map = {
            "label_0": "other", "0": "other",
            "label_1": "question", "label_2": "question", "1": "question", "2": "question",
            "label_3": "answer", "label_4": "answer", "3": "answer", "4": "answer",
            "label_5": "header", "label_6": "header", "5": "header", "6": "header",
            "label_7": "other", "label_8": "other", "7": "other", "8": "other"
        }
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = LiltForTokenClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        else:
            self.model, self.tokenizer = None, None

    def predict_labels(self, img_pil, words, boxes, word_to_block_map, num_blocks):
        if self.model is None or not words: 
            return ["other"] * num_blocks
        w, h = img_pil.size
        norm_boxes = [
            [int(1000*(x1/w)), int(1000*(y1/h)), int(1000*(x2/w)), int(1000*(y2/h))] 
            for (x1, y1, x2, y2) in boxes
        ]
        encoding = self.tokenizer(words, boxes=norm_boxes, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoding)
        predictions = np.atleast_1d(outputs.logits.argmax(-1).squeeze(0).cpu().numpy()).tolist()
        word_ids = encoding.word_ids()
        
        block_votes = {i: [] for i in range(num_blocks)}
        for pred, word_idx in zip(predictions, word_ids):
            if word_idx is not None and word_idx < len(word_to_block_map):
                raw_label = self.model.config.id2label.get(pred, "other").lower().strip()
                label_name = raw_label.replace("b-", "").replace("i-", "")
                if label_name in self.fallback_map:
                    label_name = self.fallback_map[label_name]
                elif str(pred) in self.fallback_map:
                    label_name = self.fallback_map[str(pred)]
                else:
                    label_name = "other"
                block_votes[word_to_block_map[word_idx]].append(label_name)
        return [max(set(votes), key=votes.count).lower() if votes else "other" for i, votes in block_votes.items()]

    def unload(self):
        """Освобождение VRAM и очистка кэша CUDA при переключении на другую модель"""
        if hasattr(self, 'model'): 
            del self.model
        if hasattr(self, 'tokenizer'): 
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()


