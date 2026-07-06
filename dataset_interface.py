from dotenv import load_dotenv
load_dotenv()

import os
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

BIO_LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "B-HEADER", "I-HEADER", "B-OTHER", "I-OTHER"]
BIO_MAP = {lbl: i for i, lbl in enumerate(BIO_LABELS)}

class DocumentInterpretationDataset(Dataset):
    def __init__(self, mode, tokenizer_or_processor, max_length=512, augment=False, model_type="bert"):
        self.mode = mode
        self.transform_type = model_type
        self.tokenizer = tokenizer_or_processor
        self.max_length = max_length
        self.augment = augment
        
        if mode == "train":
            self.img_dir = os.getenv("FUNSD_TRAIN_IMG", "./FUNSD_SPLIT/train/images")
            self.json_dir = os.getenv("FUNSD_TRAIN_JSN", "./FUNSD_SPLIT/train/annotations")
        elif mode == "val":
            self.img_dir = os.getenv("FUNSD_VAL_IMG", "./FUNSD_SPLIT/images")
            self.json_dir = os.getenv("FUNSD_VAL_JSN", "./FUNSD_SPLIT/annotations")
        elif mode == "test":
            self.img_dir = os.getenv("FUNSD_TEST_IMG", "./FUNSD_SPLIT/test/images")
            self.json_dir = os.getenv("FUNSD_TEST_JSN", "./FUNSD_SPLIT/test/annotations")
        else:
            raise ValueError(f"Неизвестный режим датасета: {mode}")
            
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.json_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        with open(os.path.join(self.json_dir, f"{name}.json"), "r", encoding="utf-8") as f:
            doc = json.load(f)
        image = Image.open(os.path.join(self.img_dir, f"{name}.png")).convert("RGB")
        W, H = image.size
        
        words, boxes, labels = [], [], []
        for item in doc.get("form", []):
            lbl = item["label"].upper()
            for w in item.get("words", []):
                box = w["box"]
                
                # Нормализация координат 0-1000
                nx1 = int(1000 * (box[0] / W))
                ny1 = int(1000 * (box[1] / H))
                nx2 = int(1000 * (box[2] / W))
                ny2 = int(1000 * (box[3] / H))
                
                words.append(w["text"])
                boxes.append([max(0, min(1000, nx1)), max(0, min(1000, ny1)), max(0, min(1000, nx2)), max(0, min(1000, ny2))])
                labels.append(lbl)
                
        if not words: 
            words, boxes, labels = ["empty"], [[0,0,0,0]], ["OTHER"]
        
        # --- БЛОК АУГМЕНТАЦИИ (BBox Jittering & Scaling) ---
        # Срабатывает только для train_loader, где augment=True
        if self.augment:
            augmented_boxes = []
            for box in boxes:
                # Если коробка пустая или нулевая, пропускаем ее без изменений
                if box == [0, 0, 0, 0] or not box:
                    augmented_boxes.append(box)
                    continue
                
                # Случайный сдвиг (jitter) всей рамки в пределах ±3 единиц сетки
                shift_x = random.randint(-3, 3)
                shift_y = random.randint(-3, 3)
                
                # Случайное изменение размеров (scaling) в пределах ±2 единиц сетки
                scale_w = random.randint(-2, 2)
                scale_h = random.randint(-2, 2)
                
                # Применяем трансформации с жестким ограничением в диапазоне [0, 1000]
                anx1 = max(0, min(1000, box[0] + shift_x))
                any1 = max(0, min(1000, box[1] + shift_y))
                anx2 = max(0, min(1000, box[2] + shift_x + scale_w))
                any2 = max(0, min(1000, box[3] + shift_y + scale_h))
                
                # Защитный механизм: левый верхний угол не должен стать больше правого нижнего
                if anx1 > anx2: anx1, anx2 = anx2, anx1
                if any1 > any2: any1, any2 = any2, any1
                
                augmented_boxes.append([anx1, any1, anx2, any2])
            boxes = augmented_boxes
        # --------------------------------------------------

        
        # Токенизация в зависимости от типа модели
        if self.transform_type == "bert":
            encoding = self.tokenizer(words, is_split_into_words=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        elif self.transform_type == "layoutlmv3":
            encoding = self.tokenizer(image, words, boxes=boxes, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        elif self.transform_type == "lilt":
            encoding = self.tokenizer(words, is_split_into_words=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            
        word_ids = encoding.word_ids()
        input_labels, input_boxes = [], []
        last_word_idx = None
        
        for w_idx in word_ids:
            if w_idx is None:
                input_labels.append(-100)
                if self.transform_type == "lilt": input_boxes.append([0, 0, 0, 0])
            else:
                if self.transform_type == "lilt": input_boxes.append(boxes[w_idx])
                if w_idx != last_word_idx:
                    input_labels.append(BIO_MAP.get(f"B-{labels[w_idx]}", BIO_MAP["B-OTHER"]))
                    last_word_idx = w_idx
                else:
                    input_labels.append(BIO_MAP.get(f"I-{labels[w_idx]}", BIO_MAP["I-OTHER"]))
                    
        item_dict = {k: v.squeeze(0) for k, v in encoding.items()}
        item_dict["labels"] = torch.tensor(input_labels, dtype=torch.long)
        if self.transform_type == "lilt":
            item_dict["bbox"] = torch.tensor(input_boxes, dtype=torch.long)
        return item_dict

def get_document_data_loader(mode, model_type, tokenizer_or_processor, batch_size, max_length=512, augment=False):
    dataset = DocumentInterpretationDataset(mode, tokenizer_or_processor, max_length, augment, model_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))
