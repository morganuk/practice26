from dotenv import load_dotenv
load_dotenv()

import os
import json
import streamlit as st
from PIL import Image

from document_pipeline import process_document, draw_results, draw_ground_truth
from predictors import BertPredictor, LayoutLMv3Predictor, RandomForestPredictor, YoloPredictor, LiltRelationPredictor

st.set_page_config(layout="wide", page_title="Document AI Benchmarking Platform")

# Список моделей
MODELS_CONFIG = {
    "Random Forest (Classic Baseline)": RandomForestPredictor,
    "BERT (Pure NLP)": BertPredictor,
    "YOLOv8 (Vision-Only Detection)": YoloPredictor,
    "LayoutLMv3 (Multimodal Transformer)": LayoutLMv3Predictor,
    "LiLT (Parallel Layout-Text Transformer)": LiltRelationPredictor
}

# Управление моделями в сессии Streamlit для предотвращения утечек VRAM/RAM
if "current_model_name" not in st.session_state:
    st.session_state.current_model_name = None
if "loaded_model_instance" not in st.session_state:
    st.session_state.loaded_model_instance = None

def load_and_manage_model(selected_model_name):
    if st.session_state.current_model_name != selected_model_name:
        if st.session_state.loaded_model_instance is not None:
            with st.spinner("Освобождение ресурсов предыдущей модели..."):
                st.session_state.loaded_model_instance.unload()
        
        with st.spinner(f"Загрузка весов для {selected_model_name}..."):
            st.session_state.loaded_model_instance = MODELS_CONFIG[selected_model_name]()
            st.session_state.current_model_name = selected_model_name
            
    return st.session_state.loaded_model_instance

st.title("Document AI Testing Platform")

# Панель управления в сайдбаре
st.sidebar.header("Настройки инференса")
selected_approach = st.sidebar.selectbox("Выберите архитектуру нейросети:", list(MODELS_CONFIG.keys()))
active_predictor = load_and_manage_model(selected_approach)

data_source = st.sidebar.radio("Источник документа:", ["Выбрать из датасета (FUNSD)", "Загрузить файл вручную"])

img_path_to_process = None
gt_json_path = None
original_json_data = None

DATASET_IMG_DIR = os.path.normpath(os.getenv("FUNSD_TEST_IMG", "./FUNSD_SPLIT/test/images"))
DATASET_JSON_DIR = os.path.normpath(os.getenv("FUNSD_TEST_JSN", "./FUNSD_SPLIT/test/annotations"))

if data_source == "Загрузить файл вручную":
    uploaded_file = st.file_uploader("Перетащите скан документа сюда", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        temp_upload_path = "temp_uploaded_document.png"
        image = Image.open(uploaded_file).convert("RGB")
        image.save(temp_upload_path)
        img_path_to_process = temp_upload_path
else:
    if os.path.exists(DATASET_IMG_DIR):
        available_images = sorted([f for f in os.listdir(DATASET_IMG_DIR) if f.endswith(('.png', '.jpg'))])
        if available_images:
            selected_file = st.selectbox("Выберите документ из тестовой выборки:", available_images)
            if selected_file:
                img_path_to_process = os.path.join(DATASET_IMG_DIR, selected_file)
                root_name = os.path.splitext(selected_file)[0]
                gt_json_path = os.path.join(DATASET_JSON_DIR, f"{root_name}.json")
                if os.path.exists(gt_json_path):
                    with open(gt_json_path, "r", encoding="utf-8") as f:
                        original_json_data = json.load(f)
        else:
            st.warning(f"В папке `{DATASET_IMG_DIR}` отсутствуют изображения.")
    else:
        st.error(f"Директория тестового датасета `{DATASET_IMG_DIR}` не найдена. Проверьте конфигурацию переменных окружения.")

# --- ЗАПУСК АНАЛИЗА И ИНФЕРЕНСА ---
if img_path_to_process:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходный документ")
        if data_source == "Выбрать из датасета (FUNSD)" and gt_json_path and os.path.exists(gt_json_path):
            source_img_vis = draw_ground_truth(img_path_to_process, gt_json_path)
            st.image(source_img_vis, use_container_width=True)
        else:
            st.image(Image.open(img_path_to_process), use_container_width=True)
            
    with col2:
        st.subheader(selected_approach)
        temp_inference_path = "temp_inference_run.png"
        img_tmp = Image.open(img_path_to_process).convert("RGB")
        img_tmp.save(temp_inference_path)
        
        with st.spinner("EasyOCR + Анализ структуры..."):
            predicted_form_items = process_document(temp_inference_path, active_predictor)
            
        if predicted_form_items:
            annotated_output = draw_results(temp_inference_path, predicted_form_items)
            st.image(annotated_output, use_container_width=True)
        else:
            st.warning("Текст или структура не были распознаны моделью.")
        
        if os.path.exists(temp_inference_path):
            os.remove(temp_inference_path)

    # --- СРАВНИТЕЛЬНЫЙ ВЫВОД СТРУКТУРЫ JSON ---
    st.markdown("---")
    j_col1, j_col2 = st.columns(2)
    
    with j_col1:
        st.subheader("Исходный JSON (Ground Truth)")
        if original_json_data:
            st.json(original_json_data)
        else:
            st.info("Разметка недоступна для пользовательских (загруженных вручную) файлов.")
            
    with j_col2:
        st.subheader("Сгенерированный JSON модели")
        if img_path_to_process and predicted_form_items:
            st.download_button(
                label="Скачать результат инференса",
                data=json.dumps({"form": predicted_form_items}, ensure_ascii=False, indent=4),
                file_name=f"output_{selected_approach.lower().replace(' ', '_')}.json",
                mime="application/json"
            )
            st.json(predicted_form_items)
