import numpy as np

def build_qa_links(form_items, max_distance=400):
    """
    Прокладывает направленные связи между блоками 'question' и 'answer'.
    Модифицирует поле 'linking' прямо внутри элементов списка form_items.
    """
    questions = [item for item in form_items if item["label"] == "question"]
    answers = [item for item in form_items if item["label"] == "answer"]
    
    def get_center(box):
        return [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]

    for q in questions:
        q_box = q["box"]
        q_center = get_center(q_box)
        
        best_ans = None
        min_dist = float("inf")
        
        for a in answers:
            a_box = a["box"]
            a_center = get_center(a_box)
            
            dx = a_center[0] - q_center[0]
            dy = a_center[1] - q_center[1]
            q_height = q_box[3] - q_box[1]
            
            # Проверка условий: ответ справа в строке или строго под вопросом
            is_right = (dx > 0) and (abs(dy) < q_height * 1.5)
            is_bottom = (dy > 0) and (dx >= -q_height)
            
            if is_right or is_bottom:
                weight = 0.7 if is_right else 1.0
                dist = np.linalg.norm(np.array(q_center) - np.array(a_center)) * weight
                
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_ans = a
                    
        if best_ans:
            q["linking"].append([q["id"], best_ans["id"]])
            best_ans["linking"].append([q["id"], best_ans["id"]])
            
    return form_items
