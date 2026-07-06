import sys
import train_rf
import train_bert
import train_yolo
import train_layoutlmv3
import train_lilt

def run_global_training_playlist():
    print("==============================")
    print(" Запуск автоскрипта обучения ")
    print("==============================")
    
    # Плейлист последовательного вызова подмодулей
    train_rf.main()
    train_bert.main()
    train_yolo.main()
    train_layoutlmv3.main()
    train_lilt.main()
    
    print("==============================")
    print(" Подготовка моделей завершена ")
    print("==============================")

if __name__ == "__main__":
    run_global_training_playlist()
