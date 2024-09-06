import os
from insightface.app import FaceAnalysis
import numpy as np
import cv2

# Параметры
data_dir = "./data/faces/"
output_dir = "./data/embeddings/"

# Инициализация модели
embedding_model = FaceAnalysis(providers=['CPUExecutionProvider'])
embedding_model.prepare(ctx_id=0, det_size=(640, 640))


def create_embeddings(image_path):
    print(image_path)
    img = cv2.imread(image_path)
    return embedding_model.get(img)

def save_embeddings(output, emb):
    with open(output, 'wb') as emb_file:
        emb_file.write(str(emb).encode('utf-8'))
    print(f"Эмбеддинг для {emb_file} создан и сохранен.")

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sub_dir_name in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, sub_dir_name)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(data_dir, sub_dir_name)
                image_path = os.path.join(image_path, filename)
                emb = create_embeddings(image_path)
                output_file = os.path.join(output_dir, filename + ".pkl")
            
                # Сохранение эмбеддинга
                save_embeddings(output_file, emb)
                

if __name__ == "__main__":
    main()
