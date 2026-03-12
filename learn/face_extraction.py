from deepface import DeepFace
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def face_extraction(member_path):  

    face_name = []

    for img_name in os.listdir(member_path):
        full_path = os.path.join(member_path, img_name)
        if os.path.isfile(full_path):
            face_name.append(img_name)

    print(face_name)

    backend = ['opencv']
    model = ['VGG-Face', "Facenet"]

    print("Extracting embeddings...")
    face_encodings = []
    face_image = []
    for file in face_name:
        img_path = member_path + "/" + file
        print(img_path)
        embendding_obj = DeepFace.represent(
            img_path=img_path,
            model_name=model[0],
            detector_backend=backend[0],
            enforce_detection=True
        )
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_image.append(image)
        face_encodings.append(embendding_obj[0]['embedding'])
    return np.array(face_encodings), face_image, face_name

def query_extraction(member_path):  
    backend = ['opencv']
    model = ['VGG-Face', "Facenet"]

    print("Query embeddings...")
    face_encodings = []
    face_image = []
    img_path = member_path
    print(img_path)
    embendding_obj = DeepFace.represent(
        img_path=img_path,
        model_name=model[0],
        detector_backend=backend[0],
        enforce_detection=True
    )
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_image = image
    face_encodings.append(embendding_obj[0]['embedding'])
    return np.array(face_encodings), face_image,
    
    

def display_result(face_query, face_image,  ind, dist):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1) #แสดงภาพคิวรีในตำแหน่งที่ 1 ของกริด
    plt.imshow(face_query)
    plt.title("Query Image")
    plt.axis('off')

    for i in range(3): #แสดงภาพที่ตรงกันในตำแหน่งที่ 2, 3, และ 4 ของกริด
        idx = ind[0][i]
        plt.subplot(1, 4, i + 2)
        plt.imshow(face_image[idx])
        plt.title(f"Rank {i+1} \nDist: {dist[0][i]:.4f}")
    plt.show()
    print("display images")
    