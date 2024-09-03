import cv2
import numpy as np
from PIL import Image
import os

# Path to the extracted dataset directory
dataset_path = "C:/dataset"

# Initialize face recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to retrieve images and their corresponding labels
def imgs_and_labels(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                image_paths.append(os.path.join(root, file))

    faces = []
    ids = []
    name_to_id = {}
    current_id = 0

    for imagePath in image_paths:
        img = Image.open(imagePath).convert('L')  # Convert to grayscale
        img_np = np.array(img, 'uint8')
        dir_name = os.path.split(os.path.dirname(imagePath))[-1]

        if dir_name not in name_to_id:
            name_to_id[dir_name] = current_id
            current_id += 1

        id = name_to_id[dir_name]

        faces_detected = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y + h, x:x + w])
            ids.append(id)

    print(f"Total faces detected: {len(faces)}")
    print(f"Total IDs collected: {len(ids)}")
    return faces, ids, name_to_id

# Directory to save the trained model and name-to-ID mapping
save_path = "C:/model_output"

# Ensure the directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Training the recognizer with images and labels
faces, ids, name_to_id = imgs_and_labels(dataset_path)
if len(faces) > 0 and len(ids) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.save(os.path.join(save_path, 'trainer.yml'))
    # Save the name-to-ID mapping
    with open(os.path.join(save_path, 'names.txt'), 'w') as f:
        for name, id in name_to_id.items():
            f.write(f"{id}:{name}\n")
    print(f"Model and names saved to: {save_path}")
else:
    print("No training data available. Please check your dataset.")
