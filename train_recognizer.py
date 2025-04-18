import cv2
import os
import numpy as np

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare dataset path
dataset_dir = 'dataset'

# Initialize lists to store images and labels
images = []
labels = []
names = {}

# Read each person's folder in the dataset
label = 0
for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_folder):
        names[label] = person_name  # Map label to person name
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                images.append(face_img)
                labels.append(label)
        label += 1

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)

# Save the trained model
recognizer.save('face_recognizer.yml')

# Save the names of the people
with open('names.txt', 'w') as f:
    for label, name in names.items():
        f.write(f"{label},{name}\n")

print("Training completed and model saved!")
