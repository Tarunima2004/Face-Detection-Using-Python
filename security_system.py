import cv2
import pygame
import numpy as np

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = "alarm.wav"

# Load the trained face recognizer and names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')

# Load names from the file
names = {}
with open('names.txt', 'r') as f:
    for line in f:
        label, name = line.strip().split(',')
        names[int(label)] = name

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Start the face recognition system
print("Security System Initialized. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the name or "Unknown" if not recognized
        if confidence < 100:
            name = names[label]
            cv2.putText(frame, f"{name} - {round(100 - confidence, 2)}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            name = "Unknown"
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # If "Unknown" person is detected, trigger alarm
        if name == "Unknown" and not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(alarm_sound)
            pygame.mixer.music.play(-1)  # Play alarm in loop
        elif name != "Unknown" and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()  # Stop alarm if recognized

    # Show the frame
    cv2.imshow("Face Recognition Security System", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
