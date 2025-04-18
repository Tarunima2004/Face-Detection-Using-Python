import cv2
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)
# Create a directory to store the dataset
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ask for the name of the person (or ID)
person_name = input("Enter person name or ID: ")

# Create a subfolder for the person's images
person_folder = os.path.join(dataset_dir, person_name)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Capture faces and store them in the folder
print("Collecting images... Press 'q' to stop.")

face_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_id += 1
        face_img = frame[y:y+h, x:x+w]
        
        # Save the image
        face_filename = os.path.join(person_folder, f"{person_name}_{face_id}.jpg")
        cv2.imwrite(face_filename, face_img)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

