import cv2
import os
import numpy as np
from deepface import DeepFace

# Load reference embeddings from dataset folder
def load_reference_embeddings(dataset_path="dataset"):
    references = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(filename)[0]
            image_path = os.path.join(dataset_path, filename)
            try:
                embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
                references.append((person_name, embedding))
                print(f"Loaded embedding for {person_name}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    return references

# Compute Euclidean distance
def find_best_match(embedding, references, threshold=10):
    min_dist = float("inf")
    identity = "Unknown"
    for name, ref_embedding in references:
        dist = np.linalg.norm(np.array(ref_embedding) - np.array(embedding))
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > threshold:
        return "Unknown", min_dist
    return identity, min_dist

# Load embeddings
print("Loading known faces...")
reference_embeddings = load_reference_embeddings()
print("Reference loading complete.")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Real-Time Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Face Recognition", 800, 600)

print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            # Save face temporarily to get embedding
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_img)

            embedding = DeepFace.represent(img_path=temp_face_path, model_name="Facenet")[0]["embedding"]
            label, distance = find_best_match(embedding, reference_embeddings)

            # Draw results
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} ({distance:.2f})"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

        except Exception as e:
            print(f"Recognition error: {e}")

    cv2.imshow("Real-Time Face Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting via 'q' key.")
        break
    if cv2.getWindowProperty("Real-Time Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        print("Exiting via window close.")
        break

cap.release()
cv2.destroyAllWindows()

# Clean up temporary files
if os.path.exists("temp_face.jpg"):
    os.remove("temp_face.jpg")
    