import cv2
import face_recognition
import numpy as np
import random
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["facial_recognition_db"]
faces_collection = db["faces"]  # Renaming for clarity

def generate_unique_id():
    """Generate a 5-digit unique ID that isn't already in the database."""
    while True:
        new_id = random.randint(10000, 99999)
        if not faces_collection.find_one({"unique_id": new_id}):  
            return new_id  # Return only after ensuring uniqueness

def register_face():
    """Captures a face using webcam and stores it in MongoDB if new."""

    name = input("📸 Enter your name: ").strip()
    if not name:
        print("⚠️ Name cannot be empty. Try again!")
        return
    
    unique_id = generate_unique_id()
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("🚨 Error: Camera not detected! Make sure it's connected.")
        return

    print("\n📷 Look directly at the camera. Face detection will start...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Failed to capture frame. Try restarting the camera.")
            break

        cv2.imshow("Register Face (Press 'q' to cancel)", frame)  # Show webcam feed

        # Convert frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = face_recognition.face_locations(rgb_frame)

        if detected_faces:
            print("✅ Face detected! Processing...")
            try:
                face_encoding = face_recognition.face_encodings(rgb_frame, detected_faces)[0]
                
                # Save face data to MongoDB
                face_entry = {
                    "name": name,
                    "unique_id": unique_id,
                    "encoding": face_encoding.tolist()  # MongoDB requires lists, not NumPy arrays
                }
                faces_collection.insert_one(face_entry)
                print(f"🎉 Success! Face registered for {name} (ID: {unique_id})")
                break  # Exit loop after successful registration
            except IndexError:
                print("⚠️ Unexpected error in encoding face. Try repositioning yourself.")
                continue  # Retry capturing the face

        # Allow the user to quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Face registration canceled.")
            break

    # Clean up resources
    cam.release()
    cv2.destroyAllWindows()

# Run the function
register_face()
