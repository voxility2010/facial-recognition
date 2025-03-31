import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["facial_recognition_db"]
faces_collection = db["faces"]  # More explicit variable name

def recognize_face():
    """Capture a face from the webcam and check if it exists in the database."""
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("🚨 Error: Could not access the camera. Make sure it's connected.")
        return {"message": "Camera error"}

    print("\n🔍 Searching for known faces. Please look at the camera...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera error! Failed to capture a frame.")
            break

        # Convert frame to RGB format (required by face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            print("😕 No face detected. Adjust your position and try again.")

        for face_encoding in face_encodings:
            all_faces = list(faces_collection.find())  # Convert cursor to list
            if not all_faces:
                print("⚠️ No registered faces found in the database.")
                break

            for person in all_faces:
                known_encoding = np.array(person["encoding"])

                # Compare detected face with stored encodings
                is_match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                
                if is_match:
                    print(f"✅ Recognized: {person['name']} (ID: {person['unique_id']})")

                    cam.release()
                    cv2.destroyAllWindows()

                    return {
                        "name": person["name"],
                        "unique_id": person["unique_id"],
                        "message": "User recognized"
                    }

        # Display the video feed (press 'q' to exit)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Recognition canceled by user.")
            break

    cam.release()
    cv2.destroyAllWindows()
    print("❌ No match found in the database.")
    return {"message": "User not found"}

# Run the face recognition script
if __name__ == "__main__":
    result = recognize_face()
    print(result)  # Useful for debugging or API integration
