from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import numpy as np
import random
from pymongo import MongoClient

# Flask App Initialization
app = Flask(__name__, template_folder="C:\\Users\\acer\\OneDrive\\Desktop\\facial_recognition\\template")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["facial_recognition_db"]
faces_collection = db["faces"]  # Renamed for clarity

# OpenCV Camera Initialization
camera = cv2.VideoCapture(0)

def check_existing_face(new_encoding):
    """Compare the given face encoding with database entries to check for duplicates."""
    
    # Fetch all stored faces from MongoDB
    stored_faces = faces_collection.find()
    
    for person in stored_faces:
        stored_encoding = np.array(person["encoding"])
        
        # Face recognition distance metric
        similarity_score = face_recognition.face_distance([stored_encoding], new_encoding)[0]

        # Human brain isn't perfect, so let's give some margin for error
        if similarity_score < 0.5:  # This threshold might need tweaking
            return person["name"], person["unique_id"]

    return None, None  # No match found

@app.route('/')
def home():
    """Render the homepage (simple UI to interact with face recognition system)."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream live video with face detection overlay."""
    
    def generate_frames():
        while True:
            ret, frame = camera.read()
            if not ret:
                break  # If the frame isn't read properly, just exit the loop

            # Convert to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = face_recognition.face_locations(rgb_frame)

            # Draw bounding boxes around detected faces
            for (top, right, bottom, left) in detected_faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # Blue box for faces

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """Capture an image from the webcam and attempt to register the detected face."""
    
    data = request.get_json()
    user_name = data.get("name", "Anonymous")  # Default to "Anonymous" if no name is provided

    ret, frame = camera.read()
    if not ret:
        return jsonify({"message": "⚠️ Camera error, please try again."}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        return jsonify({"message": "😕 No face detected. Try again with better lighting."}), 400

    # Get face encodings (assuming only one face is present)
    try:
        new_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    except IndexError:
        return jsonify({"message": "🚨 Error encoding face. Try repositioning yourself."}), 400

    # Check if this face is already in the database
    existing_name, existing_id = check_existing_face(new_encoding)
    if existing_name:
        return jsonify({"message": f"⚠️ This face is already registered as {existing_name} (ID: {existing_id})"}), 400

    # Register new face with a randomly assigned ID
    unique_id = random.randint(10000, 99999)  # Could be improved, but keeping it simple
    face_entry = {
        "name": user_name,
        "unique_id": unique_id,
        "encoding": new_encoding.tolist()  # MongoDB can't store NumPy arrays directly
    }
    faces_collection.insert_one(face_entry)

    return jsonify({"message": "✅ Face registered successfully!", "unique_id": unique_id})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """Attempt to recognize a face from a live camera feed."""
    
    ret, frame = camera.read()
    if not ret:
        return jsonify({"message": "⚠️ Camera issue, please retry."}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        return jsonify({"message": "😕 No recognizable face detected."})

    # Check each detected face
    for encoding in face_encodings:
        existing_name, existing_id = check_existing_face(encoding)
        if existing_name:
            return jsonify({"message": f"✅ Recognized: {existing_name} (ID: {existing_id})"})

    return jsonify({"message": "❌ No match found in the database."})

if __name__ == '__main__':
    # Debug mode is ON, for development purposes only!
    app.run(debug=True)
