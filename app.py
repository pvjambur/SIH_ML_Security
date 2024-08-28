import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import random
import face_recognition
import csv
import hashlib
from cryptography.fernet import Fernet
from base64 import urlsafe_b64encode, urlsafe_b64decode

# Load the pre-trained model safely
def generate_key():
    key = Fernet.generate_key()
    with open("encryption_key.key", "wb") as key_file:
        key_file.write(key)

generate_key()
print("Encryption key generated and saved as 'encryption_key.key'.")

model_path = r"C:\\Users\\jambu\\anaconda3\\envs\\face_antispoofing\\SIH_ML_Security-main\\Desktop.h5"

def load_model_safely(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_model_safely(model_path)

if model is None:
    st.stop()  # Stop execution if the model could not be loaded

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to detect face and apply anti-spoofing
def detect_face_and_antispoof(frame, model):
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    result = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    predictions = []
    frames_with_boxes = {"Real": [], "Spoof": []}

    if result.detections:
        for detection in result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Ensure the detected face is within frame boundaries
            if x < 0 or y < 0 or x + w > iw or y + h > ih:
                continue

            face = frame[y:y+h, x:x+w]

            # Check if face is not empty
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))  # Resize to model input size
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            # Predict if the face is real or spoofed
            prediction = model.predict(face)[0][0]
            predictions.append(prediction)

            label = "Real" if prediction >= 0.33 else "Spoof"
            color = (0, 255, 0) if label == "Real" else (0, 0, 255)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Store frame with bounding box for later display
            frames_with_boxes[label].append(frame.copy())

    return frame, predictions, frames_with_boxes

# Function to compare faces
def compare_faces(uploaded_image_path, webcam_frame):
    uploaded_image = face_recognition.load_image_file(uploaded_image_path)
    webcam_face_encoding = face_recognition.face_encodings(webcam_frame)

    if len(webcam_face_encoding) > 0:
        uploaded_face_encoding = face_recognition.face_encodings(uploaded_image)

        if len(uploaded_face_encoding) > 0:
            match = face_recognition.compare_faces([uploaded_face_encoding[0]], webcam_face_encoding[0])
            return match[0]
    return False

# Load the encryption key
def load_key():
    return open("encryption_key.key", "rb").read()

encryption_key = load_key()
fernet = Fernet(encryption_key)

# Function to encrypt data
def encrypt_data(data, fernet):
    encrypted_data = fernet.encrypt(data.encode())
    return urlsafe_b64encode(encrypted_data).decode()

# Function to decrypt data
def decrypt_data(encrypted_data, fernet):
    decrypted_data = fernet.decrypt(urlsafe_b64decode(encrypted_data.encode()))
    return decrypted_data.decode()

# Function to hash the image
def hash_image(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    return hashlib.sha256(img_data).hexdigest()

# Function to save the data to CSV
import os

def save_to_csv(aadhaar_number, name, dob, gender, image_hash, csv_path):
    encrypted_aadhaar = encrypt_data(aadhaar_number, fernet)
    encrypted_name = encrypt_data(name, fernet)
    encrypted_dob = encrypt_data(str(dob), fernet)
    encrypted_gender = encrypt_data(gender, fernet)

    # Close any existing file handles to ensure the file can be removed
    try:
        # Ensure the file is not being used before trying to remove it
        with open(csv_path, 'a', newline='') as csvfile:
            pass  # Just open and close the file to release any locks
    except PermissionError:
        st.error(f"Cannot access the file {csv_path}. Please close any programs that might be using it and try again.")
        return

    # Remove the existing CSV file if it exists
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Create a new CSV file and write the data
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Aadhaar Number', 'Name', 'DOB', 'Gender', 'Image Hash']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header and data
        writer.writeheader()
        writer.writerow({
            'Aadhaar Number': encrypted_aadhaar,
            'Name': encrypted_name,
            'DOB': encrypted_dob,
            'Gender': encrypted_gender,
            'Image Hash': image_hash
        })


# Streamlit app
def main():
    st.set_page_config(page_title="Face Detection & Anti-Spoofing", page_icon=":guardsman:", layout="wide")

    # Add sidebar with image and explanation
    st.sidebar.image(r"C:\\Users\\jambu\\anaconda3\\envs\\face_antispoofing\\SIH_ML_Security-main\\aadhar.png", use_column_width=True)
    st.sidebar.title("Face Detection & Anti-Spoofing")
    st.sidebar.write("""
        This app uses a pre-trained deep learning model to detect faces and determine whether they are real or spoofed.
        Click 'Start Video' to begin capturing frames from your webcam, and 'Stop Video' to end the capture.
    """)

    st.title("Real-Time Face Detection & Anti-Spoofing")

    # Video capture controls
    start_button = st.button("Start Video")
    stop_button = st.button("Stop Video")

    stframe = st.empty()  # Placeholder for video frame

    # Initialize session state for video capture and frames
    if 'cap' not in st.session_state:
        st.session_state['cap'] = None
    if 'captured_frames' not in st.session_state:
        st.session_state['captured_frames'] = {"Real": [], "Spoof": []}
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []
    if 'name' not in st.session_state:
        st.session_state['name'] = ""
    if 'aadhaar_number' not in st.session_state:
        st.session_state['aadhaar_number'] = ""
    if 'dob' not in st.session_state:
        st.session_state['dob'] = None
    if 'gender' not in st.session_state:
        st.session_state['gender'] = "Male"
    if 'uploaded_file_path' not in st.session_state:
        st.session_state['uploaded_file_path'] = None
    if 'real_face_detected' not in st.session_state:
        st.session_state['real_face_detected'] = False

    # Handle start button logic
    if start_button:
        if st.session_state['cap'] is None or not st.session_state['cap'].isOpened():
            st.session_state['cap'] = cv2.VideoCapture(0)
            st.session_state['captured_frames'] = {"Real": [], "Spoof": []}  # Reset frames
            st.session_state['predictions'] = []  # Reset predictions

    # Capture video and process frames
    if st.session_state['cap'] is not None and st.session_state['cap'].isOpened():
        while True:
            ret, frame = st.session_state['cap'].read()
            if not ret:
                st.write("No frame detected. Exiting...")
                break

            frame, predictions, frames_with_boxes = detect_face_and_antispoof(frame, model)
            stframe.image(frame, channels="BGR")

            # Store frames based on labels
            if frames_with_boxes["Real"]:
                st.session_state['captured_frames']["Real"].extend(frames_with_boxes["Real"])
            if frames_with_boxes["Spoof"]:
                st.session_state['captured_frames']["Spoof"].extend(frames_with_boxes["Spoof"])

            st.session_state['predictions'].extend(predictions)

            if stop_button:
                st.session_state['cap'].release()
                st.session_state['cap'] = None  # Reset the capture object
                stframe.empty()
                break

    # Handle post-capture logic after clicking "Stop Video"
    # After the video capture section where you handle post-capture logic:
    if stop_button and st.session_state['predictions']:
        avg_prediction = sum(st.session_state['predictions']) / len(st.session_state['predictions'])
        final_label = "Real" if avg_prediction >= 0.8 else "Spoof"

        st.write(f"Final Decision: {final_label} (Average Confidence: {avg_prediction:.2f})")

        # Display three frames based on the final label
        if final_label == "Real" and st.session_state['captured_frames']["Real"]:
            st.image(st.session_state['captured_frames']["Real"][:3], caption=["Real Face 1", "Real Face 2", "Real Face 3"], use_column_width=True)
            st.session_state['real_face_detected'] = True
        elif final_label == "Spoof" and st.session_state['captured_frames']["Spoof"]:
            st.image(st.session_state['captured_frames']["Spoof"][0], caption="Detected as Spoof")
            st.session_state['real_face_detected'] = False

    # Show the form only if a real face is detected
    if st.session_state.get('real_face_detected', False):
        # Input fields for name, Aadhaar number, DOB, and gender
        st.session_state['name'] = st.text_input("Enter Name")
        st.session_state['aadhaar_number'] = st.text_input("Enter Aadhaar Number")
        st.session_state['dob'] = st.date_input("Enter Date of Birth")
        st.session_state['gender'] = st.radio("Select Gender", ('Male', 'Female'))

        # Image upload and face comparison
        uploaded_file = st.file_uploader("Upload Image for Comparison", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            temp_image_path = os.path.join("temp_uploaded_image.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display uploaded image
            st.image(temp_image_path, caption="Uploaded Image for Comparison")
            
            # Compare with the captured frame
            if st.session_state['captured_frames']["Real"]:
                frame_to_compare = st.session_state['captured_frames']["Real"][0]  # Take the first "Real" frame for comparison
                match_found = compare_faces(temp_image_path, frame_to_compare)
                
                # Big notification
                if match_found:
                    st.success("✅ The uploaded image matches the real captured face!")
                else:
                    st.error("❌ The uploaded image does NOT match the real captured face!")

        # Save details to CSV
        csv_path = r"C:\\Users\\jambu\\anaconda3\\envs\\face_antispoofing\\SIH_ML_Security-main\\encrypted_data.csv"
        if uploaded_file is not None:
            image_hash = hash_image(temp_image_path)
            save_to_csv(st.session_state['aadhaar_number'], st.session_state['name'], st.session_state['dob'], st.session_state['gender'], image_hash, csv_path)
            st.success(f"Data saved successfully to {csv_path}.")


if __name__ == "__main__":
    main()
