import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import random

# Load the pre-trained model safely
model_path = r"C:\Users\jambu\anaconda3\envs\streamlit_env\SIH_ML_Security-main\Desktop.h5"

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

# Streamlit app
def main():
    st.set_page_config(page_title="Face Detection & Anti-Spoofing", page_icon=":guardsman:", layout="wide")

    # Add sidebar with image and explanation
    st.sidebar.image(r"C:\Users\jambu\anaconda3\envs\streamlit_env\SIH_ML_Security-main\aadhar.png", use_column_width=True)
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

    # Handle stop button logic and display results
    if stop_button and st.session_state['cap'] is None:
        stframe.empty()

        # Calculate the average prediction and determine the overall result
        if st.session_state['predictions']:
            avg_prediction = sum(st.session_state['predictions']) / len(st.session_state['predictions'])
            final_label = "Real" if avg_prediction >= 0.33 else "Spoof"
            st.write(f"Final Prediction: {final_label}")

            # Display 3 frames with bounding boxes for the predicted label
            display_label = "Real" if final_label == "Real" else "Spoof"
            selected_frames = random.sample(st.session_state['captured_frames'][display_label], min(3, len(st.session_state['captured_frames'][display_label])))

            st.write(f"Displaying 3 frames where the face was detected as {display_label}:")
            for i, img in enumerate(selected_frames):
                st.image(img, channels="BGR", caption=f"{display_label} Frame {i+1}")

            # Save the displayed frames locally
            for i, img in enumerate(selected_frames):
                save_path = os.path.join("saved_frames", f"{display_label}_frame_{i+1}.png")
                if not os.path.exists("saved_frames"):
                    os.makedirs("saved_frames")
                cv2.imwrite(save_path, img)
                st.write(f"Saved frame to: {save_path}")

            # If the result is Real, ask for additional details
            if final_label == "Real":
                st.write("Please enter the following details:")
                name = st.text_input("Name")
                aadhaar_number = st.text_input("Aadhaar Number")
                dob = st.date_input("Date of Birth")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])

                if st.button("Submit"):
                    st.write(f"Details Submitted:\nName: {name}\nAadhaar Number: {aadhaar_number}\nDate of Birth: {dob}\nGender: {gender}")

if __name__ == "__main__":
    main()
