# **Real-Time Face Detection & Anti-Spoofing Project**

## **Project Overview**

This project is a real-time face detection and anti-spoofing system that uses a pre-trained deep learning model to determine if a detected face is real or spoofed. It employs MediaPipe for face detection and TensorFlow for the classification of real vs. spoofed faces. The app is implemented using Streamlit for an interactive and user-friendly interface, allowing users to start and stop video capture, display analyzed frames, and save the frames locally.

## **Project Structure**

```
.
├── app.py                 # Main Streamlit application
├── Desktop.h5             # Pre-trained deep learning model (change the path to your model)
├── aadhar.png             # Sidebar image (change the path to your image)
└── saved_frames/          # Directory to save captured frames (will be created automatically)
```

## **Setting Up the Environment**

To run this project, you need to set up a Python environment with all necessary dependencies. We recommend using Anaconda to manage your environment.

### **Step 1: Install Anaconda**

1. Download and install Anaconda from the official website: [Anaconda Installation](https://www.anaconda.com/products/distribution#download-section).
2. Follow the installation instructions for your operating system.

### **Step 2: Create a New Conda Environment**

Once Anaconda is installed, open the Anaconda Prompt (or your terminal) and create a new environment:

```bash
conda create -n face_antispoofing python=3.8
```

This command creates a new environment named `face_antispoofing` with Python 3.8 installed.

### **Step 3: Activate the Environment**

Activate the newly created environment:

```bash
conda activate face_antispoofing
```

### **Step 4: Install Dependencies**

Install the required dependencies using `pip`:

1. **Streamlit**: For running the web app.
   ```bash
   pip install streamlit
   ```

2. **OpenCV**: For video capture and image processing.
   ```bash
   pip install opencv-python-headless
   ```

3. **TensorFlow**: For loading and using the pre-trained deep learning model.
   ```bash
   pip install tensorflow
   ```

4. **MediaPipe**: For real-time face detection.
   ```bash
   pip install mediapipe
   ```

5. **NumPy**: For numerical operations on image data.
   ```bash
   pip install numpy
   ```

### **Step 5: Setting Up the Project Files**

1. **Download the Project Files**: Ensure you have all the necessary files in your project directory, including `app.py`, `Desktop.h5`, and `aadhar.png`.
   
2. **Update the Paths**: Open `app.py` in your text editor and update the paths to the model (`Desktop.h5`) and image (`aadhar.png`) based on their locations on your system:
   ```python
   model_path = r"Your\Path\To\Desktop.h5"
   image_path = r"Your\Path\To\aadhar.png"
   ```

   Ensure the paths are correct, or the application will not function properly.

### **Step 6: Running the Streamlit App**

1. **Navigate to the Project Directory**:
   ```bash
   cd path\to\your\project\directory
   ```

2. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

   This command will launch the application in your default web browser. If it doesn’t open automatically, copy the URL provided in the terminal and paste it into your browser.

## **App Functions and Features**

### **1. Start Video Button**
- Clicking this button starts the webcam feed and begins processing the video frames in real-time.
- Faces detected in the frames are classified as either "Real" or "Spoof" based on the pre-trained model's predictions.

### **2. Stop Video Button**
- Clicking this button stops the video feed and processes the accumulated frames.
- The app calculates the average prediction from the captured frames to determine the final classification (Real or Spoof).
- The app displays three frames with bounding boxes around the detected faces, indicating whether the face was real (green) or spoofed (red).
- The displayed frames are saved locally in the `saved_frames` directory.

### **3. Sidebar Information**
- The sidebar includes an image and a brief explanation of the app's functionality.
- Users are prompted to enter their name, Aadhaar number, date of birth, and gender if the face is classified as "Real".

## **Important Notes**

- **Path Updates**: Ensure you update the paths in the `app.py` file for the model and image according to your system.
- **Environment Activation**: Always activate the `face_antispoofing` environment before running the app.
- **Directory Structure**: Ensure the directory structure is as specified above, or modify the paths in the code accordingly.

## **Conclusion**

This project demonstrates how to integrate face detection and anti-spoofing into a real-time application using deep learning and computer vision. The provided setup and instructions will guide you through the process of configuring your environment and running the Streamlit application. Enjoy experimenting with face detection and anti-spoofing in real time!
