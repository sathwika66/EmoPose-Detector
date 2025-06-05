# 🤖 EmoPose-Detector

A real-time **Pose and Emotion Detection** system that combines MediaPipe for human pose tracking and a Convolutional Neural Network (CNN) for facial emotion classification. Built using **OpenCV**, **TensorFlow/Keras**, and **MediaPipe**, this project is perfect for computer vision enthusiasts and developers working on human-computer interaction (HCI), assistive technology, or smart surveillance systems.

---

## 📸 Features

- 🎭 Real-time facial emotion detection (`Angry`, `Happy`, `Neutral`, `Sad`, `Surprise`)
- 🕺 Pose detection using **MediaPipe**
- 📷 Integrated live webcam capture using OpenCV
- ✅ Custom-trained `.h5` emotion model
- 🧠 Lightweight and easy to run on most systems

---

## 🗂️ Folder Structure

EmoPose-Detector/
- │
- ├── models/ # Pre-trained emotion detection model (.h5)
- │ └── Emotion_Detection_FIXED_FINAL.h5
- │
- ├── haarcascades/ # Haar Cascade XML for face detection
- │ └── haarcascade_frontalface_default.xml
- │
- ├── src/ # Source code
- │ └── main.py
- │
- ├── requirements/ 
- │ ├── requirements.txt
- │
- └── README.md

## 🚀 Getting Started

### 💻 Setup Environment (Anaconda Recommended)
- using pip:
- pip install -r requirements/requirements.txt

### 🧠 Run the Project
Make sure your webcam is connected. Then:

- python src/main.py
- Press q to quit the webcam window.

### 📥 Download Model Files
- Before running, ensure that:
- Emotion_Detection_FIXED_FINAL.h5 is placed in models/
- haarcascade_frontalface_default.xml is placed in haarcascades/

These files are large and may need to be manually added to the GitHub repo or shared via cloud storage.

### 🔍 Emotion Classes
The model detects 5 emotions:

Angry 😠

Happy 😊

Neutral 😐

Sad 😢

Surprise 😲


### 🙋‍♀️ Author
Sathwika Dimmiti
Computer Vision Enthusiast | AI/ML Developer
📫 Reach me at: www.linkedin.com/in/sathwika-dimmiti-462005211
