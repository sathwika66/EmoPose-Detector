import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize MediaPipe drawing and pose modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ✅ Load the fixed emotion detection model
classifier = load_model(r"C:\Users\Sathwika Dimmiti\Desktop\sathwika docs\emopose\Emotion_Detection_FIXED_FINAL.h5")
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(
    r"C:\Users\Sathwika Dimmiti\Desktop\sathwika docs\emopose\EmoPose-master\EmoPose-master\haarcascade_frontalface_default.xml"
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose_detection.process(image_rgb)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Detect faces in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict emotion
                preds = classifier.predict(roi, verbose=0)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(image, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Display result
        cv2.imshow('Pose and Emotion Detection', image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
