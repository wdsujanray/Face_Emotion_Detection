from keras.models import load_model
from time import sleep
from keras.utils import img_to_array  # Changed import path
import cv2
import numpy as np

# Optional: Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier(r'D:\A A A A A ADTU\Semister-4\Mini Project\Face_Emotion_Detection\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\A A A A A ADTU\Semister-4\Mini Project\Face_Emotion_Detection\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi, verbose=0)[0]  # Added verbose=0 to reduce output
            label = emotion_labels[prediction.argmax()]
            confidence = prediction.max()  # Get confidence score
            
            # Display label with confidence
            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detector', frame)
    
    # Press 'q' to quit, 's' to save screenshot
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('screenshot.png', frame)
        print("Screenshot saved!")

cap.release()
cv2.destroyAllWindows()