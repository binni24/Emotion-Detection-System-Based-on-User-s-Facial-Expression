# import cv2
# import numpy as np
# from keras.models import model_from_json


# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # load json and create model
# json_file = open('D:/project/facial-emtion-recognition-model-main/model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # load weights into new model
# emotion_model.load_weights("model/emotion_model.keras")
# print("Loaded model from disk")

# # start the webcam feed
# cap = cv2.VideoCapture(0)

# # pass video path here 
# #cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))
#     if not ret:
#         break
#     face_detector = cv2.CascadeClassifier('cascadeClassifier/haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     # take each face available on the camera and Preprocess it
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     cv2.imshow('Emotion Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import os
import cv2
import numpy as np
from keras.models import model_from_json

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# File paths
model_json_path = r'D:/project/facial-emtion-recognition-model-main/model/emotion_model.json'
model_weights_path = r'D:/project/facial-emtion-recognition-model-main/model/emotion_model.keras'
cascade_path = r'D:/project/facial-emtion-recognition-model-main/cascadeClassifier/haarcascade_frontalface_default.xml'

# Ensure model JSON file exists
if not os.path.exists(model_json_path):
    raise FileNotFoundError(f"Model JSON file not found at: {model_json_path}")

# Ensure model weights file exists
if not os.path.exists(model_weights_path):
    raise FileNotFoundError(f"Model weights file not found at: {model_weights_path}")

# Ensure Haar cascade file exists
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at: {cascade_path}")

# Load model structure
with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Create model from JSON
emotion_model = model_from_json(loaded_model_json)

# Load model weights
emotion_model.load_weights(model_weights_path)
print("Loaded model from disk")

# Initialize Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cascade_path)

# Start the webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not access the webcam. Please check your camera connection.")

print("Press 'q' to exit the webcam feed.")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Resize frame for better display
    try:
        frame = cv2.resize(frame, (1280, 720))
    except Exception as e:
        print(f"Error resizing frame: {e}")
        continue

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)

        # Extract ROI for emotion prediction
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Display predicted emotion
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the video feed with emotion detection
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
