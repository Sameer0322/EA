import cv2
import numpy as np
import dlib
from keras.models import model_from_json
from imutils import face_utils
import pymongo

# Load emotion recognition model
def load_emotion_model():
    f_json = open(r"C:\Users\saxen\OneDrive\Desktop\Emotion Analyzer\Test\test\emotiondetector.json", "r")
    m_json = f_json.read()
    f_json.close()
    model = model_from_json(m_json)
    model.load_weights(r"C:\Users\saxen\OneDrive\Desktop\Emotion Analyzer\Test\test\emotiondetector.h5")
    return model

# Feature extraction for emotion recognition
def feature_extraction(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to compute distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to determine eye state (sleepy, drowsy, active)
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 'Active'
    elif 0.21 < ratio <= 0.25:
        return 'Drowsy'
    else:
        return 'Sleepy'

# Load face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\saxen\OneDrive\Desktop\Emotion Analyzer\Test\test\shape_predictor_68_face_landmarks.dat")

# Load emotion recognition model
emotion_model = load_emotion_model()

client = pymongo.MongoClient("mongodb://localhost:27017")

# Create or use a database
db = client["EA"]

# Create or use a collection within the database
collection = db["data"]
# Create video capture object
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    all_landmarks = []
    all_statuses = []

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        all_landmarks.extend(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        status = ''
        if left_blink == 'Sleepy' or right_blink == 'Sleepy':
            status = 'Sleepy'
            color = (255, 0, 0)
        elif left_blink == 'Drowsy' or right_blink == 'Drowsy':
            status = 'Drowsy'
            color = (0, 0, 255)
        else:
            status = 'Active'
            color = (0, 255, 0)

        all_statuses.append((status, color, (x1, y1, x2, y2)))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ensure face_region is not empty before processing it
        if y2 > y1 and x2 > x1:
            face_region = frame[y1:y2, x1:x2]
            if not face_region.size == 0:
                image = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (48, 48))
                img = feature_extraction(image)
                pred = emotion_model.predict(img)
                emotion_label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][pred.argmax()]

                # Display emotion and eye state on the frame
                cv2.putText(frame, f"Emotion: {emotion_label}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Eye State: {status}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                combined_info = {
                    'emotion': emotion_label,
                    'eye_state': status,
                    # Add other relevant details
                }
                collection.insert_one(combined_info)
                # Handle the combined information (e.g., store in a database, update attendance)
                print(combined_info)  # Placeholder for your action with the combined information

    landmarks_frame = frame.copy()
    for (x, y) in all_landmarks:
        cv2.circle(landmarks_frame, (x, y), 1, (255, 255, 255), -1)
    cv2.imshow("Facial Landmarks", landmarks_frame)

    status_frame = frame.copy()
    for status, color, (x1, y1, x2, y2) in all_statuses:
        cv2.rectangle(status_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(status_frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Status of Faces", status_frame)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
