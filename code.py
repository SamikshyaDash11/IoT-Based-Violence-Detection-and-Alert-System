#model_train_and_save:-

import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    LSTM, Dense, Dropout, TimeDistributed
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import os
import cv2
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    TimeDistributed, LSTM, Dense, Dropout
)


IMG_SIZE = 64
FRAMES_PER_VIDEO = 10
TRAIN_DIR = "videos/train"
TEST_DIR = "videos/test"


import os
print(os.getcwd())

def load_video_paths(base_dir):
    video_paths = []
    labels = []

    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        for video in os.listdir(label_path):
            video_paths.append(os.path.join(label_path, video))
            labels.append(label)

    return video_paths, labels


train_videos, train_labels = load_video_paths(TRAIN_DIR)
test_videos, test_labels = load_video_paths(TEST_DIR)


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // FRAMES_PER_VIDEO, 1)

    count = 0
    while len(frames) < FRAMES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
        count += step

    cap.release()

        # 🔴 CRITICAL SAFETY CHECK
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(frames[-1])

    return np.array(frames)


def build_feature_array(video_list):
    data = []
    for video in tqdm(video_list):
        data.append(extract_frames(video))
    return np.array(data, dtype=np.float32)


# Example: process 500 videos at a time
X_train_part1 = build_feature_array(train_videos[:500])
X_train_part2 = build_feature_array(train_videos[500:1000])
# and so on
X_train = np.concatenate([X_train_part1, X_train_part2], axis=0)


X_test_part1 = build_feature_array(test_videos[:500])
X_test_part2 = build_feature_array(test_videos[500:1000])
X_test = np.concatenate([X_test_part1, X_test_part2], axis=0)


le = LabelEncoder()
le.fit(train_labels)

y_train = le.transform(train_labels)
y_test = le.transform(test_labels)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)



model = Sequential()

model.add(TimeDistributed(
    Conv2D(32, (3,3), activation='relu'),
    input_shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)
))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Flatten()))

model.add(LSTM(128))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



y_train = y_train[::2]
y_test  = y_test[::2]




print("X_train samples:", X_train.shape[0])
print("y_train samples:", y_train.shape[0])

print("X_test samples :", X_test.shape[0])
print("y_test samples :", y_test.shape[0])
print("Train match:", X_train.shape[0] == y_train.shape[0])
print("Test match :", X_test.shape[0] == y_test.shape[0])



model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test, y_test)
)



# Evaluate on test set (loss and accuracy)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")



model.save("violence_detection_model.h5")
print("Model saved as violence_detection_model.h5")




def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)

   
    prediction = model.predict(frames)[0]

    label = "Violence" if np.argmax(prediction) == 1 else "Non-Violence"
    confidence = float(np.max(prediction))

    return label, confidence



#test_model:-

try:
    label, score = predict_video("videos/test/Violence/V_94.mp4")
    print("Prediction:", label)
    print("Confidence:", score)
except Exception as e:
    print("ERROR:", e)
print(y_train[0])






#converting_model_h5_to_onnx:-
import tensorflow as tf
import tf2onnx

# Load your Keras model
model = tf.keras.models.load_model("violence_detection_model.h5")

# ONNX output path
onnx_path = "violence_detection_model.onnx"

# Convert the model
spec = (tf.TensorSpec(model.input.shape, tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
with open(onnx_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print("Conversion complete! ONNX saved at:", onnx_path)






#full_main_code_in_rpi_to_run_model:-
import cv2

import numpy as np

import requests

import time

import onnxruntime as ort



# ---------------- CONFIG ----------------

MODEL_PATH = "violence_detection_model.onnx"

IMG_SIZE = 64



CONFIDENCE_THRESHOLD = 0.90

ALERT_COOLDOWN = 300   # 5 minutes (300 seconds)



BOT_TOKEN = "8019384845:AAE19msl7_BR7ybM_ADUCZ_0vAI2wJ7El10"

CHAT_ID = "1627232522"



SEQUENCE_LENGTH = 10

frame_buffer = []



# ---------------- LOAD ONNX MODEL ----------------

session = ort.InferenceSession(

    MODEL_PATH,

    providers=["CPUExecutionProvider"]

)



input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name



print("ONNX model loaded successfully")



# ---------------- TELEGRAM ALERT ----------------

def send_telegram_alert(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    params = {

        "chat_id": CHAT_ID,

        "text": message

    }

    try:

        response = requests.get(url, params=params, timeout=10)

        print("Telegram alert sent:", response.status_code)

    except Exception as e:

        print("Telegram error:", e)



# ---------------- CAMERA SETUP ----------------

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)



cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FPS, 30)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)



time.sleep(2)



if not cap.isOpened():

    print("Camera error")

    exit()



# ---------------- ALERT TIMERS ----------------

last_violence_alert_time = 0

last_non_violence_alert_time = 0



print("Violence detection system running...")



# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()

    if not ret:

        continue



    # --- Preprocessing ---

    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    frame_normalized = frame_resized.astype(np.float32) / 255.0



    # --- Frame Buffer ---

    frame_buffer.append(frame_normalized)

    if len(frame_buffer) > SEQUENCE_LENGTH:

        frame_buffer.pop(0)



    if len(frame_buffer) < SEQUENCE_LENGTH:

        continue



    input_data = np.array(frame_buffer, dtype=np.float32)

    input_data = np.expand_dims(input_data, axis=0)



    # --- ONNX Inference ---

    prediction = session.run(

        [output_name],

        {input_name: input_data}

    )[0]



    non_violence_prob = float(prediction[0][0])

    violence_prob = float(prediction[0][1])



    label = "Violence" if violence_prob > non_violence_prob else "Non-Violence"

    confidence = max(violence_prob, non_violence_prob)



    # --- Overlay ---

    cv2.putText(

        frame,

        f"{label}: {confidence:.2f}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 0, 255) if label == "Violence" else (0, 255, 0),

        2

    )



    # --- Alert Logic ---

    now = time.time()



    # Violence alert

    if (

        violence_prob >= CONFIDENCE_THRESHOLD and

        (now - last_violence_alert_time) >= ALERT_COOLDOWN

    ):

        send_telegram_alert(

            "?? ALERT: Violence detected (>90% confidence)"

        )

        last_violence_alert_time = now



    # Non-violence alert

    if (

        non_violence_prob >= CONFIDENCE_THRESHOLD and

        (now - last_non_violence_alert_time) >= ALERT_COOLDOWN

    ):

        send_telegram_alert(

            "? STATUS: Non-Violence detected (>90% confidence)"

        )

        last_non_violence_alert_time = now



    cv2.imshow("Violence Detection", frame)



    if cv2.waitKey(1) & 0xFF == ord("q"):

        break



# ---------------- CLEANUP ----------------

cap.release()

cv2.destroyAllWindows()

