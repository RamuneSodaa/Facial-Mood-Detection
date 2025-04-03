import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# Load model
model_path = 'models/face_model.h5'
model = load_model(model_path, compile=False)

# Emotion classes
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Parameters
adversarial_mode = True
epsilon = 0.01

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FGSM attack
def fgsm_attack(model, img, eps):
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        label = tf.one_hot(tf.argmax(pred, axis=1), depth=7)
        loss = tf.keras.losses.CategoricalCrossentropy()(label, pred)
    grad = tape.gradient(loss, x)
    adv_img = x + eps * tf.sign(grad)
    return tf.clip_by_value(adv_img, 0.0, 1.0).numpy()

# Prediction buffer
buffer_size = 5
pred_buffer = deque(maxlen=buffer_size)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        if adversarial_mode:
            adv_face = fgsm_attack(model, face, epsilon)
            pred = model.predict(adv_face, verbose=0)
        else:
            pred = model.predict(face, verbose=0)

        pred_buffer.append(pred[0])
        avg_pred = np.mean(pred_buffer, axis=0)
        label = class_names[np.argmax(avg_pred)]

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Real-time Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
