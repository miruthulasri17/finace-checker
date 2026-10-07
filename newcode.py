import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# ==============================
# TEXT TO SPEECH SETUP
# ==============================
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ==============================
# LOAD DATASET
# ==============================
img_size = 128
batch_size = 16

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    'currency_dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

class_names = list(train_data.class_indices.keys())

# ==============================
# MODEL
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train once
model.fit(train_data, epochs=5)

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_size, img_size))
    img = img / 255.0
    img = np.reshape(img, (1, img_size, img_size, 3))

    prediction = model.predict(img)
    index = np.argmax(prediction)
    label = class_names[index]

    current_time = time.time()

    # Speak ONLY if new or after delay
    if label != last_spoken or (current_time - last_time > 3):
        speak(f"{label} rupees")
        last_spoken = label
        last_time = current_time

    # NO TEXT DISPLAY (only camera view)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
