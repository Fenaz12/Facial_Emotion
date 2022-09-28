#!/usr/bin/env python
# coding: utf-8

# In[21]:


import flask
import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
import cv2
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('Moddel.h5')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_image(frame):
    """
    Read image using pil
    Convert to numpy array and to grayscale
    Detect face in the image
    Crop the image
    Resize and normalize the image
    """
    face_roi = 0
    face_detected = False
    pil_image = Image.open(io.BytesIO(frame))
    frame = np.array(pil_image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("No Faces Detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]
                face_detected = True
    if face_detected:
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)  # fourth dimension
        return final_image / 255.0  # normalizing
    else:
        return "No Face Detected"


def predict_result(frame):
    return np.argmax(model.predict(frame))


@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'selectedFile' not in request.files:
        return "Please try again. The Image doesn't exist"

    # file = request.files.get('file')
    file = request.files.get('selectedFile')
    if not file:
        return
    # Read the image
    img_bytes = file.read()
    # Prepare the image
    frame = preprocess_image(img_bytes)
    # predict
    if type(frame) == str:
        prediction_result = "No Face Detected"
    else:
        prediction_result = predict_result(frame)
    # Return on a JSON format
    return jsonify(str(prediction_result))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
