# convert_moisture_to_tflite.py

import tensorflow as tf
import os

MODEL_PATH = "models/soil_moisture_classifier.h5"
TFLITE_MODEL_PATH = "models/soil_moisture_classifier.tflite"

def convert_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded soil moisture Keras model.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite soil moisture model saved to {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    convert_model()
