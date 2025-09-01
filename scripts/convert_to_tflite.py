# convert_to_tflite.py

import tensorflow as tf
import os

MODEL_PATH = "models/soil_type_classifier.h5"
TFLITE_MODEL_PATH = "models/soil_type_classifier.tflite"

def convert_model():
    # Load the trained Keras model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded Keras model.")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations for smaller size and faster inference
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model to disk
    os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    convert_model()
