import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# Load the model
MODEL_PATH = "model.tflite"  # Update if your model has a different name
LABELS_PATH = "labels.txt"

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape (assuming model expects 224x224 images)
input_shape = input_details[0]['shape']
img_size = (input_shape[1], input_shape[2])  # (height, width)

# Folder containing test images
TEST_FOLDER = "test/"

# Process each image in the test folder
for img_name in os.listdir(TEST_FOLDER):
    img_path = os.path.join(TEST_FOLDER, img_name)

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {img_name}, could not read the image.")
        continue

    # Preprocess the image
    img = cv2.resize(img, img_size)  # Resize to model's expected size
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]

    # Print results
    print(f"Image: {img_name} → Prediction: {labels[predicted_index]} (Confidence: {confidence:.2f})")

print("Testing complete!")
