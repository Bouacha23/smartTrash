import tensorflow.lite as tflite
from PIL import Image

def load_labels(filename):
    file_dict = {}  
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()  
            
            if len(parts) == 2:
                key = int(parts[0])  
                value = parts[1]  
                file_dict[key] = value  

    return file_dict

filename = "labels.txt" 
labels = load_labels(filename)
print(labels)



# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image manually (without numpy)
def preprocess_image(image_path, input_shape):
    # Open the image and resize it to the input size expected by the model
    image = Image.open(image_path).resize((input_shape[1], input_shape[2]))

    # Convert image to a list of pixel values (manual normalization)
    pixel_values = list(image.getdata())
    pixel_values = [value / 255.0 for value in pixel_values]  # Normalize to [0, 1]

    # Convert to a list of lists (adding batch dimension)
    image_list = [pixel_values]
    
    return image_list

# Run inference
image = preprocess_image("trash85.jpg", input_details[0]['shape'])
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# Get the output tensor and extract the result
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the predicted class manually (without numpy)
predicted_class = 0
max_value = output_data[0][0]  # Initialize with the first value

for i in range(1, len(output_data[0])):
    if output_data[0][i] > max_value:
        max_value = output_data[0][i]
        predicted_class = i

# Assuming labels is a list, print the predicted class
labels = ["plastic", "metal", "paper", "glass", "cardboard"]  # Modify according to your labels
print(f"Predicted class: {labels[predicted_class]}")
