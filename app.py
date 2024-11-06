import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist  # Added for loading MNIST dataset
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Load your dataset and preprocess it here (e.g., MNIST or IAM Handwriting Database)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
# Step 2: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)
# Step 1: Load and preprocess data (MNIST dataset as an example)
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data to match the expected input shape of your model
# In your case, it would be (num_samples, 28, 28, 1) for grayscale images
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoded format
num_classes = 10  # MNIST has 10 classes (digits 0-9)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# Step 3: Build the model
input_shape = (28, 28, 1)  # Adjust this based on your image size
num_classes = 26  # Assuming recognition of capital letters only

# Define the CNN model
input_img = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

# Add LSTM layers for sequence modeling
x = layers.RepeatVector(5)(x)  # Assuming maximum length of text is 5
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64, return_sequences=True)(x)

output = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)

model = Model(input_img, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Model Training
# Train the model using the augmented data

# Step 5: Evaluate the Model
# Evaluate the model on the test set and print the results

# Step 6: Deployment - Recognize handwritten text from an image
def recognize_text(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape[:2], color_mode='grayscale')
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array.reshape(1, *input_shape)
    image_array = image_array / 255.0

    predicted_sequence = model.predict(image_array)
    predicted_labels = np.argmax(predicted_sequence, axis=-1)

    recognized_text = ''
    for label in predicted_labels[0]:
        if label == 0:
            break
        recognized_text += chr(ord('A') + label - 1)  # Convert label to corresponding letter

    return recognized_text

# Example usage:
image_path = 'path_to_your_test_image.png'
recognized_text = recognize_text(image_path)
print("Recognized Text:", recognized_text)

# Step 7: Implement a user-friendly interface (web app, mobile app, or CLI) to interact with the model and recognize text from images.

# Optional Step 8: Fine-tuning and Transfer Learning
# Consider using pre-trained models or fine-tuning on a more extensive dataset for better performance.
