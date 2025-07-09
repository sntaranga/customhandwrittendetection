import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
import pickle # To save the label_encoder

# --- Configuration ---
DATASET_DIR = 'custom_language_dataset' # Matches the output from generation script
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 300 # Adjust as needed based on dataset size and complexity
MODEL_SAVE_PATH = 'handwritten_custom_language_model.h5' # New model name

# --- 1. Load Dataset ---
print(f"Loading dataset from: {DATASET_DIR}")
data = []
labels = []
class_names = sorted(os.listdir(DATASET_DIR))

if not class_names or not os.path.isdir(os.path.join(DATASET_DIR, class_names[0])):
    print(f"Error: Dataset directory '{DATASET_DIR}' is empty or not correctly structured.")
    print("Please ensure you have run 'generate_handwritten_dataset.py' successfully and labeled data.")
    exit()

class_names = [name for name in class_names if os.path.isdir(os.path.join(DATASET_DIR, name))]

for class_name in class_names: # class_name will be your string ID (e.g., '1001')
    class_path = os.path.join(DATASET_DIR, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert('L') # 'L' for grayscale
            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=-1)

            data.append(img_array)
            labels.append(class_name) # Store the string ID as the label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

if not data:
    print("No images loaded. Please check your dataset directory and image files.")
    exit()

data = np.array(data)
labels = np.array(labels)

print(f"Loaded {len(data)} images.")
print(f"Classes found in dataset (your custom IDs): {class_names}")

# Encode labels to numerical format. LabelEncoder will map your string IDs to 0, 1, 2...
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
one_hot_labels = to_categorical(integer_encoded_labels, num_classes=num_classes)

print(f"Labels encoded to {num_classes} classes.")
print("Mapping of your custom IDs to internal integer labels:")
for i, custom_id_string in enumerate(label_encoder.classes_):
    print(f"  '{custom_id_string}' (your ID) -> integer label {i}")


# --- 2. Split Dataset ---
X_train, X_temp, y_train, y_temp = train_test_split(data, one_hot_labels, test_size=0.2, random_state=42, stratify=one_hot_labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# --- 3. Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)

# --- 4. Build the CNN Model ---
# Model might need to be deeper/wider for many complex classes
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SIZE + (1,)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model ---
print("\nStarting model training...")
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    verbose=1)

# --- 6. Evaluate the Model ---
print("\nEvaluating model on the test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# --- 7. Save the Model and Label Encoder ---
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("LabelEncoder saved to 'label_encoder.pkl'")


# --- Optional: Visualize Training History ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# --- Optional: Prediction example ---
from tensorflow.keras.models import load_model

def predict_custom_char(model_path, encoder_path, image_path, target_size=(28, 28)):
    model = load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    try:
        img = Image.open(image_path).convert('L') # Grayscale
        img_array = np.array(img).astype('float32') / 255.0

        # Preprocess the new image similarly to how training images were processed
        # For simplicity, if your 'new_image_path' is already a single char image:
        img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)

        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_id_string = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence = np.max(predictions[0]) * 100

        print(f"\nPrediction for {os.path.basename(image_path)}:")
        print(f"Predicted ID: {predicted_id_string} (Confidence: {confidence:.2f}%)")
        return predicted_id_string, confidence
    except Exception as e:
        print(f"Error predicting on image {image_path}: {e}")
        return None, None

# Example usage (after training and saving the model):
# You would need a new image file for this test
# new_image_to_predict_path = 'path/to/your/new_custom_char_image.png'
# predict_custom_char(MODEL_SAVE_PATH, 'label_encoder.pkl', new_image_to_predict_path)