import cv2
import numpy as np
import os
import shutil

# --- Configuration ---
RAW_IMAGES_DIR = 'raw_handwritten_images'
OUTPUT_DATASET_DIR = 'custom_language_dataset' # Dedicated folder for your language
IMAGE_SIZE = (28, 28) # Still a good starting point for isolated chars

# Define your custom character IDs (your "Unicode numbers") here.
# These can be actual Unicode code points (e.g., 0xE000 for PUA)
# or just any unique numerical IDs you've assigned to your custom glyphs.
CUSTOM_GLYPH_IDS = [
    # Example: Let's say these are the IDs for your 5 custom characters
    1001, 1002, 1003, 1004, 1005,
    # You can mix them with standard characters if your language includes them
    # '0', '1', 'A', 'B', # Example of including standard chars
]
# Convert these IDs to strings for use as folder names and labels
CLASSES = [str(id) for id in CUSTOM_GLYPH_IDS]

# Create directories if they don't exist
if not os.path.exists(RAW_IMAGES_DIR):
    os.makedirs(RAW_IMAGES_DIR)
    print(f"Created directory: {RAW_IMAGES_DIR}. Please put your raw handwritten image files here.")
    print("Example: a file named 'my_language_chars_1.png' containing a line of your custom chars.")
    print("Script will now exit. Run again after adding images.")
    exit()

# Clear and create output directories for fresh dataset generation
if os.path.exists(OUTPUT_DATASET_DIR):
    shutil.rmtree(OUTPUT_DATASET_DIR)
os.makedirs(OUTPUT_DATASET_DIR)
for class_name in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, class_name), exist_ok=True)

print(f"Output dataset will be saved to: {OUTPUT_DATASET_DIR}")
print(f"Please enter the numerical ID for each detected character. Press 'q' to skip.")
print(f"Expected IDs are: {', '.join(CLASSES)}") # Show the user the expected IDs

# --- Helper Function to Preprocess and Segment an Image (No changes needed) ---
def process_and_segment_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image {image_path}. Skipping.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small noise or very large components (adjust thresholds as needed)
        if w < 10 or h < 10 or w > img.shape[1] * 0.9 or h > img.shape[0] * 0.9:
            continue

        char_roi = thresh[y:y+h, x:x+w]

        max_dim = max(w, h)
        top = (max_dim - h) // 2
        bottom = max_dim - h - top
        left = (max_dim - w) // 2
        right = max_dim - w - left

        padded_char_roi = cv2.copyMakeBorder(char_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        resized_char = cv2.resize(padded_char_roi, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        characters.append(resized_char)

    return characters

# --- Main Dataset Generation Loop ---
image_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
if not image_files:
    print(f"No image files found in {RAW_IMAGES_DIR}. Please add your handwritten images.")
    exit()

total_saved_chars = 0
for image_file in image_files:
    print(f"\nProcessing image: {image_file}")
    full_image_path = os.path.join(RAW_IMAGES_DIR, image_file)
    segmented_characters = process_and_segment_image(full_image_path)

    if not segmented_characters:
        print(f"No characters detected in {image_file}. Skipping.")
        continue

    for i, char_img in enumerate(segmented_characters):
        cv2.imshow(f"Character from {image_file} ({i+1}/{len(segmented_characters)})", char_img)
        file_label = image_file.split("_")
        #label_input = input(f"Enter ID for this character (e.g., {CLASSES[0]}, or 'q' to skip): ").strip()
        label_input = file_label[0]
        if label_input == 'q':
            print("Skipping character.")
            cv2.destroyAllWindows()
            continue
        elif label_input in CLASSES: # Check if the input label is in your defined CLASSES
            save_dir = os.path.join(OUTPUT_DATASET_DIR, label_input)
            os.makedirs(save_dir, exist_ok=True)
            file_count = len(os.listdir(save_dir))
            save_path = os.path.join(save_dir, f"char_{file_count + 1:04d}.png")
            cv2.imwrite(save_path, char_img)
            total_saved_chars += 1
            print(f"Saved character to: {save_path}")
        else:
            print(f"Invalid ID '{label_input}'. ID must be one of {', '.join(CLASSES)}. Skipping character.")

        cv2.destroyAllWindows()

print(f"\nDataset generation complete! Total characters saved: {total_saved_chars}")
print(f"Your dataset is located at: {os.path.abspath(OUTPUT_DATASET_DIR)}")
print("Now you can proceed to the 'train_handwritten_model.py' to train your model.")