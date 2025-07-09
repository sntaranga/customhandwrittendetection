import os
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageTk

import tensorflow as tf
import numpy as np
import cv2
# Removed: from tensorflow.python.keras.utils.np_utils import normalize (not used)

WIDTH,HEIGHT = 280,280
# WHITE = (255,255,255) # Not directly used
BRUSH_SIZE = 12 # Increased for better visibility
BRUSH_COLOR = "black"
BG_COLOR = "white"

MODEL_IMAGE_SIZE = 28

class DrawApp:
    def __init__(self,root):
        self.root = root
        self.root.title("Handwritten Input Palette") # Changed 'pallet' to 'palette'
        self.debug_window = None

        self.prediction_label = tk.Label(root,text="Draw a letter and click to predict",font=("Helvetica",16))
        self.prediction_label.pack(pady=10)

        self.canvas = tk.Canvas(root,width=WIDTH,height=HEIGHT,bg=BG_COLOR,cursor="cross") # Used BG_COLOR
        self.canvas.pack(pady=10)

        # Use 'L' mode for grayscale image, 255 for white background
        self.image = Image.new("L",(WIDTH,HEIGHT),color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Changed to Button-1 for initial click, and strat_draw to start_draw
        self.canvas.bind("<Button-1>",self.start_draw)
        self.canvas.bind("<B1-Motion>",self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        #self.class_names = ["1001","1002"]
        CUSTOM_GLYPH_IDS = [
            # Example: Let's say these are the IDs for your 5 custom characters
            1001, 1002, 1003, 1004, 1005,
            # You can mix them with standard characters if your language includes them
            # '0', '1', 'A', 'B', # Example of including standard chars
        ]
        # Convert these IDs to strings for use as folder names and labels
        CLASSES = [str(id) for id in CUSTOM_GLYPH_IDS]
        self.class_names = CLASSES
        button_frame =  tk.Frame(root)
        button_frame.pack(pady=10)
        tk.Button(button_frame,text="Clear",command=self.clear).pack(side="left",padx=10)
        tk.Button(button_frame, text="Predict", command=self.predict_letter).pack(side="left", padx=10)
        tk.Button(button_frame, text="Save as PNG", command=self.save).pack(side="left", padx=10)
        self.text_box = tk.Text(root, height=1, width=20)
        self.text_box.pack()

        self.last_x = None
        self.last_y = None

        self.model = self.load_model()

        # --- PROCESSED IMAGE DISPLAY in main window ---
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="Model Input (Processed)", font=("Helvetica", 14)).pack(pady=(20, 5))
        self.processed_image_label = tk.Label(main_frame, bg="lightgray")
        self.processed_image_label.pack()
        self.clear() # Initialize the processed image display

    def start_draw(self,event): # Renamed from strat_draw
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self,event):
        self.last_x = None
        self.last_y = None

    def draw_on_canvas(self,event):
        x,y = event.x,event.y
        # r = 8 # This variable was not used in line drawing
        if self.last_x is not None and self.last_y is not None: # Ensure last_x/y are not None
            # Draw on Tkinter canvas with smooth lines
            self.canvas.create_line(self.last_x,self.last_y,x,y,fill=BRUSH_COLOR,width=BRUSH_SIZE,
                                     capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on PIL Image (fill=0 for black in 'L' mode) with round joints
            self.draw.line([self.last_x,self.last_y,x,y],fill=0,width=BRUSH_SIZE, joint="round")

        self.last_x = x
        self.last_y = y

    def clear(self):
        self.canvas.delete("all")
        # Reset the PIL image to a white background (255 for 'L' mode)
        self.draw.rectangle([0,0,WIDTH,HEIGHT],fill=255)

        self.prediction_label.config(text="Draw a letter and click Predict")
        # Clear the processed image label with a black blank image for visual clarity
        blank_img = ImageTk.PhotoImage(Image.new("L", (140, 140),color=0))
        self.processed_image_label.config(image=blank_img)
        self.processed_image_label.image = blank_img
        if self.debug_window:
            self.debug_window.destroy()  # Close debug window on clear
            self.debug_window = None

    def save(self):
        path = "raw_handwritten_images"
        filename_prefix =  self.text_box.get("1.0",tk.END).strip()
        filename = self.get_next_filename(filename_prefix,".png",path)
        self.image.save(f"{path}/{filename}")

        messagebox.showinfo("Save Image", f"Drawing saved as {filename}") # Changed print to messagebox

    def get_next_filename(self,prefix, extension, directory):
        i = 1
        while True:
            filename = f"{prefix}{i}{extension}"
            full_path = os.path.join(directory, filename)
            if not os.path.exists(full_path):
                return filename
            i += 1
    def load_model(self):
        try:
            model = tf.keras.models.load_model("handwritten_custom_language_model.h5")
            print("Model loaded successfully.")
            return model
        except Exception as e:
            # Fixed f-string syntax
            messagebox.showerror("Model Load Error", f"Could not load model: {e}\nPlease ensure 'handwritten_modal' exists.")
            return None

    def predict_letter(self):
        if not self.model:
            messagebox.showwarning("Prediction Error","Model is not loaded.") # Added return
            return

        pil_image = self.image # PIL 'L' mode image (black lines on white background)
        # Convert PIL Image (already grayscale) to NumPy array
        original_gray_np = np.array(pil_image)

        # Thresholding for model input: white digits on black background (common for MNIST)
        # If your model expects black digits on white background, change cv2.THRESH_BINARY_INV to cv2.THRESH_BINARY
        _, thresholded_for_model = cv2.threshold(original_gray_np, 128, 255, cv2.THRESH_BINARY_INV)

        # Resize to the model's input size (28x28)
        resized_image = cv2.resize(thresholded_for_model,(MODEL_IMAGE_SIZE,MODEL_IMAGE_SIZE),interpolation=cv2.INTER_AREA)

        # Normalize pixel values to 0.0-1.0
        normalized_image = resized_image.astype('float32')/255.0

        # Add batch and channel dimensions for the model: (1, height, width, 1)
        image_for_prediction = np.reshape(normalized_image, (1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 1))

        prediction = self.model.predict(image_for_prediction)

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_char = self.class_names[predicted_index]

        result_text = f"Prediction: {predicted_char} ({confidence:.2f}%)"
        self.prediction_label.config(text=result_text)
        print(f"Prediction array: {prediction}")
        print(f"Predicted character: {predicted_char} with confidence {confidence:.2f}%")

        # --- DISPLAY THE PROCESSED IMAGE IN THE MAIN WINDOW ---
        # Convert normalized float image (0.0-1.0) back to 0-255 uint8 for display.
        # Scale up for better visibility (140x140 from 28x28) using nearest neighbor for sharp pixels.
        display_img_28x28_8bit = (normalized_image * 255).astype(np.uint8)
        display_img_scaled = cv2.resize(display_img_28x28_8bit, (140, 140), interpolation=cv2.INTER_NEAREST)

        processed_photo = ImageTk.PhotoImage(image=Image.fromarray(display_img_scaled))
        self.processed_image_label.config(image=processed_photo)
        self.processed_image_label.image = processed_photo # Keep reference

        # Show debug info in a separate window, passing the correct stages
        self._show_debug_info(original_gray_np, thresholded_for_model, normalized_image, image_for_prediction)


    def _show_debug_info(self, original_gray_img, thresholded_img, normalized_img, prediction_img):
        """Creates or updates a debug window to show image processing stages."""
        if not self.debug_window or not self.debug_window.winfo_exists():
            self.debug_window = tk.Toplevel(self.root)
            self.debug_window.title("Debug Inspector")

            # 1. Original Grayscale Image (from PIL)
            self.debug_window.original_frame = tk.Frame(self.debug_window, padx=5, pady=5)
            self.debug_window.original_frame.pack(side=tk.LEFT)
            tk.Label(self.debug_window.original_frame, text="1. Original Grayscale").pack()
            self.debug_window.original_label = tk.Label(self.debug_window.original_frame)
            self.debug_window.original_label.pack()

            # 2. Thresholded Image (input to resizing)
            self.debug_window.thresholded_frame = tk.Frame(self.debug_window, padx=5, pady=5)
            self.debug_window.thresholded_frame.pack(side=tk.LEFT)
            tk.Label(self.debug_window.thresholded_frame, text="2. Thresholded Image").pack() # Corrected label
            self.debug_window.thresholded_label = tk.Label(self.debug_window.thresholded_frame)
            self.debug_window.thresholded_label.pack()

            # 3. Normalized & Resized Image (28x28)
            self.debug_window.normalized_frame = tk.Frame(self.debug_window, padx=5, pady=5)
            self.debug_window.normalized_frame.pack(side=tk.LEFT)
            tk.Label(self.debug_window.normalized_frame, text="3. Normalized & Resized (28x28)").pack() # Corrected label
            self.debug_window.normalized_label = tk.Label(self.debug_window.normalized_frame)
            self.debug_window.normalized_label.pack()

            # 4. Reshaped for Model (Visually same as Normalized & Resized)
            self.debug_window.prediction_frame = tk.Frame(self.debug_window, padx=5, pady=5)
            self.debug_window.prediction_frame.pack(side=tk.LEFT)
            tk.Label(self.debug_window.prediction_frame, text="4. Reshaped for Model").pack() # Corrected label
            self.debug_window.prediction_label_img = tk.Label(self.debug_window.prediction_frame)
            self.debug_window.prediction_label_img.pack()

        # --- Prepare images for display ---

        # 1. Original Grayscale Image (from PIL, full size, resized for display)
        original_resized = cv2.resize(original_gray_img, (140, 140), interpolation=cv2.INTER_AREA)
        original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_resized))
        self.debug_window.original_label.config(image=original_photo)
        self.debug_window.original_label.image = original_photo # Keep reference

        # 2. Thresholded Image (white on black or black on white, depending on THRESH_BINARY_INV/BINARY)
        # This parameter is now correctly receiving 'thresholded_for_model' from predict_letter
        thresh_display_resized = cv2.resize(thresholded_img, (140, 140), interpolation=cv2.INTER_NEAREST)
        thresh_photo = ImageTk.PhotoImage(image=Image.fromarray(thresh_display_resized))
        self.debug_window.thresholded_label.config(image=thresh_photo)
        self.debug_window.thresholded_label.image = thresh_photo  # Keep reference

        # 3. Normalized Image (0.0-1.0 float, 28x28, converted back to 0-255 for display)
        norm_display = (normalized_img * 255).astype(np.uint8)
        norm_resized = cv2.resize(norm_display, (140, 140), interpolation=cv2.INTER_NEAREST)
        norm_photo = ImageTk.PhotoImage(image=Image.fromarray(norm_resized))
        self.debug_window.normalized_label.config(image=norm_photo)
        self.debug_window.normalized_label.image = norm_photo  # Keep reference

        # 4. Image for Prediction (4D array, visually identical to normalized_img after squeeze)
        pred_display_squeezed = prediction_img.squeeze()
        pred_display_8bit = (pred_display_squeezed * 255).astype(np.uint8)
        pred_resized = cv2.resize(pred_display_8bit, (140, 140), interpolation=cv2.INTER_NEAREST)
        pred_photo = ImageTk.PhotoImage(image=Image.fromarray(pred_resized))
        self.debug_window.prediction_label_img.config(image=pred_photo)
        self.debug_window.prediction_label_img.image = pred_photo  # Keep reference

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()