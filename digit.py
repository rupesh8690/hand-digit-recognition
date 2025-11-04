import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw
import os

#  Train and Save Model 
model_path = r"D:\MCA\Third-sem\ML Experiment\project\digit_nn_model.h5"
history_path = r"D:\MCA\Third-sem\ML Experiment\project\train_history.npy"

train_accuracy = None
test_accuracy = None

if not os.path.exists(model_path):
    print("Training model... Please wait.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Build Neural Network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(model_path)

    # Save accuracy for future display
    np.save(history_path, history.history, allow_pickle=True)
    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]
    print(f" Model trained successfully.\nTrain Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

else:
    print(" Model already exists. Loading...")
    model = keras.models.load_model(model_path)

    #  load dictionary from npy
    if os.path.exists(history_path):
        hist = np.load(history_path, allow_pickle=True).item()
        train_accuracy = hist.get('accuracy', [0])[-1]
        test_accuracy = hist.get('val_accuracy', [0])[-1]
    else:
        # As a fallback, re-evaluate model on test data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        train_accuracy = test_accuracy  # fallback


#  Tkinter GUI
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer - Neural Network (MNIST)")
        self.root.configure(bg="#e8f0fe")

        # Center window
        window_width = 420
        window_height = 520
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

        # Title Label
        tk.Label(root, text="Handwritten Digit Recognizer", font=("Arial", 18, "bold"),
                 bg="#e8f0fe", fg="#1a73e8").pack(pady=10)

        # Canvas for Drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white",
                                highlightbackground="#1a73e8", highlightthickness=2)
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        btn_frame = tk.Frame(root, bg="#e8f0fe")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Predict Digit", command=self.predict_digit,
                  bg="#34a853", fg="white", font=("Arial", 12), width=15).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas,
                  bg="#ea4335", fg="white", font=("Arial", 12), width=10).grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="Model Info", command=self.show_model_info,
                  bg="#fbbc04", fg="black", font=("Arial", 12), width=12).grid(row=0, column=2, padx=10)

        # Output Label
        self.label = tk.Label(root, text="Draw a digit (0–9) above",
                              font=("Arial", 14, "italic"), bg="#e8f0fe", fg="#333")
        self.label.pack(pady=15)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label.config(text="Draw a digit (0–9) above", fg="#333")

    def predict_digit(self):
        img = self.image.resize((28, 28)).convert("L")
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert color
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.label.config(
            text=f"Predicted Digit: {digit}  (Confidence: {confidence:.2f}%)",
            fg="#1a73e8"
        )

    def show_model_info(self):
        messagebox.showinfo(
            "Model Information",
            f" Model: Neural Network (2 Hidden Layers)\n"
            f" Training Accuracy: {train_accuracy * 100:.2f}%\n"
            f" Testing Accuracy: {test_accuracy * 100:.2f}%\n\n"
            f" Model Path:\n{model_path}"
        )


# Run the App 
root = tk.Tk()
app = DigitApp(root)
root.mainloop()
