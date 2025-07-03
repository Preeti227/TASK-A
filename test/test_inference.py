import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.filters import apply_all_filters
from src.model import build_cnn_model
from tensorflow.keras.models import load_model

model = build_cnn_model()
model.load_weights("best_model_weights.h5")  # Save best weights during training

img_path = input("\nEnter image path: ").strip()
img = cv2.imread(img_path)
if img is None:
    print(f"Cannot read image at {img_path}")
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = apply_all_filters(img)
    img_input = np.expand_dims(processed, axis=0)
    pred = model.predict(img_input)[0][0]
    label = "Female" if pred < 0.4 else "Male"
    confidence = (1 - pred) * 100 if label == "Female" else pred * 100
    print(f"\nPredicted Gender: {label} (Confidence: {confidence:.2f}%)")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{label} ({confidence:.2f}%)", fontsize=16, color='blue')
    plt.show()
