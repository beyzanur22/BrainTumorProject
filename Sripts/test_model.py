from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#.keras formatından yüklüyorm
model = load_model("../beyin_tumor_modeli.keras")

IMG_SIZE = 128

#Test klasörü
unlabeled_folder = r"C:\Users\goktu\BrainTumor\TestImg"

#tahminler
for filename in sorted(os.listdir(unlabeled_folder)):
    if filename.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(unlabeled_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Görsel okunamadı: {filename}")
            continue

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        prediction = model.predict(input_img)[0][0]
        label = "TUMOR" if prediction >= 0.5 else "NO TUMOR"

        print(f"{filename}: {label} ({prediction:.2f})")

        plt.imshow(img, cmap="gray")
        plt.title(f"{filename} → Tahmin: {label}")
        plt.axis("off")
        plt.show()


