import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

#model1'i tümör olup olmadığını tahmin etmek için yükledik
model1 = load_model("../beyin_tumor_modeli.keras")

#model2'yi varsa tümörün tipini sınıflandırmak için yükledik
model2 = load_model("../siniflandiran_model.keras")

#IMG_SIZE görselleri 128x128 boyutuna sabitlemek için kullandık
IMG_SIZE = 128

#class_names listesini model2'nin çıktı sıralamasına göre tanımladık
class_names = ["glioma", "meningioma", "pituitary"]

#test edilecek görsellerin bulunduğu klasörü kendi bilgisayar yolumuza göre belirledik
test_folder = r"C:\Users\goktu\BrainTumor\TestImg"

#klasördeki tüm görselleri alfabetik sırayla gezerek işledik
for filename in sorted(os.listdir(test_folder)):
    if filename.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(test_folder, filename)

        #görseli gri tonlamalı (grayscale) olarak yükledik
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Görsel okunamadı: {filename}")
            continue

        #görseli yeniden boyutlandırdık ve normalize ettik
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        #ilk model ile görselde tümör olup olmadığını tahmin ettik
        tumor_prob = model1.predict(input_img)[0][0]
        print(f"\n🧠 [{filename}] — Tümör Olasılığı: {tumor_prob:.2f}")

        #tümör yoksa ekrana yazdık
        if tumor_prob < 0.5:
            print("🔍 Tahmin: Tümör YOK")
            label = "No Tumor"
        else:
            #tümör varsa ikinci modeli kullanarak tipini sınıflandırdık
            print("🔍 Tahmin: Tümör VAR → Model 2 çalıştırılıyor...")
            type_probs = model2.predict(input_img)[0]
            tumor_type = class_names[np.argmax(type_probs)]
            confidence = np.max(type_probs)
            label = f"{tumor_type.upper()} ({confidence:.2f})"

        #tahmin sonuçlarını görsel ile birlikte ekranda gösterdik
        plt.imshow(img, cmap="gray")
        plt.title(f"{filename} → Tahmin: {label}")
        plt.axis("off")
        plt.show()
