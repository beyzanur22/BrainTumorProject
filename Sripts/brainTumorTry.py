# Gerekli kütüphaneleri yüklüyoruz
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Görsellerin boyutu
IMG_SIZE = 128

# Etiketleri sayısal değerlere eşlemek için bir sözlük
label_map = {"no_tumor": 0, "tumor": 1}

# JSON anotasyonları okuyup, görselleri ve etiketleri yükleyen fonksiyon
def load_dataset(folder_path):
    X = []  # Görseller
    y = []  # Etiketler

    annotation_path = os.path.join(folder_path, "annotations.json")  # Etiket dosyası
    with open(annotation_path, "r") as f:
        annotations = json.load(f)  # Etiket bilgilerini JSON'dan oku

    print(f"{folder_path} klasöründe toplam görsel: {len(annotations)}")

    for key, value in annotations.items():
        filename = value["filename"]
        regions = value["regions"]

        # Etiket belirleniyor anotasyon yoksa 'no_tumor', varsa 'tumor'
        if len(regions) == 0:
            label = "no_tumor"
        else:
            label = regions[0].get("region_attributes", {}).get("shape", "tumor")
            if label not in label_map:
                label = "tumor"

        label_id = label_map[label]  # Etiketi sayısal değere çeviriyoruz

        # Görsel dosyasını .jpg ya da .png olarak alıyoruz. 
        for ext in [".jpg", ".png"]:
            img_path = os.path.join(folder_path, filename.replace(".jpg", ext).replace(".png", ext))
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Görseli gri tonlamada oku
                break
        else:
            print(f"Atlandı: {filename} (dosya bulunamadı)")
            continue

        if img is None:
            print(f"Atlandı: {filename} (okunamadı)")
            continue

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Görseli sabit boyuta getirieyoruz
        normalized = resized / 255.0  # Piksel değerlerini 0-1 aralığına getiriyoruz

        X.append(normalized)  # Görseli listeye ekle
        y.append(label_id)    # Etiketini listeye ekle

    #veri kümesini numpy dizisine çevirdik
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # 4 boyutlu hale getir (batch, height, width, channel)
    y = np.array(y)
    return X, y

# Ana klasör yolunu belirtiyoruz
base_path = r"C:\Users\goktu\BrainTumor\Datalar\BrainTumorData"

# Eğitim, doğrulama ve test verilerini yüklüyoruz
X_train, y_train = load_dataset(os.path.join(base_path, "train"))
X_val, y_val     = load_dataset(os.path.join(base_path, "val"))
X_test, y_test   = load_dataset(os.path.join(base_path, "test"))

# Veri kümesi boyutlarını yazdırıyoruz
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

# 1. Model: Tümör var mı yok mu? İkili sınıflandıran model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),  # İlk katman
    MaxPooling2D((2, 2)),                                              # Havuzlama
    Conv2D(64, (3, 3), activation='relu'),                             # İkinci katman
    MaxPooling2D((2, 2)),
    Flatten(),                                                        # Veriyi düzleştir
    Dense(64, activation='relu'),                                     # Tam bağlantılı katman
    Dropout(0.5),                                                     # %50 dropout ile overfitting'i azalt
    Dense(1, activation='sigmoid')                                    # Çıkış: 0 ya da 1 (no_tumor vs tumor)
])

# Modelin derlenmesi (loss fonksiyonu ve optimizer belirleniyor)
model.compile(
    optimizer=Adam(learning_rate=0.001),                # Öğrenme oranı düşük seçildi
    loss='binary_crossentropy',                         # İkili sınıflandırma için uygun loss
    metrics=['accuracy']                                # Doğruluk metriği
)

# Modeli eğitiyoruz
history = model.fit(
    X_train, y_train,                      # Eğitim verisi
    validation_data=(X_val, y_val),        # Doğrulama verisi
    epochs=10,                             # Epoch sayısı, denemelerim sonucu en optimal 10 çıktı.
    batch_size=16                          # Mini-batch boyutu
)

# Modeli test verisiyle değerlendiriyoruz
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_acc:.2f}")

# Eğitim süreci doğruluk grafiği
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Eğitilen modeli .keras formatında kaydediyoruz
model.save("../beyin_tumor_modeli.keras")  # Not: .. ifadesiyle üst dizine kaydediyor

print("1. Model .keras formatında başarıyla kaydedildi.")
