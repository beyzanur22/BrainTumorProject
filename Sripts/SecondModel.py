#Gerekli kütüphaneleri içe aktardık
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#Model parametrelerini tanımladık
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

#Klasör yollarını kendi dosya yapımıza göre ayarladık
base_path = r"C:\Users\goktu\TumorTespitProjesi\Two_Stage_Classification"
train_dir = os.path.join(base_path, "Training")
val_dir = os.path.join(base_path, "Validation")
test_dir = os.path.join(base_path, "Testing")

#Eğitim seti için veri artırma işlemlerini tanımladık
augmentation_layer = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.1)
])

#Veri setlerini klasörlerden yükledik
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

#Sınıf isimlerini yazdırarak sıralamayı kontrol ettik
class_names = train_ds.class_names
print("Modelin sınıf sıralaması:", class_names)

#Veri setlerini performans için ön işledik
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augmentation_layer(x), y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

#Model mimarisini oluşturduk, bu asamada sınıflandırıyor
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

#Modeli derledik
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Erken durdurma ve model kontrol callback'lerini ayarladık
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_ckpt = ModelCheckpoint("best_siniflandiran_model.keras", monitor='val_loss', save_best_only=True)

#Modeli eğittik
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, model_ckpt]
)

#Modeli kaydettik
model.save("siniflandiran_model.keras")
print("2. Model başarıyla .keras formatında kaydedildi.")

#Eğitim ve doğrulama doğruluklarını çizdik
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.title("Tümör Tipi Sınıflandırma (3 Sınıf)")
plt.legend()
plt.show()