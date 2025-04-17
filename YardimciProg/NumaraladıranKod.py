#görselleri numaralndırıp karmaşıklığı azaltmak için yazdığım görsel numaralandırma programı
import os

# Etiketsiz görsellerin bulunduğu klasör
folder_path = r"C:\Users\goktu\X\EtiketsizGorseller"

# Sadece .jpg ve .png dosyaları al, sıralı olsun
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))])

start_index = 652 #istediğin bir baslangic indexti gir

for i, filename in enumerate(image_files):
    ext = os.path.splitext(filename)[1].lower()  # .jpg ya da .png
    new_name = f"{start_index + i}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    if new_name != filename:
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(old_path, new_path)
        print(f"✔️ {filename} → {new_name}")