#tömürsüz görseller için annotations oluşturan programım. labellerı boş olduğundan model tömürsüz olarak anlıyor
import os
import json

folder_path = r"C:\Users\goktu\X\EtiketsizGorseller"

annotations = {}

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png")):
        filepath = os.path.join(folder_path, filename)
        size = os.path.getsize(filepath)
        key = f"{filename}{size}"

        annotations[key] = {
            "filename": filename,
            "size": size,
            "regions": [],
            "file_attributes": {}
        }

with open(os.path.join(folder_path, "annotations.json"), "w") as f:
    json.dump(annotations, f, indent=2)

print(f"annotations.json oluşturuldu! ({len(annotations)} görsel)")
