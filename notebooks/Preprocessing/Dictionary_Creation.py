import os
import json

file_path = "Dictionary.json"
load_path = r"C:\Users\sonal\Projectpr\dataset\raw\plantvillage"

dic = {"PlantVillage": {}}

# 🔥 first loop (color, grayscale, segmented)
for category in os.listdir(load_path):
    category_path = os.path.join(load_path, category)

    if not os.path.isdir(category_path):
        continue

    print("\nCategory:", category)

    # 🔥 second loop (actual disease folders)
    for folder in os.listdir(category_path):
        if "___" not in folder:
            continue

        plant, disease = folder.split("___")
        folder_path = os.path.join(category_path, folder)

        if not os.path.isdir(folder_path):
            continue

        images = []
        for img in os.listdir(folder_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(folder_path, img))

        print("Processing:", folder, "| Images:", len(images))

        if len(images) == 0:
            continue

        # 🔥 include category level
        if category not in dic["PlantVillage"]:
            dic["PlantVillage"][category] = {}

        if disease not in dic["PlantVillage"][category]:
            dic["PlantVillage"][category][disease] = {}

        dic["PlantVillage"][category][disease][plant] = sorted(images)

# save
with open(file_path, "w") as f:
    json.dump(dic, f, indent=4)

print("\n✅ Dictionary created successfully")