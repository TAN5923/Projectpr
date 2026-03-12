import os
import json

file_path = "Dictionary.json"

# Kaggle dataset path
load_path ="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"

dic = {"PlantVillage": {}}
for folder in os.listdir(load_path):
    if "___" not in folder:
        continue
    plant, disease = folder.split("___")
    folder_path = os.path.join(load_path, folder)
    if not os.path.isdir(folder_path):
        continue
    images = []
    for img in os.listdir(folder_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(folder_path, img))

    if len(images) == 0:
        continue
    if disease not in dic["PlantVillage"]:
        dic["PlantVillage"][disease] = {}
    dic["PlantVillage"][disease][plant] = sorted(images)


with open(file_path, "w") as f:
    json.dump(dic, f)

print("Dictionary created successfully")
