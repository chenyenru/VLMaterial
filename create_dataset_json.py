# create_dataset_json.py
import json
import os
from pathlib import Path

img_dir = os.getcwd() / Path('Procedural_Test')
print(f"{img_dir}")
images = sorted([p.name for p in img_dir.glob('*.png')])
images.extend(sorted([p.name for p in img_dir.glob('*.jpg')]))

data = []
for i, img in enumerate(images, start=1):
    data.append({
        "id": f"case_{i:05d}",
        "image": img,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWrite a Python function with Blender API to create a material node graph for this image.",
            }
        ]
    })

with open(img_dir / 'dataset.json', 'w') as f:
    json.dump(data, f, indent=2)