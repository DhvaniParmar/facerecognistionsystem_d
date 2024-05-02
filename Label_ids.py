import os
import pickle

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

label_ids = {}
current_id = 0  # Initialize current_id

# Load images and labels
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            label = os.path.basename(root).replace(" ", "-").lower()
            # Convert label to numeric
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

# Save label mappings
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)