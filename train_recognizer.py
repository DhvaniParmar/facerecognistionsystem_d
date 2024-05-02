# import cv2
# import os
# import numpy as np
# import pickle

# # Load the LBPH face recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Directory containing the images
# images_dir = "images/"

# # Initialize lists for faces and labels
# x_train = []
# y_labels = []
# label_ids = {}

# current_id = 0  

# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load images and labels
# for root, dirs, files in os.walk(r'D:\facedetection and recognition\images'):
#     for file in files:
#         if file.endswith("png") or file.endswith("jpg"):
#             path = os.path.join(root, file)
#             label = os.path.basename(root)
            
#             # Convert label to numeric
#             if label not in label_ids:
#                 label_ids[label] = current_id
#                 current_id += 1
#             id_ = label_ids[label]
            
#             # Read and preprocess the image
#             img = cv2.imread(path)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
#             for (x, y, w, h) in faces:
#                 roi_gray = gray[y:y+h, x:x+w]
#                 x_train.append(roi_gray)
#                 y_labels.append(id_)

# # Train the recognizer if there is data
# if len(x_train) > 0:
#     recognizer.train(x_train, np.array(y_labels))

#     # Save the recognizer model
#     recognizer.save("recognizer.yml")

#     # Save label mappings
#     with open("labels.pickle", 'wb') as f:
#         pickle.dump(label_ids, f)
# else:
#     print("No training data found. Make sure there are faces in the images.")
import cv2
import os
import numpy as np
import pickle

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize lists for faces and labels
x_train = []
y_labels = []
label_ids = {}
current_id = 0  

# Load images and labels
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # Convert label to numeric
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            
            # Read and preprocess the image
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize image to a consistent size
            resized_img = cv2.resize(gray, (100, 100))
            x_train.append(resized_img)
            y_labels.append(id_)

# Train the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train, np.array(y_labels))

# Save the recognizer model
recognizer.save("recognizer.yml")

# Save label mappings
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)