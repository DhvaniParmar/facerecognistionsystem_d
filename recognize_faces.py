# import cv2
# import os
# import numpy as np
# import pickle

# # Function to resize and prepare training data
# def prepare_training_data(data_folder, target_size=(100, 100)):
#     faces = []
#     labels = []
#     label_ids = {}
#     current_id = 0

#     for root, dirs, files in os.walk(data_folder):
#         for subdir in dirs:
#             label = subdir.replace(" ", "-").lower()
#             label_ids[label] = current_id
#             current_id += 1
#             for filename in os.listdir(os.path.join(root, subdir)):
#                 path = os.path.join(root, subdir, filename)
#                 if filename.endswith("png") or filename.endswith("jpg"):
#                     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#                     if img is not None:
#                         # Resize the image to the target size
#                         img = cv2.resize(img, target_size)
#                         faces.append(img)
#                         labels.append(label_ids[label])

#     return faces, labels, label_ids

# # Prepare training data
# faces, labels, label_ids = prepare_training_data(r'D:\facedetection and recognition\Images')

# # Check if the training data is empty
# if len(faces) == 0:
#     print("Error: No training data found.")
#     exit()

# # Train the face recognizer (using Fisher Face recognizer)
# recognizer = cv2.face.FisherFaceRecognizer_create()
# recognizer.train(faces, np.array(labels))

# # Save the trained recognizer model
# recognizer.save("recognizer.yml")

# # Save label mappings
# with open("labels.pickle", 'wb') as f:
#     pickle.dump(label_ids, f)

# print("Training completed successfully.")
import cv2
import pickle

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer.yml')

# Load labels
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Recognize faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45: # Confidence threshold
            name = labels[id_]
        else:
            name = "Unknown"
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the output
    cv2.imshow('Face Recognition', frame)
    
    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()