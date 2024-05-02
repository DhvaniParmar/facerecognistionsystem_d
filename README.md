Face Detection and Recognition System

This project implements a face detection and recognition system using OpenCV in Python. It allows you to train a face recognizer on a dataset of images and perform face recognition on images or videos.

 Table of Contents

1. [Installation](#installation)
2. [Code Structure and Functionality](#code-structure-and-functionality)
3. [Usage Instructions](#usage-instructions)
4. [Limitations and Future Improvements](#limitations-and-future-improvements)

 Installation

 Required Libraries

- OpenCV
- NumPy
- Pickle (for saving/loading label mappings)

 Installation Instructions

1. Install OpenCV:
   bash
   pip install opencv-python
   
2. Install NumPy:
   bash
   pip install numpy
   

 Code Structure and Functionality

 Overview
The project consists of two main scripts:
- train_recognizer.py: Script to train the face recognizer using images in the dataset.
- recognize_faces.py: Script to perform face detection and recognition on images or videos using the trained recognizer.

 train_recognizer.py
- Reads images from the dataset folder.
- Preprocesses the images (converts to grayscale, resizes).
- Trains a face recognizer (LBPH Face recognizer).
- Saves the trained recognizer model and label mappings to files.

 recognize_faces.py
- Loads the trained recognizer model and label mappings.
- Utilizes a pre-trained face detection model.
- Detects faces in images or video frames.
- Recognizes faces using the trained recognizer.
- Draws bounding boxes around detected faces and labels them with recognized names.

 Usage Instructions

 Training the Recognizer
1. Place your dataset images in a folder (e.g., Images).
2. Run the following command to train the recognizer:
   bash
   python train_recognizer.py
   

Recognizing Faces
1. Ensure that the recognizer model (recognizer.yml) and label mappings (labels.pickle) are available.
2. Run the following command to perform face recognition on images:
   bash
   python recognize_faces.py --image <path_to_image>
   
3. To perform face recognition on a video, run:
   bash
   python recognize_faces.py --video <path_to_video>
   

 Limitations and Future Improvements

 Limitations
- Performance constraints may arise with large datasets or high-resolution images/videos.
- Accuracy may vary depending on lighting conditions, image quality, and face poses.

 Future Improvements
- Implement more advanced face recognition algorithms for improved accuracy.
- Optimize performance for real-time processing.
- Handle occlusions and variations in facial expressions more effectively.
