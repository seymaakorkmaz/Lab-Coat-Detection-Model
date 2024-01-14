# LAB COAT DETECTION

This project was developed during my summer internship at ASELSAN A.Ş., focusing on addressing an important occupational health and safety concern: identifying employees who do not wear lab coats in the laboratory. The project utilizes the YOLO (You Only Look Once) model for object recognition and classification.

## YOLO (You Only Look Once)

YOLO is a deep learning model specifically designed for object detection. Its distinctive feature is its ability to analyze an image and detect objects in a single pass, making it faster and more efficient compared to traditional object detection methods.

The primary goal of this project is to enhance laboratory safety by automating the identification of individuals not wearing lab coats. Through the use of the YOLO model, the system can perform real-time object detection and classification, contributing to a safer working environment.

## Data Augmentation

[data-augmentation.py](data-augmentation.py) script creates 1000 new instances by applying specified augmentation operations. These new samples are combined with samples obtained from the original data set to create an expanded data set.

The script includes operations such as rotation, flipping, zooming, and contrast adjustment to diversify the dataset.

## Lab Coat Detection

[lab-coat-detection.py](lab-coat-detection.py) script overview:
- Imports the necessary libraries and models.
- Takes camera images and processes these images with the YOLOv5 model.
- Calculates counts for specific labels (e.g., 'person', 'lab-coat', 'logo') using the outputs of the YOLOv5 model.
- Evaluates overlapping areas and security areas between two different groups of objects.
- Plots the results on the screen and indicates safe and unsafe areas.


