# VLM-Based Tracking & Object Detection Pipeline

An automated computer vision and data extraction pipeline utilizing generative AI to simulate dynamic robotic trajectories and reveal camouflaged objects from static images.

## Project Overview
This repository contains the code for a multi-step visual perception pipeline. The system takes static images, utilizes the Veo 3.1 generative video model to hallucinate motion, and then applies tracking and object detection algorithms to extract physical parameters and identify hidden objects[cite: 2].

## Key Features & Methodology
* **Kinematic Extraction:** Converts video frames to HSV color space, applying threshold masks and morphological operations to isolate targets[cite: 2]. Uses image moments to track centroids and convert pixel displacement into real-world velocity and acceleration matrices[cite: 2].
* **Signal Smoothing:** Implements a Savitzky-Golay filter (order 3 polynomial, window size 25) with zero-padding to process raw extracted kinematics[cite: 2]. This handles frame-to-frame jitter and pixel quantization noise, successfully retaining 99.8% of the original travel distance[cite: 2].
* **Camouflaged Object Detection:** Leverages YOLO11x for frame-by-frame analysis on the generated videos[cite: 2]. Applies a strict 0.7 probability threshold to filter noise, successfully identifying the class and bounding boxes of objects (e.g., dogs, horses, owls) that remain entirely undetectable in the initial static frames[cite: 2].

## Tech Stack
* **Languages:** Python
* **Computer Vision:** OpenCV, YOLO11x[cite: 2]
* **Generative AI:** Veo 3.1[cite: 2]
* **Data Processing:** SciPy (Savitzky-Golay filtering)[cite: 2], NumPy, Matplotlib

## Results
* Successfully extracted stable velocity ($V_x$, $V_y$) and acceleration ($A_x$, $A_y$) profiles for a mobile robot navigating a maze and performing a penalty kick[cite: 2].
* Demonstrated that animating static images via VLM prompts allows standard detection models to identify camouflaged elements with high confidence (>0.90)[cite: 2].
