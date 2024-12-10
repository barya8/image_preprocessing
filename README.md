# Image Processing Project

This project performs various image processing tasks, including image preprocessing, face detection, gesture recognition, and skeleton extraction. It uses libraries such as OpenCV, TensorFlow, PIL, and scikit-image to enhance and analyze images.

---

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Folder Structure](#folder-structure)
6. [Acknowledgements](#acknowledgements)

---

## Features

### 1. Image Preprocessing
- Counts files in a directory to analyze dataset structure.
- Enhances images by adjusting brightness, contrast, color, sharpness, and applying gamma correction.
- Adds noise to images and applies mean and median filters to denoise them.
- Performs edge detection using Sobel, Laplacian, and Canny methods.

### 2. Face Detection
- Detects faces using a pre-trained Haar Cascade classifier.
- Draws rectangles around detected faces.
- Saves processed images in a designated folder.

### 3. Gesture Recognition
- Converts images to binary format.
- Extracts skin-colored areas for gesture analysis using HSV color filtering.

### 4. Skeleton Extraction
- Uses a pre-trained model to extract body and hand skeletons.
- Saves skeleton visualizations for further analysis.

---

## Requirements

### Libraries:
- Python 3.7+
- OpenCV
- NumPy
- Pillow
- TensorFlow
- scikit-image
- Matplotlib
- Seaborn
- SciPy

### Pre-trained Models:
- `body_pose_model.pth` (Body pose estimation model)
- `hand_pose_model.pth` (Hand pose estimation model)
- The models are heavy so they are not appear in this repo

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the pre-trained models (`body_pose_model.pth` and `hand_pose_model.pth`) in the project directory.

---

## Usage

### 1. Preprocessing Images
To preprocess images:
- Place your dataset in a folder named `Face Images` in the project directory.
- Run the script:
  ```bash
  python main.py
  ```

### 2. Results
Processed images will be saved in the respective folders:
- **Original and Enhanced Images**: Saved as `originalPhoto.png` and `originalAndProcessedPhoto.png`.
- **Edge Detection Results**: Saved as `edge_detection_output.png`.

### 3. Face Detection
Detected faces are saved in the `face_detected_images` folder.

### 4. Gesture Recognition
Binary and gesture-extracted images are saved in `binary_images` and `gesture_images` folders, respectively.

### 5. Skeleton Extraction
Skeleton-extracted images are saved in the `skeleton_images` folder.

---

## Folder Structure
```
project_directory/
├── Face Images/                # Dataset folder
├── face_detected_images/       # Output: Face detection results
├── binary_images/              # Output: Binary images
├── gesture_images/             # Output: Gesture recognition results
├── skeleton_images/            # Output: Skeleton extraction results
├── body_pose_model.pth         # Pre-trained body pose model
├── hand_pose_model.pth         # Pre-trained hand pose model
├── main.py                     # Main script
├── requirements.txt            # Python dependencies
```

---

## Acknowledgements
- OpenCV for computer vision tools.
- TensorFlow for machine learning integration.
- scikit-image for image analysis.
- PIL for image enhancement.
- Haar Cascade classifier for face detection.

