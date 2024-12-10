import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL import ImageEnhance
from skimage.io import imread
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.ndimage import uniform_filter, median_filter
import matplotlib.pyplot as plt
import copy
import os, random, pathlib, warnings, itertools, math

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix

from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
import cv2 as cv
from src import body
from src import util
from src.body import Body
from src.hand import Hand

# =====================================================================
# 2. Image Preprocessing - Understanding Data Patterns
# =====================================================================
def count_files(rootdir):
    """
    Counts and prints the number of files in each subfolder within a given directory.

    Args:
        rootdir (str): Root directory to count files in.
    """
    for path in pathlib.Path(rootdir).iterdir():
        if path.is_dir():
            print("There are " + str(len([name for name in os.listdir(path) \
                                          if os.path.isfile(os.path.join(path, name))])) + " files in " + \
                  str(path.name))

def adjust_gamma(image_array, gamma):
    """
    Apply gamma correction to an image array and returns it as a PIL Image.

    Args:
        image_array (np.ndarray): The image array to adjust gamma for.
        gamma (float): The gamma correction factor.

    Returns:
        PIL.Image: Gamma corrected image.
    """
    image_array = np.power(image_array, gamma)  # Apply gamma correction
    image_array=np.clip(image_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array)

def denoise_image(image):
    """
    Adds Gaussian noise to an image and applies mean and median filters to denoise it.

    Args:
        image (PIL.Image): Image to add noise and denoise.

    Returns:
        tuple: Noisy image, mean-filtered image, median-filtered image.
    """
    image_array = np.array(image) / 255.0  # Convert to float in [0, 1]
    noisy_img = random_noise(image_array, mode='gaussian', var=0.02)  # Add Gaussian noise
    denoised_mean_img = uniform_filter(image_array, size=5)
    denoised_median_img = median_filter(image_array, size=5)
    return noisy_img, denoised_mean_img, denoised_median_img

def preprocess():
    """
    Preprocesses a random image from the dataset by applying enhancements, gamma correction, and denoising.
    Shows and saves results at various processing stages.
    """
    for i in range(number_of_images):
        folder = dataset
        jpg_files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
        a = random.choice(jpg_files)

        image = Image.open(os.path.join(folder, a)).convert('RGB')
        # Display original image
        image_duplicate = image.copy()
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title(label='Orignal', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(image)
        plt.savefig('originalPhoto')
        #plt.show()

        # Apply enhancements to the image
        image1 = ImageEnhance.Color(image_duplicate).enhance(1.35)
        image1 = ImageEnhance.Brightness(image1).enhance(1.25)
        image1 = ImageEnhance.Contrast(image1).enhance(1.45)
        image1 = ImageEnhance.Sharpness(image1).enhance(2.5)
        image1 = image1.convert('RGB')

        # Apply gamma correction and denoising
        image_array = np.array(image1) / 255.0  # Normalize to [0, 1]
        image1_corrected = adjust_gamma(image_array, 1.2)  # Adjust gamma
        noisy_img, denoised_mean_img, denoised_median_img = denoise_image(image1_corrected)

        # Display processed image
        plt.subplot(1, 2, 2)
        plt.title(label='Processed', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(image1)
        plt.savefig('originalAndProcessedPhoto')
        plt.show()

        # Display denoised & noised images
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title(label='Denoised Mean', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(denoised_mean_img)

        plt.subplot(1, 3, 2)
        plt.title(label='Noisy', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(noisy_img)

        plt.subplot(1, 3, 3)
        plt.title(label='Denoised Median', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(denoised_median_img)
        plt.savefig('originalAndProcessedPhotoWithNoise')
        plt.show()

        # Convert to grayscale for edge detection
        noisy_img_gray = cv.cvtColor((noisy_img * 255).astype(np.uint8), cv.COLOR_RGB2GRAY)

        # Edge Detection (Sobel and Laplacian)
        sobelx = cv.Sobel(src=noisy_img_gray, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)  # Sobel X
        sobely = cv.Sobel(src=noisy_img_gray, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Y
        edges_laplacian = cv.Laplacian(noisy_img_gray, cv.CV_64F, ksize=3)
        edges_laplacian = np.uint8(np.absolute(edges_laplacian))
        edges_canny = cv.Canny(noisy_img_gray, 100, 200)

        # Display Sobel X edge detection
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title(label='Sobel X', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(sobelx, cmap='gray')

        # Display Laplacian edge detection
        plt.subplot(1, 3, 2)
        plt.title(label='Laplacian', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(edges_laplacian, cmap='gray')
        #plt.show()

        # Display Canny edge detection
        plt.subplot(1, 3 , 3)
        plt.title(label='Canny', size=15, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(edges_canny, cmap='gray')
        plt.savefig('edge_detection_output')
        plt.show()

# =====================================================================
# 3. Face Detection and Database Creation
# =====================================================================

def detect_face(image, scaleFactor, minNeighbors, minSize):
    """
    Detects faces in an image and draws rectangles around them.

    Args:
        image (np.ndarray): Image to detect faces in.
        scaleFactor (float): Scale factor for image resizing.
        minNeighbors (int): Minimum number of neighbors each rectangle should have.
        minSize (tuple): Minimum size for a detected face.

    Returns:
        np.ndarray: Image with rectangles drawn around detected faces.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(image_gray,
                      scaleFactor=scaleFactor,
                      minNeighbors=minNeighbors,
                      minSize=minSize)
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 255, 0), 3)
    return image

# =====================================================================
# 4. Gesture Detection and Database Creation
# =====================================================================

def convert_to_binary(image_path, output_path, threshold_value=127):
    """
    Converts an image to binary (black and white) and saves it.

    Args:
        image_path (str): Path to the original image.
        output_path (str): Path to save the binary image.
        threshold_value (int): Threshold value for binarization.
    """
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, binary_image)


def extract_gesture_area(image_path, output_path):
    """
    Extracts and saves the skin-colored area in an image for gesture analysis.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the gesture-extracted image.
    """
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([10, 30, 120], dtype=np.uint8)
    upper_skin = np.array([50, 110, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    gesture_area = cv2.bitwise_and(image, image, mask=skin_mask)

    gray_gesture_area = cv2.cvtColor(gesture_area, cv2.COLOR_BGR2GRAY)
    _, binary_gesture_area = cv2.threshold(gray_gesture_area, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, binary_gesture_area)
    print(f"Gesture area saved at: {output_path}")

# =====================================================================
# 5. Skeleton Extraction and Database Creation
# =====================================================================

def extract_skeleton(image_path, output_path):
    """
    Extracts body and hand skeletons from an image and saves the output.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the skeleton-extracted image.
    """
    oriImg = cv2.imread(image_path)
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # Detect hands and draw hand poses
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    cv2.imwrite(output_path, canvas)
    print(f"Skeleton area saved at: {output_path}")

# =====================================================================
#                           MAIN EXECUTION FLOW
# =====================================================================

if __name__ == "__main__":
    K.clear_session()

    # Load the pre-trained Haar Cascade classifier for face detection
    fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_estimation = Body('body_pose_model.pth')
    hand_estimation = Hand('hand_pose_model.pth')

    # Set up directories and process images
    dataset = os.path.join(os.getcwd(), 'Face Images')
    count_files(dataset)
    number_of_images = 1  # Number of images to display

    preprocess()

    face_detected_folder = os.path.join(dataset, "face_detected_images")
    binary_folder = os.path.join(dataset, "binary_images")
    gesture_folder = os.path.join(dataset, "gesture_images")
    skeleton_folder = os.path.join(dataset, "skeleton_images")

    for folder in [face_detected_folder, binary_folder, gesture_folder, skeleton_folder]:
        os.makedirs(folder, exist_ok=True)

    for filename in os.listdir(dataset):
        image_path = os.path.join(dataset, filename)
        if os.path.isfile(image_path):
            image = np.array(Image.open(image_path).convert('RGB'))
            processed_image = detect_face(image=image
                                          , scaleFactor=1.9,
                                          minNeighbors=3,
                                          minSize=(30, 30))

            face_detected_path = os.path.join(face_detected_folder, filename)
            cv2.imwrite(face_detected_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            binary_path = os.path.join(binary_folder, filename)
            convert_to_binary(face_detected_path, binary_path)
            gesture_path=os.path.join(gesture_folder,filename)
            extract_gesture_area(face_detected_path, gesture_path)
            skeleton_path = os.path.join(skeleton_folder, filename)
            extract_skeleton(face_detected_path, skeleton_path)