import os
import sys
import cv2
import numpy as np
from skimage.util import random_noise
from PIL import Image

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)
    return noisy

def apply_gaussian_smoothing(image, kernel_size=(5, 5), sigma_x=0):
    # Apply Gaussian smoothing (blurring)
    smoothed = cv2.GaussianBlur(image, kernel_size, sigma_x)
    return smoothed

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=salt_prob + pepper_prob)
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def apply_median_filter(image, kernel_size=5):
    # Apply median filtering
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered

def add_speckle_noise(image):
    noisy = random_noise(image, mode='speckle')
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def add_poisson_noise(image):
    noisy = random_noise(image, mode='poisson')
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def add_pixelation_noise(image, pixelation_level=128):
    height, width = image.shape[:2]
    temp = cv2.resize(image, (pixelation_level, pixelation_level), interpolation=cv2.INTER_LINEAR)
    noisy = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return noisy

def apply_random_affine(image):
    rows, cols, ch = image.shape

    # Define random points for affine transformation
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([
        [50 + np.random.randint(-20, 20), 50 + np.random.randint(-20, 20)],
        [200 + np.random.randint(-20, 20), 50 + np.random.randint(-20, 20)],
        [50 + np.random.randint(-20, 20), 200 + np.random.randint(-20, 20)],
    ])

    # Get affine transformation matrix
    M = cv2.getAffineTransform(pts1, pts2)

    # Apply affine transformation
    affine_transformed = cv2.warpAffine(image, M, (cols, rows))
    return affine_transformed

def save_image(output_folder, filename, image):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

def apply_noise_to_images(input_folder):
    output_folders = {
        "gaussian": os.path.join(input_folder, "gaussian_noise"),
        "gaussian_smoothed": os.path.join(input_folder, "gaussian_smoothed"),
        "salt_and_pepper": os.path.join(input_folder, "salt_and_pepper_noise"),
        "salt_and_pepper_median_filtered": os.path.join(input_folder, "salt_and_pepper_median_filtered"),
        "speckle": os.path.join(input_folder, "speckle_noise"),
        "poisson": os.path.join(input_folder, "poisson_noise"),
        "pixelation": os.path.join(input_folder, "pixelation_noise"),
        "affine_transformed": os.path.join(input_folder, "affine_transformed")
    }

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue

            # Apply Gaussian noise
            noisy_gaussian = add_gaussian_noise(image)
            # Apply Gaussian smoothing after adding Gaussian noise
            smoothed_gaussian = apply_gaussian_smoothing(noisy_gaussian)

            # Apply salt and pepper noise
            noisy_salt_and_pepper = add_salt_and_pepper_noise(image)
            # Apply median filtering after salt and pepper noise
            median_filtered = apply_median_filter(noisy_salt_and_pepper)

            # Apply other types of noise
            noisy_speckle = add_speckle_noise(image)
            
            noisy_poisson = add_poisson_noise(image)
            # Apply Gaussian smoothing after adding Poisson noise
            # smoothed_poisson = apply_gaussian_smoothing(noisy_poisson)

            noisy_pixelation = add_pixelation_noise(image)

            # Apply affine transformation
            affine_transformed = apply_random_affine(image)

            # Save noisy and transformed images in separate folders
            save_image(output_folders["gaussian"], filename, noisy_gaussian)
            save_image(output_folders["gaussian_smoothed"], filename, smoothed_gaussian)
            save_image(output_folders["salt_and_pepper"], filename, noisy_salt_and_pepper)
            save_image(output_folders["salt_and_pepper_median_filtered"], filename, median_filtered)
            save_image(output_folders["speckle"], filename, noisy_speckle)
            save_image(output_folders["poisson"], filename, noisy_poisson)
            save_image(output_folders["pixelation"], filename, noisy_pixelation)
            save_image(output_folders["affine_transformed"], filename, affine_transformed)

if __name__ == "__main__":
    input_folder = "/home/khanm/workfolder/MFA_Net/data/imgs/"  # Change to your folder path
    apply_noise_to_images(input_folder)
