import os
import cv2
import numpy as np
import argparse
from glob import glob

def process_masks(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of all mask files in the input folder
    mask_files = glob(os.path.join(input_folder, "*.png"))

    for mask_file in mask_files:
        # Read the binary mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading {mask_file}")
            continue

        kernel_ed = np.ones((3, 3), np.uint8)            # make sure that erosion does not cause loop opening in bladder or the refinement may fail
                                                         # if pixelization removal cause loop opening, avoid it by reducing kernel size
        thresholded_mask = cv2.erode(mask.astype(np.uint8), kernel_ed, iterations=1)
        
        # thresholded_mask = cv2.dilate(mask.astype(np.uint8), kernel_ed, iterations=1)

        # Apply Gaussian smoothing
        smoothed_mask = cv2.GaussianBlur(thresholded_mask, (15, 15), 0)
        
        smoothed_mask = cv2.dilate(smoothed_mask.astype(np.uint8), kernel_ed, iterations=1)

        # Apply thresholding
        _, thresholded_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

        # Construct the output file path
        output_file = os.path.join(output_folder, os.path.basename(mask_file))

        # Save the processed mask
        cv2.imwrite(output_file, thresholded_mask)
        print(f"Processed and saved {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process binary masks with Gaussian smoothing and thresholding.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing 8-bit binary masks.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where processed masks will be saved.")
    args = parser.parse_args()

    process_masks(args.input_folder, args.output_folder)
