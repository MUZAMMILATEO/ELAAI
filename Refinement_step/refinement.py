import numpy as np
import cv2
from scipy.ndimage import center_of_mass
from skimage.segmentation import flood_fill
import matplotlib.pyplot as plt
import os
import sys

def adaptive_tolerance(mri_scan, binary_mask):
    # Apply the binary mask to the MRI scan
    masked_mri = cv2.bitwise_and(mri_scan, mri_scan, mask=binary_mask)
    
    # Compute the histogram of the masked MRI scan
    hist = cv2.calcHist([masked_mri], [0], binary_mask, [256], [0, 256])
    
    # Find the most common intensity value in the masked region (mode)
    mode_intensity = np.argmax(hist)
    
    # Compute the standard deviation of the intensities in the masked region
    std_intensity = np.std(masked_mri[binary_mask > 0])
    
    # Set tolerance as a fraction of the standard deviation
    tolerance = std_intensity * 1.5
    
    return int(tolerance)
    
holefill = 3                  # If final masks have many holes, increase holefilling value
iterations = 1
connIter = 1
toleranceFactor = 1.0 #0.75 If intensity variations are large, make the factor large to capture variability
thicknessIter = 5
factorErode = 0.15 #0.45 Reduce this factor to increase the thickness of the final mask (subtracted mask)
inErode = 0
pad_width_val = 99
kerSize = 7        # 5
kernel_ed = np.ones((kerSize, kerSize), np.uint8)  # Adjust kernel size as needed, increase to remove small circular holes

def process_images(mri_scan_path, binary_mask_path, output_path, resized_mask_output_path, edges_output_path, subtracted_output_path):
    # Load images
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    mri_scan = cv2.imread(mri_scan_path, cv2.IMREAD_GRAYSCALE)
    
    binary_mask = cv2.erode(binary_mask.astype(np.uint8), kernel_ed, iterations=inErode)

    # Print shapes for debugging
    print(f"Processing {mri_scan_path}")
    print(f"Shape of MRI scan: {mri_scan.shape}")
    print(f"Shape of binary mask: {binary_mask.shape}")

    # Pad the binary mask and MRI scan with zeros
    pad_width = pad_width_val
    binary_mask_padded = np.pad(binary_mask, pad_width, mode='constant', constant_values=0)
    mri_scan_padded = np.pad(mri_scan, pad_width, mode='constant', constant_values=0)

    # Resize the binary mask to match the dimensions of the MRI scan
    binary_mask_resized = cv2.resize(binary_mask_padded, (mri_scan_padded.shape[1], mri_scan_padded.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask_resized = cv2.GaussianBlur(binary_mask_resized, (11, 11), 0)

    # Set all pixels above 128 to 255 and below 128 to 0
    _, binary_mask_resized = cv2.threshold(binary_mask_resized, 128, 255, cv2.THRESH_BINARY)
    binary_mask_resized = cv2.dilate(binary_mask_resized.astype(np.uint8), kernel_ed, iterations=iterations)
    binary_mask_resized = cv2.erode(binary_mask_resized.astype(np.uint8), kernel_ed, iterations=iterations//2)
    
    #################################################
    
    binary_mask_resized_center_fill = flood_fill(binary_mask_resized, (0,0), new_value=255, tolerance=0.5)
    binary_mask_resized_center_fill = cv2.bitwise_not(binary_mask_resized_center_fill)
    com = center_of_mass(binary_mask_resized_center_fill)
    
    if np.any(np.isnan(com)):
        com = center_of_mass(binary_mask_resized)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # Compute the center of mass of the bladder region in the binary mask
    # com = center_of_mass(binary_mask_resized)
    com = tuple(map(int, com))

    # Print center of mass for debugging
    print(f"Center of mass: {com}")

    # Check if the center of mass is within the image bounds
    if com[0] < 0 or com[0] >= mri_scan_padded.shape[0] or com[1] < 0 or com[1] >= mri_scan_padded.shape[1]:
        raise ValueError("Computed center of mass is out of image bounds")

    # Calculate adaptive tolerance
    tolerance = adaptive_tolerance(mri_scan_padded, binary_mask_resized)
    tolerance = tolerance * toleranceFactor
    print(f"Adaptive tolerance: {tolerance}")

    # Flood fill the inside of the input binary mask completely with white pixels
    binary_mask_resized_diler = cv2.dilate(binary_mask_resized.astype(np.uint8), kernel_ed, iterations=connIter)
    binary_mask_resized_diler = cv2.erode(binary_mask_resized_diler.astype(np.uint8), kernel_ed, iterations=connIter)
    filled_binary_mask = flood_fill(binary_mask_resized_diler, com, new_value=255)

    # Perform morphological closing to fill any unwanted gaps
    kernel = np.ones((holefill, holefill), np.uint8)
    closed_filled_binary_mask = cv2.morphologyEx(filled_binary_mask, cv2.MORPH_CLOSE, kernel)

    # Compute the edges from the completely filled new mask
    edges_filled_mask = cv2.Canny(closed_filled_binary_mask, 100, 200)

    # Normalize the MRI scan for better region growing
    norm_mri_scan = cv2.normalize(mri_scan_padded, None, 0, 255, cv2.NORM_MINMAX)

    # Perform region growing from the center of mass on the normalized MRI scan
    flooded = flood_fill(norm_mri_scan, com, new_value=255, tolerance=tolerance)

    # Create a mask for the expanded region
    expanded_region = flooded == 255

    # Convert expanded_region to uint8 for morphologyEx
    expanded_region_uint8 = expanded_region.astype(np.uint8) * 255

    # Apply morphological closing to fill any unwanted gaps
    kernel = np.ones((holefill, holefill), np.uint8)
    expanded_region_uint8 = cv2.morphologyEx(expanded_region_uint8, cv2.MORPH_CLOSE, kernel)

    # Convert expanded_region_uint8 back to original type
    expanded_region = expanded_region_uint8 == 255

    # Convert to uint8 for visualization
    expanded_region_uint8 = (expanded_region * 255).astype(np.uint8)

    # Restrict the flooding of the bladder with input bladder mask
    closed_filled_binary_mask_eroded = cv2.erode(closed_filled_binary_mask.astype(np.uint8), kernel_ed, iterations=thicknessIter)
    expanded_region_uint8 = cv2.bitwise_and(closed_filled_binary_mask_eroded, expanded_region_uint8)

    # Compute the edges from the flooded mask with noise reduction
    # Apply Gaussian blur to reduce noise in the flooded mask
    blurred_flooded_mask = cv2.GaussianBlur(expanded_region_uint8, (5, 5), 0)
    edges_flooded_mask = cv2.Canny(blurred_flooded_mask, 100, 200)

    # Compute the XOR mask between the new filled binary mask and the flooded mask
    xor_mask = cv2.bitwise_xor(closed_filled_binary_mask, expanded_region_uint8)
    xor_mask = cv2.bitwise_and(closed_filled_binary_mask, xor_mask)

    # Find contours in the XOR mask
    contours, _ = cv2.findContours(xor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Label connected components in the XOR mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(xor_mask, connectivity=4)
    largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Skip the background component

    # Create a mask for the largest component
    largest_component_mask = np.zeros_like(xor_mask, dtype=np.uint8)
    largest_component_mask[labels == largest_component_idx] = 255

    # Update the given bladder wall mask with the largest component mask
    updated_color_mask = np.zeros((mri_scan_padded.shape[0], mri_scan_padded.shape[1], 3), dtype=np.uint8)
    updated_color_mask[:, :, 1] = expanded_region_uint8
    updated_color_mask[:, :, 2] = binary_mask_resized

    # Compute the center of mass of the largest component mask
    com_largest = center_of_mass(largest_component_mask)
    com_largest = tuple(map(int, com_largest))

    # Print center of mass for debugging
    print(f"Center of mass of largest component: {com_largest}")

    # Calculate thicknesses
    thicknesses = []
    height, width = largest_component_mask.shape

    for angle in np.arange(0,360,0.5):
        theta = np.deg2rad(angle)
        line_mask = np.zeros_like(largest_component_mask)
        # Calculate the end point of the line (1000 pixels long, which should be more than enough)
        end_x = int(com_largest[0] + 1000 * np.cos(theta))
        end_y = int(com_largest[1] + 1000 * np.sin(theta))
    
        # Draw the line on the mask
        cv2.line(line_mask, (com_largest[1], com_largest[0]), (end_y, end_x), 255, 1)
    
        # Compute the intersection of the line with the largest component mask
        intersection = cv2.bitwise_and(largest_component_mask, line_mask)
    
        if 0 < angle < 180:
            # Consider only the right side of the intersection
            intersection[com_largest[0]:, :] = 0
        elif 180 < angle < 360:
            # Consider only the left side of the intersection
            intersection[:com_largest[0], :] = 0
        elif angle == 0:
            # Consider only the bottom side of the intersection
            intersection[:, com_largest[1]:] = 0
        elif angle == 180:
            # Consider only the top side of the intersection
            intersection[:, :com_largest[1]] = 0

        # Count the number of intersecting pixels (thickness)
        thickness = np.sum(intersection) / 255  # Each white pixel is 255, so divide by 255
        thicknesses.append(thickness)

    
    median_thickness = np.median(thicknesses)
    erosion_iterations = int(np.ceil(median_thickness * factorErode))
    
    print(f"Median thickness: {median_thickness}")
    print(f"Erosion iterations based on {factorErode} of median thickness: {erosion_iterations}")

    # Remove padding from the final output
    def remove_padding(image, pad_width):
        return image[pad_width:-pad_width, pad_width:-pad_width]

    largest_component_mask = remove_padding(largest_component_mask, pad_width)
    expanded_region_uint8 = remove_padding(expanded_region_uint8, pad_width)
    closed_filled_binary_mask = remove_padding(closed_filled_binary_mask, pad_width)
    edges_flooded_mask = remove_padding(edges_flooded_mask, pad_width)
    edges_filled_mask = remove_padding(edges_filled_mask, pad_width)
    mri_scan = remove_padding(mri_scan_padded, pad_width)
    flooded = remove_padding(flooded, pad_width)

    # Compute the union of expanded_region_uint8 and largest_component_mask
    new_mask = cv2.bitwise_or(expanded_region_uint8, largest_component_mask)

    # Erode the new_mask
    new_mask_eroded = cv2.erode(new_mask, kernel_ed, iterations=erosion_iterations)

    # Subtract new_mask_updated with expanded_region_uint8 to obtain subtracted_mask
    subtracted_mask = cv2.subtract(new_mask_eroded, expanded_region_uint8)
    
    # Compute the edges of expanded_region_uint8
    expanded_region_edges = cv2.Canny(expanded_region_uint8, 100, 200)

    # Check if expanded_region_edges is an image with all zeros
    if np.all(expanded_region_edges == 0):
        largest_component_mask_resized = cv2.resize(largest_component_mask, (expanded_region_uint8.shape[1], expanded_region_uint8.shape[0]), interpolation=cv2.INTER_NEAREST)
        expanded_region_edges = cv2.Canny(largest_component_mask_resized, 100, 200)

    expanded_region_edges_output_filename = os.path.join(edges_output_path, os.path.splitext(os.path.basename(mri_scan_path))[0] + '_edges.png')
    cv2.imwrite(expanded_region_edges_output_filename, expanded_region_edges)
    
    subtracted_mask = cv2.bitwise_or(subtracted_mask, expanded_region_edges)
    
    # Create a color overlay
    overlay = np.zeros((mri_scan.shape[0], mri_scan.shape[1], 3), dtype=np.uint8)

    # Color the xor_mask region with green
    overlay[subtracted_mask == 255] = [0, 204, 0]  # Green

    # Color the rest of the regions with red
    overlay[subtracted_mask == 0] = [255, 255, 204]  # Red

    # Convert the grayscale MRI scan to BGR format
    mri_scan_color = cv2.cvtColor(mri_scan, cv2.COLOR_GRAY2BGR)

    # Blend the overlay with the original MRI scan
    alpha = 0.5  # Transparency factor
    blended_image = cv2.addWeighted(mri_scan_color, alpha, overlay, alpha, 0)

    # Save the resized binary mask at three times the original resolution
    largest_component_mask_resized = cv2.resize(largest_component_mask, (mri_scan.shape[1] * 3, mri_scan.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    resized_mask_output_filename = os.path.join(resized_mask_output_path, os.path.splitext(os.path.basename(binary_mask_path))[0] + '_resized.png')
    cv2.imwrite(resized_mask_output_filename, largest_component_mask_resized)

    edge_mask = cv2.bitwise_or(edges_flooded_mask, edges_filled_mask)
    
    # Checking condition for the presence of inner wall
    # Apply Gaussian Blur to smooth the image
    smoothed_mask = cv2.GaussianBlur(largest_component_mask, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(smoothed_mask, 50, 150)

    # Use morphological operations to enhance the edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the mask to draw contours on
    contour_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_display, contours, -1, (0, 255, 0), 2)

    # Count the number of nested contours (one inside the other)
    nested_count = 0
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:  # Check if the contour has a parent
            nested_count += 1

    # Save the edge image with contours for debugging
    resized_mask_output_filename = os.path.join(resized_mask_output_path, os.path.splitext(os.path.basename(binary_mask_path))[0] + '_curCount.png')
    cv2.imwrite(resized_mask_output_filename, contour_display)

    # Print the number of nested contours and update the mask if necessary
    if nested_count < 2:
        subtracted_mask = expanded_region_edges
        print(f"The number of nested curves is {nested_count}!")
    else:
        subtracted_mask = subtracted_mask

    # Save the subtracted mask
    subtracted_mask_output_filename = os.path.join(subtracted_output_path, os.path.splitext(os.path.basename(mri_scan_path))[0] + '_subtracted.png')
    cv2.imwrite(subtracted_mask_output_filename, subtracted_mask)

    # Save the plot directly to the output path
    plt.figure(figsize=(20, 8))

    plt.subplot(2, 3, 1)
    plt.title('Input MRI')
    plt.imshow(mri_scan, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 2)
    plt.title('Initial mask')
    plt.imshow(binary_mask_resized, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 3)
    plt.title('Flooded Region')
    plt.imshow(flooded, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.title('Bladder walls')
    plt.imshow(edge_mask, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.title('Final mask')
    plt.imshow(largest_component_mask, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.title('Overlaid image')
    plt.imshow(blended_image, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(os.path.join(output_path, os.path.splitext(os.path.basename(mri_scan_path))[0] + '_plot.png'))
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage: python script.py <path_to_mri_scans> <path_to_binary_masks> <output_path> <resized_mask_output_path> <edges_output_path> <subtracted_output_path>")
        sys.exit(1)

    mri_scans_path = sys.argv[1]
    binary_masks_path = sys.argv[2]
    output_path = sys.argv[3]
    resized_mask_output_path = sys.argv[4]
    edges_output_path = sys.argv[5]
    subtracted_output_path = sys.argv[6]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(resized_mask_output_path):
        os.makedirs(resized_mask_output_path)

    if not os.path.exists(edges_output_path):
        os.makedirs(edges_output_path)

    if not os.path.exists(subtracted_output_path):
        os.makedirs(subtracted_output_path)

    mri_scan_files = [f for f in os.listdir(mri_scans_path) if f.endswith('png')]
    binary_mask_files = [f for f in os.listdir(binary_masks_path) if f.endswith('png')]

    for mri_scan_file in mri_scan_files:
        binary_mask_file = os.path.splitext(mri_scan_file)[0] + '.png'
        if binary_mask_file in binary_mask_files:
            print(os.path.join(mri_scans_path, mri_scan_file))
            print(os.path.join(binary_masks_path, binary_mask_file))
            process_images(
                os.path.join(mri_scans_path, mri_scan_file),
                os.path.join(binary_masks_path, binary_mask_file),
                output_path,
                resized_mask_output_path,
                edges_output_path,
                subtracted_output_path
            )
        else:
            print(f"No corresponding binary mask found for {mri_scan_file}")
