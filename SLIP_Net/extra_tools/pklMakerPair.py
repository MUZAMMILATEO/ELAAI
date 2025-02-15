import os
import numpy as np
import cv2
import argparse
import pickle

def read_images(image_dir, mask_dir):
    image_files = os.listdir(image_dir)
    image_files.sort(key=lambda x: int(x.split('.')[0].split('-')[1]))
    mask_files = os.listdir(mask_dir)
    mask_files.sort(key=lambda x: int(x.split('.')[0].split('-')[1].split('_')[0]))
    images = []
    masks = []
    for img_file, mask_file in zip(image_files, mask_files):
        if img_file.endswith('.png') and mask_file.endswith('.png'):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            images.append((img, img_file))  # Save image along with its filename
            masks.append(mask)
    return images, masks

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val + 1e-6)
    else:
        return image

def create_pkl_files(image_dir, mask_dir, output_dir):
    images, masks = read_images(image_dir, mask_dir)
    num_pairs = min(len(images), len(masks)) - 1
    for i in range(num_pairs):
        fixed_image, fixed_image_name = images[i]
        moving_image, _ = images[i + 1]
        
        # Normalize images and masks
        fixed_image = normalize_image(fixed_image)
        moving_image = normalize_image(moving_image)
        fixed_mask = normalize_image(masks[i])
        moving_mask = normalize_image(masks[i + 1])
        
        data = {
            'fixed_image': fixed_image,
            'moving_image': moving_image,
            'fixed_mask': fixed_mask,
            'moving_mask': moving_mask
        }
        
        output_filename = f'{os.path.splitext(fixed_image_name)[0]}.pkl'
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
def create_pkl_files_rev(image_dir, mask_dir, output_dir):
    images, masks = read_images(image_dir, mask_dir)
    num_pairs = min(len(images), len(masks)) - 1
    for i in range(num_pairs , 0, -1):
        fixed_image, fixed_image_name = images[i]
        moving_image, _ = images[i - 1]
        
        # Normalize images and masks
        fixed_image = normalize_image(fixed_image)
        moving_image = normalize_image(moving_image)
        fixed_mask = normalize_image(masks[i])
        moving_mask = normalize_image(masks[i - 1])
        
        data = {
            'fixed_image': fixed_image,
            'moving_image': moving_image,
            'fixed_mask': fixed_mask,
            'moving_mask': moving_mask
        }
        
        output_filename = f'{os.path.splitext(fixed_image_name)[0]}_rev.pkl'
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Convert MRI slice images and masks to PKL files")
    parser.add_argument("--image_dir", help="Path to directory containing MRI slice image files")
    parser.add_argument("--mask_dir", help="Path to directory containing corresponding masks")
    parser.add_argument("--output_dir", help="Path to the output directory")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_pkl_files(args.image_dir, args.mask_dir, args.output_dir)
    
    # comment the following line to generate pairs in forward direction only
    create_pkl_files_rev(args.image_dir, args.mask_dir, args.output_dir)

if __name__ == "__main__":
    main()
