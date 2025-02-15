import os
import shutil
import random

def move_samples(train_folder, val_folder, fraction):
    # Define paths to positive and negative subfolders in train and val directories
    train_pos_dir = os.path.join(train_folder, 'blad')
    train_neg_dir = os.path.join(train_folder, 'noBlad')
    val_pos_dir = os.path.join(val_folder, 'blad')
    val_neg_dir = os.path.join(val_folder, 'noBlad')

    # Create validation subdirectories if they do not exist
    os.makedirs(val_pos_dir, exist_ok=True)
    os.makedirs(val_neg_dir, exist_ok=True)

    # List all files in positive and negative subfolders in train directory
    pos_files = os.listdir(train_pos_dir)
    neg_files = os.listdir(train_neg_dir)

    # Determine the number of files to move
    num_pos_to_move = int(len(pos_files) * fraction)
    num_neg_to_move = int(len(neg_files) * fraction)

    # Randomly select files to move
    pos_files_to_move = random.sample(pos_files, num_pos_to_move)
    neg_files_to_move = random.sample(neg_files, num_neg_to_move)

    # Move positive samples
    for file in pos_files_to_move:
        src_path = os.path.join(train_pos_dir, file)
        dst_path = os.path.join(val_pos_dir, file)
        shutil.move(src_path, dst_path)

    # Move negative samples
    for file in neg_files_to_move:
        src_path = os.path.join(train_neg_dir, file)
        dst_path = os.path.join(val_neg_dir, file)
        shutil.move(src_path, dst_path)

    print(f"Moved {num_pos_to_move} positive samples and {num_neg_to_move} negative samples to validation folder.")

# Example usage
train_folder = '/home/khanm/workfolder/bw/MFA_Net/classification/train/'
val_folder = '/home/khanm/workfolder/bw/MFA_Net/classification/val/'
fraction = 0.2  # 20% of samples will be moved

move_samples(train_folder, val_folder, fraction)
