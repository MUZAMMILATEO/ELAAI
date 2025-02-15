import os
import sys
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage.io import imread
from ModelArchitecture.MFA_Net import create_model
from ModelArchitecture.DiceLoss import dice_metric_loss  # Import the dice_metric_loss function
from CustomLayers.ConvBlock2DMK import MultiplyWithExtractedWeights, MultiplyWithExtractedWeightsMidScope, ExtractWeightLayer  # Import your custom layer

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Set paths
model_weights_path = '/home/khanm/workfolder/MFA_Net/ModelSaveTensorFlow/MFANet_filters_17_2024-08-06_10-54-13.h5'  # Path to the saved model
input_dir = '/home/khanm/workfolder/TransMorph_Transformer_for_Medical_Image_Registration/IXI/TransMorph/MKfiles/module02a/MRI/' # '/home/khanm/workfolder/bw/MFA_Net/data_test/imgs/'  # Path to directory containing input images
output_dir = '/home/khanm/workfolder/TransMorph_Transformer_for_Medical_Image_Registration/IXI/TransMorph/MKfiles/module02a/initialMask/' #'/home/khanm/workfolder/bw/MFA_Net/prediction/'  # Path to directory to save output segmentation masks

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Model parameters (ensure these match those used during training)
img_height = 128 #128 # 352 #128
img_width = 128 #128 # 352 #128
input_channels = 3
output_classes = 1
starting_filters = 17  # Ensure this matches the training configuration

# Load model architecture
model = create_model(img_height, img_width, input_channels, output_classes, starting_filters)

# Load model
model = load_model(model_weights_path, custom_objects={'dice_metric_loss': dice_metric_loss, 'MultiplyWithExtractedWeights': MultiplyWithExtractedWeights, 'MultiplyWithExtractedWeightsMidScope': MultiplyWithExtractedWeightsMidScope, 'ExtractWeightLayer': ExtractWeightLayer})

# Function to preprocess the input image
def preprocess_image(image_path, img_height, img_width):
    image = imread(image_path)
    pillow_image = Image.fromarray(image)
    pillow_image = pillow_image.resize((128, 128))
    pillow_image = pillow_image.resize((img_height, img_width))
    image = np.array(pillow_image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to postprocess the output segmentation
def postprocess_segmentation(segmentation, original_height, original_width):
    segmentation = segmentation.squeeze()  # Remove batch dimension
    segmentation = (segmentation > 0.5).astype(np.uint8)  # Binarize
    segmentation = Image.fromarray(segmentation * 255)  # Convert to PIL Image
    segmentation = segmentation.resize((original_height, original_width))  # Resize to original size
    return segmentation

# Loop over all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust according to your image formats
        input_image_path = os.path.join(input_dir, filename)
        output_segmentation_path = os.path.join(output_dir, filename)

        # Preprocess the input image
        input_image = preprocess_image(input_image_path, img_height, img_width)

        # Predict the segmentation map
        predicted_segmentation = model.predict(input_image)

        # Postprocess the output segmentation
        original_image = Image.open(input_image_path)
        original_height, original_width = original_image.size
        output_segmentation = postprocess_segmentation(predicted_segmentation, original_height, original_width)

        # Save the output segmentation map
        output_segmentation.save(output_segmentation_path)

        print(f"Segmentation saved at: {output_segmentation_path}")

print("All segmentations have been processed and saved.")
