import os
import sys
import tensorflow as tf
from keras.models import load_model, Model
from PIL import Image
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import exposure
from ModelArchitecture.MFA_Net import create_model
from ModelArchitecture.DiceLoss import dice_metric_loss  # Import the dice_metric_loss function
from CustomLayers.ConvBlock2DMK import MultiplyWithExtractedWeights, MultiplyWithExtractedWeightsMidScope, ExtractWeightLayer  # Import your custom layer

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Set paths
model_weights_path = '/home/khanm/workfolder/MFA_Net/ModelSaveTensorFlow/MFANet_filters_704_2000_17_2024-08-07_09-18-41.h5'  # Path to the saved model
input_image_path = '/home/khanm/workfolder/MFA_Net/bladder_tumor_test/ZGT_24_t2_blade_sag-50.png'  # Path to single input image
output_dir = '/home/khanm/workfolder/MFA_Net/data_test/'  # Path to directory to save output segmentation masks and feature maps

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Model parameters (ensure these match those used during training)
img_height = 704
img_width = 704
input_channels = 3
output_classes = 1
starting_filters = 17  # Ensure this matches the training configuration

# Load the original model
original_model = load_model(model_weights_path, custom_objects={'dice_metric_loss': dice_metric_loss, 'MultiplyWithExtractedWeights': MultiplyWithExtractedWeights, 'MultiplyWithExtractedWeightsMidScope': MultiplyWithExtractedWeightsMidScope, 'ExtractWeightLayer': ExtractWeightLayer})

# Function to preprocess the input image
def preprocess_image(image_path, img_height, img_width):
    image = imread(image_path)
    pillow_image = Image.fromarray(image)
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

# Function to save feature maps as a grid
def save_feature_maps(feature_maps, output_path):
    num_filters = feature_maps.shape[-1]
    size = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(size, size, figsize=(12, 12))

    for i in range(size):
        for j in range(size):
            ax = axes[i, j]
            if i * size + j < num_filters:
                ax.imshow(feature_maps[0, :, :, i * size + j], cmap='gray') #'viridis')
            ax.axis('off')

    plt.savefig(output_path)
    plt.close()

# Function to multiply feature maps elementwise and upsample to original image size
def multiply_and_upsample_feature_maps(feature_maps, original_height, original_width):
    multiplied_map = np.sum(feature_maps, axis=-1)
    multiplied_map = (multiplied_map - np.min(multiplied_map)) / (np.max(multiplied_map) - np.min(multiplied_map))  # Normalize
    multiplied_map = Image.fromarray((multiplied_map * 255).astype(np.uint8))  # Convert to PIL Image
    multiplied_map = multiplied_map.resize((original_height, original_width))  # Resize to original size
    return np.array(multiplied_map) / 255.0  # Normalize back to [0, 1] range

# Function to enhance contrast of the heatmap
def enhance_contrast(heatmap):
    return exposure.equalize_adapthist(heatmap, clip_limit=0.03)

# Function to overlay heatmap on the input image
def overlay_heatmap_on_image(image_path, heatmap, output_path):
    image = Image.open(image_path).convert('RGBA')
    heatmap = np.uint8(plt.cm.jet(heatmap) * 255)
    heatmap = Image.fromarray(heatmap).resize(image.size)
    heatmap = heatmap.convert("RGBA")
    blended = Image.blend(image, heatmap, alpha=0.5)
    blended.save(output_path)

for i in range(1, 2):
    # Create a model that outputs the feature maps of the desired intermediate layer
    intermediate_layer_name = 'conv2d_272' #'multiply_with_extracted_weights_{}'.format(i)
    intermediate_layer_model = Model(inputs=original_model.input, outputs=original_model.get_layer(intermediate_layer_name).output)

    # Preprocess the input image
    input_image = preprocess_image(input_image_path, img_height, img_width)

    # Predict the segmentation map
    predicted_segmentation = original_model.predict(input_image)

    # Postprocess the output segmentation
    original_image = Image.open(input_image_path)
    original_height, original_width = original_image.size
    output_segmentation = postprocess_segmentation(predicted_segmentation, original_height, original_width)

    # Save the output segmentation map
    output_segmentation_path = os.path.join(output_dir, 'segmentation.png')
    output_segmentation.save(output_segmentation_path)
    print(f"Segmentation saved at: {output_segmentation_path}")

    # Get intermediate layer output
    intermediate_output = intermediate_layer_model.predict(input_image)

    # Save the feature maps
    feature_maps_output_path = os.path.join(output_dir, 'feature_maps_{}.png'.format(i))
    save_feature_maps(intermediate_output, feature_maps_output_path)
    print(f"Feature maps saved at: {feature_maps_output_path}")

    # Multiply feature maps and upsample
    multiplied_feature_map = multiply_and_upsample_feature_maps(intermediate_output[0], original_height, original_width)

    # Enhance the contrast of the multiplied feature map
    enhanced_feature_map = multiplied_feature_map #enhance_contrast(multiplied_feature_map)

    # Rescale the multiplied feature map to [0, 1]
    rescaled_feature_map = enhanced_feature_map # (enhanced_feature_map - np.min(enhanced_feature_map)) / (np.max(enhanced_feature_map) - np.min(enhanced_feature_map))

    # Save the heatmap overlaid image
    heatmap_output_path = os.path.join(output_dir, 'heatmap_overlay_{}.png'.format(i))
    overlay_heatmap_on_image(input_image_path, rescaled_feature_map, heatmap_output_path)
    print(f"Heatmap overlay saved at: {heatmap_output_path}")

print("All segmentations, feature maps, and heatmap overlays have been processed and saved.")
