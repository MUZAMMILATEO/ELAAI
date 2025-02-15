import sys
sys.path.append('/home/khanm/.local/lib/python3.11/site-packages')
sys.path.append('/deepstore/software/anaconda3/2023.09/lib/python3.11/site-packages')

import os
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
from ModelArchitecture.MFA_Net import create_model
from ModelArchitecture.DiceLoss import dice_metric_loss
from CustomLayers.ConvBlock2DMK import MultiplyWithExtractedWeights, MultiplyWithExtractedWeightsMidScope, ExtractWeightLayer
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt
from tensorflow import keras
tf.keras.backend.set_image_data_format('channels_last')  # Ensure correct backend

sys.stdout.reconfigure(encoding='utf-8')

model_weights_path = '/home/khanm/workfolder/MFA_Net/ModelSaveTensorFlow/MFANet_filters_704_2000_17_2024-08-07_09-18-41.h5'  # Path to the saved model
input_dir = '/home/khanm/workfolder/MFA_Net/bladder_tumor_test/'  # Path to directory containing input images
output_dir = '/home/khanm/workfolder/MFA_Net/prediction_X/'  # Path to directory to save output segmentation masks
os.makedirs(output_dir, exist_ok=True)

img_height = 704
img_width = 704
input_channels = 3
output_classes = 1
starting_filters = 17

model = create_model(img_height, img_width, input_channels, output_classes, starting_filters)
model = load_model(model_weights_path, custom_objects={'dice_metric_loss': dice_metric_loss, 'MultiplyWithExtractedWeights': MultiplyWithExtractedWeights, 'MultiplyWithExtractedWeightsMidScope': MultiplyWithExtractedWeightsMidScope, 'ExtractWeightLayer': ExtractWeightLayer})

def preprocess_image(image_path, img_height, img_width):
    image = Image.open(image_path)  # Use PIL to open the image
    image = image.convert('RGB')  # Ensure the image has 3 channels (RGB)
    image = image.resize((img_height, img_width))  # Resize the image
    image = np.array(image)  # Convert the PIL image to a NumPy array
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    return image

def postprocess_segmentation(segmentation, original_height, original_width):
    segmentation = segmentation.squeeze()
    segmentation = (segmentation > 0.5).astype(np.uint8)
    segmentation = Image.fromarray(segmentation * 255)
    segmentation = segmentation.resize((original_height, original_width))
    return segmentation

# Initialize GradCAM
replace2linear = ReplaceToLinear()
gradcam = Gradcam(model,
                  model_modifier=replace2linear,
                  clone=True)

# Use a specific layer by name for GradCAM
target_layer_name = 'multiply_with_extracted_weights_9'  # Replace with the actual layer name you want to use

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_image_path = os.path.join(input_dir, filename)
        output_segmentation_path = os.path.join(output_dir, filename)

        input_image = preprocess_image(input_image_path, img_height, img_width)

        # Save the input image after preprocessing
        preprocessed_image = (input_image.squeeze() * 255).astype(np.uint8)
        Image.fromarray(preprocessed_image).save(output_segmentation_path.replace('.png', '_preprocessed.png'))

        # Predict the segmentation map
        predicted_segmentation = model.predict(input_image)

        # Generate GradCAM heatmap using the target layer by name
        cam = gradcam(CategoricalScore([0]), input_image, penultimate_layer=target_layer_name)
        heatmap = np.uint8(255 * cam[0])
        
        print(f'The shape of the heatmap is: {heatmap.shape}')

        # Ensure heatmap is 3-channel (RGB)
        if len(heatmap.shape) == 2:  # If heatmap is grayscale
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        elif heatmap.shape[2] == 1:  # If heatmap is single-channel
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (img_width, img_height))

        # Overlay heatmap on the original image
        original_image = np.array(Image.open(input_image_path).resize((img_height, img_width)))
        if True:  # Convert grayscale to BGR
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
        overlay = cv2.addWeighted(original_image, 1.0, heatmap, 0.0, 0)

        # Save the heatmap and the segmentation result
        Image.fromarray(overlay).save(output_segmentation_path.replace('.png', '_heatmap.png'))
        Image.fromarray(heatmap).save(output_segmentation_path.replace('.png', '_only_heatmap.png'))

        output_segmentation = postprocess_segmentation(predicted_segmentation, original_image.shape[0], original_image.shape[1])
        output_segmentation.save(output_segmentation_path)

        print(f"Segmentation and GradCAM heatmap saved at: {output_segmentation_path}")

print("All segmentations and heatmaps have been processed and saved.")
