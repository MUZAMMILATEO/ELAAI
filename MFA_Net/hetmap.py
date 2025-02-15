import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage.transform import resize
from ModelArchitecture.DiceLoss import dice_metric_loss
from CustomLayers.ConvBlock2DMK import MultiplyWithExtractedWeights, MultiplyWithExtractedWeightsMidScope, ExtractWeightLayer

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Set the path to your saved model and input image
model_weights_path = '/home/khanm/workfolder/MFA_Net/ModelSaveTensorFlow/MFANet_filters_704_17_2024-08-06_18-26-22.h5'
input_image_path = '/home/khanm/workfolder/bw/MFA_Net/data/imgs/ZGT_10_t2_blade_sag-52.png'  # Replace with the actual image path
output_heatmap_path = '/home/khanm/workfolder/bw/MFA_Net/heatmap.png'  # Path to save heatmap
output_feature_maps_path = '/home/khanm/workfolder/bw/MFA_Net/feature_maps_grid.png'  # Path to save feature maps grid

# Load the model
model = load_model(model_weights_path, custom_objects={
    'dice_metric_loss': dice_metric_loss,
    'MultiplyWithExtractedWeights': MultiplyWithExtractedWeights,
    'MultiplyWithExtractedWeightsMidScope': MultiplyWithExtractedWeightsMidScope,
    'ExtractWeightLayer': ExtractWeightLayer
})

# Load the input image and preprocess it
input_image = load_img(input_image_path)
input_array = img_to_array(input_image)

# Resize the input image to the model's expected input shape (704x704)
input_array_resized = resize(input_array, (704, 704, 3), preserve_range=True)
input_array_resized = np.expand_dims(input_array_resized, axis=0)  # Add batch dimension

# Create a new model to extract the output of the t53 layer
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('multiply_with_extracted_weights_9').output)

# Get the feature maps from the l1o layer
feature_maps = intermediate_layer_model.predict(input_array_resized)

# Resize each feature map to the original image size (448x448 in this case)
height, width = input_array.shape[0:2]
resized_feature_maps = np.zeros((feature_maps.shape[0], height, width, feature_maps.shape[-1]))

for i in range(feature_maps.shape[-1]):
    resized_feature_maps[0, :, :, i] = resize(feature_maps[0, :, :, i], (height, width))

# Fuse the feature maps by taking the sum across the channel dimension
fused_feature_map = np.sum(resized_feature_maps[0], axis=-1)

# Normalize the fused feature map to range between 0 and 1
fused_feature_map -= fused_feature_map.min()
fused_feature_map /= fused_feature_map.max()

# Generate and save the heatmap
plt.imshow(fused_feature_map, cmap='viridis')
plt.colorbar()
plt.title('Heatmap from l1o Layer Feature Maps')
plt.savefig(output_heatmap_path)
plt.close()

# Generate and save the grid of feature maps
num_feature_maps = resized_feature_maps.shape[-1]
grid_size = int(np.ceil(np.sqrt(num_feature_maps)))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes = axes.flatten()

for i in range(num_feature_maps):
    ax = axes[i]
    ax.imshow(resized_feature_maps[0, :, :, i], cmap='viridis')
    ax.axis('off')

for i in range(num_feature_maps, len(axes)):
    axes[i].axis('off')

plt.suptitle('Feature Maps from l1o Layer')
plt.savefig(output_feature_maps_path)
plt.close()

print(f"Heatmap saved to {output_heatmap_path}")
print(f"Feature maps grid saved to {output_feature_maps_path}")
