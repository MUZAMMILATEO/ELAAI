import os
import sys
import tensorflow as tf
from keras.models import load_model
from ModelArchitecture.DiceLoss import dice_metric_loss
from CustomLayers.ConvBlock2DMK import MultiplyWithExtractedWeights, MultiplyWithExtractedWeightsMidScope, ExtractWeightLayer

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Set the path to your saved model
model_weights_path = '/home/khanm/workfolder/MFA_Net/ModelSaveTensorFlow/MFANet_filters_704_17_2024-08-06_18-26-22.h5'

# Load the model
model = load_model(model_weights_path, custom_objects={
    'dice_metric_loss': dice_metric_loss,
    'MultiplyWithExtractedWeights': MultiplyWithExtractedWeights,
    'MultiplyWithExtractedWeightsMidScope': MultiplyWithExtractedWeightsMidScope,
    'ExtractWeightLayer': ExtractWeightLayer
})

# Print the names of all the layers
for layer in model.layers:
    print(layer.name)

