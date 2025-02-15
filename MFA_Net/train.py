import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import MFA_Net
from ImageLoader.ImageLoader2D import load_data
from CustomLayers.ConvBlock2DMK import MultiplyWithExtractedWeights, MultiplyWithExtractedWeightsMidScope, ExtractWeightLayer  # Import your custom layer

# Checking the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Setting the model parameters
img_size = 704 #352
dataset_type = '/home/khanm/workfolder/bw/MFA_Net/data/' # 'kvasir'  # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb
learning_rate = 1e-2
seed_value = 58800
filters = 17  # Number of filters, the paper presents the results with 17 and 34
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# ct = datetime.now()
ct = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Ensure unique timestamp format

model_type = "MFA_Net"

# Define checkpoint and model save paths
checkpoint_dir = 'checkpoints'
model_save_dir = 'ModelSaveTensorFlow'

# Ensure directories exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# Sanitize paths to avoid double slashes
dataset_type_sanitized = dataset_type.strip('/').replace('/', '_')

progress_path = os.path.join(checkpoint_dir, f'{model_type}_progress_csv_filters_{filters}_{ct}.csv')
progressfull_path = os.path.join(checkpoint_dir, f'{model_type}_progress_filters_{filters}_{ct}.txt')
plot_path = os.path.join(checkpoint_dir, f'{model_type}_progress_plot_filters_{filters}_{ct}.png')
model_path = os.path.join(model_save_dir, f'MFANet_{model_type}_filters_704_2000_{filters}_{ct}.h5')

EPOCHS = 2000 #500 #600
min_loss_for_saving = 1.0 # 0.2

# Loading the data
X, Y = load_data(img_size, img_size, -1, dataset_type)

# Splitting the data, seed for reproducibility
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=seed_value)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.111, shuffle=True, random_state=seed_value)

# Defining the augmentations
aug_train = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True),
])

def augment_images():
    x_train_out = []
    y_train_out = []

    for i in range(len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'])

    return np.array(x_train_out), np.array(y_train_out)

# Creating the model
model = MFA_Net.create_model(img_height=img_size, img_width=img_size, input_chanels=3, out_classes=1, starting_filters=filters)

# Compiling the model
model.compile(optimizer=optimizer, loss=dice_metric_loss)

# Training the model
step = 0

for epoch in range(EPOCHS):
    print(f'Training, epoch {epoch}')
    print('Learning Rate: ' + str(learning_rate))

    step += 1

    image_augmented, mask_augmented = augment_images()

    csv_logger = tf.keras.callbacks.CSVLogger(progress_path, append=True, separator=';')

    model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=4, validation_data=(x_valid, y_valid), verbose=1, callbacks=[csv_logger])

    prediction_valid = model.predict(x_valid, verbose=0)
    loss_valid = dice_metric_loss(y_valid, prediction_valid)
    loss_valid = loss_valid.numpy()
    print("Loss Validation: " + str(loss_valid))

    prediction_test = model.predict(x_test, verbose=0)
    loss_test = dice_metric_loss(y_test, prediction_test)
    loss_test = loss_test.numpy()
    print("Loss Test: " + str(loss_test))

    with open(progressfull_path, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\nval_loss: ' + str(loss_valid) + '\ntest_loss: ' + str(loss_test) + '\n\n\n')

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print("Saved model with val_loss: ", loss_valid)
        model.save(model_path)

    del image_augmented
    del mask_augmented

    gc.collect()

# Computing the metrics and saving the results
print("Loading the model")
model = tf.keras.models.load_model(model_path, custom_objects={'dice_metric_loss': dice_metric_loss, 'MultiplyWithExtractedWeights': MultiplyWithExtractedWeights, 'MultiplyWithExtractedWeightsMidScope': MultiplyWithExtractedWeightsMidScope, 'ExtractWeightLayer': ExtractWeightLayer})

prediction_train = model.predict(x_train, batch_size=4)
prediction_valid = model.predict(x_valid, batch_size=4)
prediction_test = model.predict(x_test, batch_size=4)

print("Predictions done")

dice_train = f1_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
dice_test = f1_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
dice_valid = f1_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

print("Dice finished")

miou_train = jaccard_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
miou_test = jaccard_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
miou_valid = jaccard_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

print("Miou finished")

precision_train = precision_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
precision_test = precision_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
precision_valid = precision_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

print("Precision finished")

recall_train = recall_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
recall_test = recall_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
recall_valid = recall_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

print("Recall finished")

accuracy_train = accuracy_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
accuracy_test = accuracy_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))

print("Accuracy finished")

# Sanitize dataset_type for use in filenames
final_file = f'results_{model_type}_{filters}_{ct}.txt'
print(final_file)

with open(final_file, 'a') as f:
    f.write(dataset_type_sanitized + '\n\n')
    f.write('dice_train: ' + str(dice_train) + ' dice_valid: ' + str(dice_valid) + ' dice_test: ' + str(dice_test) + '\n\n')
    f.write('miou_train: ' + str(miou_train) + ' miou_valid: ' + str(miou_valid) + ' miou_test: ' + str(miou_test) + '\n\n')
    f.write('precision_train: ' + str(precision_train) + ' precision_valid: ' + str(precision_valid) + ' precision_test: ' + str(precision_test) + '\n\n')
    f.write('recall_train: ' + str(recall_train) + ' recall_valid: ' + str(recall_valid) + ' recall_test: ' + str(recall_test) + '\n\n')
    f.write('accuracy_train: ' + str(accuracy_train) + ' accuracy_valid: ' + str(accuracy_valid) + ' accuracy_test: ' + str(accuracy_test) + '\n\n\n\n')

print('File done')