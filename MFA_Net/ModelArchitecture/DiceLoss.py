import tensorflow as tf
from keras.src import backend as K

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = tf.reshape(ground_truth, [-1])
    predictions = tf.reshape(predictions, [-1])
    intersection = tf.reduce_sum(predictions * ground_truth)
    union = tf.reduce_sum(predictions) + tf.reduce_sum(ground_truth)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice