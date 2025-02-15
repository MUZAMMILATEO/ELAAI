from keras.layers import BatchNormalization, add, GlobalAveragePooling2D, Dense, Multiply, Reshape, Layer
from keras.layers import Conv2D
from keras import backend as K
import tensorflow as tf

kernel_initializer = 'he_uniform'
dense_units = 16

def conv_block_2DMK(x, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
    result = x
    for i in range(0, repeat):
        if block_type == 'separated':
            result = separated_conv2D_block(result, filters, size=size, padding=padding)
        elif block_type == 'mfa':
            result = mfa_conv2D_block(result, filters, size=size)
        elif block_type == 'midscope':
            result = MultiplyWithExtractedWeightsMidScope(filters)(result)
        elif block_type == 'widescope':
            result = MultiplyWithExtractedWeights(filters)(result)  # Use the custom layer
        elif block_type == 'resnet':
            result = resnet_conv2D_block(result, filters, dilation_rate)
        elif block_type == 'conv':
            result = Conv2D(filters, (size, size),
                            activation='relu', kernel_initializer=kernel_initializer, padding=padding)(result)
        elif block_type == 'double_convolution':
            result = double_convolution_with_batch_normalization(result, filters, dilation_rate)
        else:
            return None
    return result

class MultiplyWithExtractedWeights(Layer):
    def __init__(self, filters, **kwargs):
        super(MultiplyWithExtractedWeights, self).__init__(**kwargs)
        self.filters = filters
        self.dense_layer = Dense(2, use_bias=False)
        self.extract_weight_layer = ExtractWeightLayer(self.dense_layer)
        self.conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=1)
        self.conv2 = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.conv3 = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=3)
        self.batch_norm1 = BatchNormalization(axis=-1)
        self.batch_norm2 = BatchNormalization(axis=-1)
        self.batch_norm3 = BatchNormalization(axis=-1)

    def call(self, x):
        xOut = self.conv1(x)
        xOut = self.batch_norm1(xOut)
        xOut = self.conv2(xOut)
        xOut = self.batch_norm2(xOut)
        xOut = self.conv3(xOut)
        xOut = self.batch_norm3(xOut)

        gap = GlobalAveragePooling2D()(xOut)
        dense_output = self.dense_layer(gap)
        weight_vector_output = self.extract_weight_layer(gap)
        weight_vector_output = tf.expand_dims(weight_vector_output, axis=0)
        weight_vector_output = tf.tile(weight_vector_output, [tf.shape(xOut)[0], 1])

        xOut0 = Multiply()([xOut, weight_vector_output])

        return xOut0

class MultiplyWithExtractedWeightsMidScope(Layer):
    def __init__(self, filters, **kwargs):
        super(MultiplyWithExtractedWeightsMidScope, self).__init__(**kwargs)
        self.filters = filters
        self.dense_layer = Dense(2, use_bias=False)
        self.extract_weight_layer = ExtractWeightLayer(self.dense_layer)
        self.conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=1)
        self.conv2 = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=3)
        self.batch_norm1 = BatchNormalization(axis=-1)
        self.batch_norm2 = BatchNormalization(axis=-1)

    def call(self, x):
        xOut = self.conv1(x)
        xOut = self.batch_norm1(xOut)
        xOut = self.conv2(xOut)
        xOut = self.batch_norm2(xOut)

        gap = GlobalAveragePooling2D()(xOut)
        dense_output = self.dense_layer(gap)
        weight_vector_output = self.extract_weight_layer(gap)
        weight_vector_output = tf.expand_dims(weight_vector_output, axis=0)
        weight_vector_output = tf.tile(weight_vector_output, [tf.shape(xOut)[0], 1])

        xOut0 = Multiply()([xOut, weight_vector_output])

        return xOut0

class ExtractWeightLayer(tf.keras.layers.Layer):
    def __init__(self, dense_layer, **kwargs):
        super(ExtractWeightLayer, self).__init__(**kwargs)
        self.dense_layer = dense_layer

    def call(self, inputs):
        dense_weights = self.dense_layer.kernel
        first_unit_weights = dense_weights[:, 0]
        return first_unit_weights

def mfa_conv2D_block(x, filters, size):
    x = BatchNormalization(axis=-1)(x)
    x1 = MultiplyWithExtractedWeights(filters)(x)
    x2 = MultiplyWithExtractedWeightsMidScope(filters)(x)
    x3 = conv_block_2DMK(x, filters, 'resnet', repeat=1)
    x4 = conv_block_2DMK(x, filters, 'resnet', repeat=2)
    x5 = conv_block_2DMK(x, filters, 'resnet', repeat=3)
    x6 = separated_conv2D_block(x, filters, size=6, padding='same')
    x = add([x1, x2, x3, x4, x5, x6])
    x = BatchNormalization(axis=-1)(x)
    return x

def separated_conv2D_block(x, filters, size=3, padding='same'):
    x = Conv2D(filters, (1, size), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters, (size, 1), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def midscope_conv2D_block(x, filters):
    kernel_initializer = 'he_normal'
    xOut = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=1)(x)
    xOut = BatchNormalization(axis=-1)(xOut)
    xOut = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=2)(xOut)
    xOut = BatchNormalization(axis=-1)(xOut)
    gap = GlobalAveragePooling2D()(xOut)
    print(f"Shape of the gap vector: {gap.shape}")
    dense_layer = Dense(2, use_bias=False)
    dense_output = dense_layer(gap)
    extract_weight_layer = ExtractWeightLayer(dense_layer)
    weight_vector_output = extract_weight_layer(gap)
    weight_vector_output = tf.expand_dims(weight_vector_output, axis=0)
    print(f"Shape of the weight vector: {weight_vector_output.shape}")
    xOut0 = Multiply()([xOut, weight_vector_output])
    print(f"Shape of the xOut0 map: {xOut0.shape}")
    return xOut0

def widescope_conv2D_block(x, filters):
    xOut = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    xOut = BatchNormalization(axis=-1)(xOut)
    xOut = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=2)(xOut)
    xOut = BatchNormalization(axis=-1)(xOut)
    xOut = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=3)(xOut)
    xOut = BatchNormalization(axis=-1)(xOut)
    print(f"Shape of the xOut map: {xOut.shape}")
    gap = GlobalAveragePooling2D()(xOut)
    print(f"Shape of the gap vector: {gap.shape}")
    dense_layer = Dense(2, use_bias=False)
    dense_output = dense_layer(gap)
    extract_weight_layer = ExtractWeightLayer(dense_layer)
    weight_vector_output = extract_weight_layer(gap)
    weight_vector_output = tf.expand_dims(weight_vector_output, axis=0)
    print(f"Shape of the weight vector: {weight_vector_output.shape}")
    xOut0 = Multiply()([xOut, weight_vector_output])
    print(f"Shape of the xOut0 map: {xOut0.shape}")
    return xOut0

def resnet_conv2D_block(x, filters, dilation_rate=1):
    x1 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x_final = add([x, x1])
    x_final = BatchNormalization(axis=-1)(x_final)
    return x_final

def double_convolution_with_batch_normalization(x, filters, dilation_rate=1):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    return x
