import tensorflow as tf
from tensorflow.keras import layers
from ..registry import HEADS

@HEADS.register_module
class MaskHead(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Convolution block 1
        self.conv_1 = layers.Conv2D(256, (3, 3), padding="same", name="mask_conv_1")
        self.bn_1 = layers.BatchNormalization(name='mask_bn_1')
        self.activation_1 = layers.ReLU()
        # Convolution block 2
        self.conv_2 = layers.Conv2D(256, (3, 3), padding="same", name="mask_conv_2")
        self.bn_2 = layers.BatchNormalization(name='mask_bn_2')
        self.activation_2 = layers.ReLU()
        # Convolution block 3
        self.conv_3 = layers.Conv2D(256, (3, 3), padding="same", name="mask_conv_3")
        self.bn_3 = layers.BatchNormalization(name='mask_bn_3')
        self.activation_3 = layers.ReLU()
        # Convolution block 4
        self.conv_4 = layers.Conv2D(256, (3, 3), padding="same", name="mask_conv_4")
        self.bn_4 = layers.BatchNormalization(name='mask_bn_4')
        self.activation_4 = layers.ReLU()
        # Deconv to 28x28
        self.deconv = layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu",
                                             name="mask_deconv")
        self.masks = layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid", name="mask")
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        masks = []
        pooled_rois_list = inputs
        for pooled_rois in pooled_rois_list:
            x = self.conv_1(pooled_rois)
            x = self.bn_1(x)
            x = self.activation_1(x)
            x = self.conv_2(x)
            x = self.bn_2(x)
            x = self.activation_2(x)
            x = self.conv_3(x)
            x = self.bn_3(x)
            x = self.activation_3(x)
            x = self.conv_4(x)
            x = self.bn_4(x)
            x = self.activation_4(x)
            x = self.deconv(x)
            masks.append(self.masks(x))
        return masks