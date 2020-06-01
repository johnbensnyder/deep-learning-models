import tensorflow as tf
from tensorflow.keras import layers
from ..registry import HEADS

@HEADS.register_module
class MaskHead(tf.keras.Model):
    def __init__(self, num_classes, depth=4):
        super().__init__()
        self.depth = depth
        input_shape = tf.keras.Input([28, 28, 1])
        self.conv_0 = tf.keras.layers.Conv2D(256, (3, 3), 
                                            padding="same", activation='relu', 
                                            name="mask_conv_0")
        for layer in range(1, self.depth):
            self.__dict__['conv_{}'.format(layer)] = tf.keras.layers.Conv2D(256, (3, 3), 
                                                    padding="same", activation='relu', 
                                                    name="mask_conv_{}".format(layer))
        self.deconv = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, 
                                                    activation="relu",
                                                    name="mask_deconv")
        self.masks = tf.keras.layers.Conv2D(num_classes, (1, 1), 
                                                strides=1, activation="sigmoid", name="mask")
            
    @tf.function(experimental_relax_shapes=True)
    def call(self, mask_rois, training=True):
        for layer in range(self.depth):
            mask_rois = self.__dict__['conv_{}'.format(layer)](mask_rois)
        mask_rois = self.deconv(mask_rois)
        return self.masks(mask_rois)