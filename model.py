"""Tensorflow model for the warp operation"""
import tensorflow as tf

mp_sz = 96


class MyModel(tf.keras.Model):
    def __init__(self):
        """
        The __init__ function sets up the layers of our model.

        :param self: Represent the instance of the class
        :return: Nothing
        """
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5))
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5))
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.convo = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

    def call(self, inputs, training=None, mask=None):
        """
        The call function is the main function of a layer.
        It takes as input a tensor or list of tensors and returns a tensor.

        :param self: Represent the instance of the class
        :param inputs: Pass the input data to the model
        :param training: Control the use of dropout
        :param mask: Mask the input tensor
        :return: The output of the model
        """
        x = tf.image.resize(inputs, [mp_sz, mp_sz])
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.convo(x)
        return x
