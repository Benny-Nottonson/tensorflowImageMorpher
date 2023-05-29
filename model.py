"""Tensorflow model for the warp operation"""
import tensorflow as tf

MAP_SIZE = 96


class MyModel(tf.keras.Model):
    """Main model class"""
    def __init__(self):
        """
        The __init__ function sets up the layers of our model.

        :param self: Represent the instance of the class
        :return: Nothing
        """
        super().__init__()
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
        tensor = tf.image.resize(inputs, [MAP_SIZE, MAP_SIZE])
        tensor = self.conv1(tensor)
        tensor = self.act1(tensor)
        tensor = self.conv2(tensor)
        tensor = self.act2(tensor)
        tensor = self.convo(tensor)
        return tensor
