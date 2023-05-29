import tensorflow as tf

mp_sz = 96


class MyModel(tf.keras.Model):
    def __init__(self):
        """
        The __init__ function is called when the object is created.
        It sets up the layers of our model, and defines how they are connected to each other.
        The first two layers are convolutional layers with 64 filters of size 5x5, followed by a LeakyReLU activation function.
        The third layer is also a convolutional layer with 64 filters of size 5x5, followed by another LeakyReLU activation function.

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
        The call function is the main function of a layer. It takes as input a tensor or list of tensors and returns
        a tensor or list of tensors. The call function can be overridden when writing custom layers.

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
