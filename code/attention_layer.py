from keras import backend as K
from keras.layers import Layer

class Attention(Layer):
    def __init__(self, return_sequences=False):
        super(Attention, self).__init__()
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

''' In the above code, the Attention layer is defined as a custom layer by subclassing the `Layer` class from Keras.
    It calculates attention weights for each time step in the input sequence and
    applies those weights to the input sequence. '''