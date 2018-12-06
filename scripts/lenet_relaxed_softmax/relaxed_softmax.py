from keras import backend as K
from keras.engine.topology import Layer
from keras import activations

class RelaxedSoftmax(Layer):

    def __init__(self, **kwargs):
        self.activation = activations.get('softmax')
        super(RelaxedSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        super(RelaxedSoftmax, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # assert isinstance(x, list)
        # logits = x
        merged_logits = x
        relaxed_softmax_ouputs = self.activation(merged_logits)
        return relaxed_softmax_ouputs

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        shape_a = input_shape
        return shape_a
