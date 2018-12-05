from keras import backend as K
from keras.engine.topology import Layer

class RelaxedSoftmax(Layer):

    def __init__(self, **kwargs):
        super(RelaxedSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        super(RelaxedSoftmax, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        logits, temperature = x
        merged_logits = logits * temperature
        return merged_logits

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a
