import numpy as np
from .. import helperFunctions as hF

class EmbeddingLayer:
    def __init__(self, input_dim, output_dim, no_context, batch_size, bias = True):
        """
            input_dim: dimension of input sample (e.g one-hot of a word)
            output_dim: dimension of output vector (e.g vector representation of a word) = dimension of hidden layer
        """
        self._no_context = no_context
        self._bias = bias
        self._batch_size = None
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._W_input = hF.Xavier((input_dim, output_dim))
        self._W_output = hF.Xavier((output_dim, input_dim))
        self._W = {"W1": self._W_input, "W2": self._W_output}
        if bias:
            self._b_input = np.ones((output_dim, 1))
            self._b_output = np.ones((input_dim, 1))
            self._b = {"b1": self._b_input, "b2": self._b_output}

    def forward(self, x):
        """
            x: one-hot word representaion (input_dim, batch_size)
        """
        self._batch_size = x.shape[1]
        for i in range(2):
            x = np.matmul(np.transpose(self._W["W"+str(i+1)]), x)
            if self._bias:
                x = x + self._b["b"+str(i+1)]

        y = softmax(x)
