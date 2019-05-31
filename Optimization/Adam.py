import numpy as np

class Adam:
    def __init__(self, layer, gradients, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, bias = True):
        self._layer = layer
        self._gradients = gradients
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._bias = bias

        self._v_corrected = {}
        self._v_weight = {}

        self._s_corrected = {}
        self._s_weight

        if self._bias:
            self._v_bias = {}
            self._s_bias = {}

    def update_weight(self, i):
        self._lr = self._lr * np.sqrt(1 - self._beta2**i) / (1 - self._beta1**i)
