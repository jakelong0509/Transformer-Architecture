import numpy as np
import os
import sys
sys.path.append("..")
import helperFunctions as hF
import activationFunctions as aF
import matplotlib.pyplot as plt


class Embedding:
    def __init__(self, input_dim, output_dim, k, dictionary):
        """
            input_dim: dimension of input sample (e.g one-hot of a word)
            output_dim: dimension of output vector (e.g vector representation of a word) = dimension of hidden layer
        """
        self._dic = dictionary
        self._k = k
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._W_input = hF.Xavier((input_dim, output_dim))
        self._W_output = hF.Xavier((output_dim, input_dim))


    def forward(self, x, y_positive_oh, y_negative_oh):
        """
            x: one-hot word representaion (input_dim, batch_size)
            y_positive_oh: one-hot postive target word (no_pos_target, input_dim, batch_size)
            y_negative_oh: one_hot negative target words (no_neg_target, input_dim, batch_size)
        """
        batch_size = x.shape[1]
        h = np.matmul(np.transpose(self._W_input), x)# shape = (output_dim, batch_size)
        v_pos = np.matmul(self._W_output, y_positive_oh) # shape = (no_pos_target, output_dim, batch_size)
        v_neg = np.matmul(self._W_output, y_negative_oh) # shape = (no_neg_target, output_dim, batch_size)
        L = (1/batch_size) * (-np.sum(np.log(aF.sigmoid(v_pos * h)), axis = (1, 0, 2))) - np.sum(np.log(aF.sigmoid(-v_neg * h)), axis = (1, 0, 2)) # sum output_dim 1st, number of target 2nd, batch_size 3rd
        print("Loss: ", L)
        cache = (h, v_pos, v_neg, x, y_positive_oh, y_negative_oh)
        return cache

    def backward(self, cache):
        """
        """
        h, v_pos, v_neg, x, y_positive_oh, y_negative_oh = cache
        batch_size = h.shape[1]
        # shape = (no_pos_target, batch_size)
        d_pos = aF.sigmoid(np.sum(v_pos * h, axis = 1)) - 1

        # shape = (input_dim, no_pos_target, batch_size)*(no_pos_target, batch_size) => swap => (no_pos_target, input_dim, batch_size)
        dvh_pos = np.swapaxes(np.swapaxes(y_positive_oh, 0, 1) * d_pos, 0, 1)

        #shape = (no_neg_target, batch_size)
        d_neg = aF.sigmoid(np.sum(v_neg * h, axis = 1))

        # shape = (input_dim, no_neg_target, batch_size)*(no_neg_target, batch_size) => swap => (no_neg_target, input_dim, batch_size)
        dvh_negative = np.swapaxes(np.swapaxes(y_negative_oh, 0, 1) * d_neg, 0, 1)

        # shape = (no_pos_target + no_neg_target = no_target, input_dim, batch_size)
        dvh = np.append(dvh_pos, dvh_negative, axis=0)

        # shape = (no_target, input_dim, output_dim) => swap => (no_target, output_dim, input_dim) => sum => (output_dim, input_dim)
        dv = np.sum((1/batch_size) * np.swapaxes(np.matmul(dvh, np.transpose(h)), 1,2), axis = 0)

        # shape = (output_dim, no_pos_target, batch_size) * (no_pos_target, batch_size) => (output_dim, no_pos_target, batch_size) => sum axis 1 => (output_dim, batch_size)
        dh = (1/batch_size) * np.sum(np.swapaxes(v_pos, 0, 1) * d_pos, axis = 1) + np.sum(np.swapaxes(v_neg, 0, 1) * d_neg, axis = 1)

        return dv, dh

    def update_weight(self, dv, dh, x, lr = 0.1):
        self._W_output = self._W_output - lr * dv
        end, batch_size = x.shape
        for b in range(batch_size):
            x_index = hF.get_index(0, end-1, x[:,b])
            self._W_input[x_index, :] = self._W_input[x_index, :] - lr * dh[:, b]

if __name__ == "__main__":
    layer = Embedding(10,5,0,0)
    y_pos = np.zeros((1,10,3))
    y_neg = np.zeros((2,10,3))
    x = np.zeros((10,3))
    y_pos[0,4,0] = 1
    y_pos[0,2,1] = 1
    y_pos[0,5,2] = 1
    y_neg[0,0,0] = 1
    y_neg[0,1,1] = 1
    y_neg[0,3,2] = 1
    y_neg[1,7,0] = 1
    y_neg[1,8,1] = 1
    y_neg[1,9,2] = 1
    x[6,0] = 1
    x[3,1] = 1
    x[8,2] = 1
    for l in range(100000):
        cache = layer.forward(x, y_pos, y_neg)

        h, v_pos, v_neg, x, y_positive_oh, y_negative_oh = cache
        dv, dh = layer.backward(cache)
        layer.update_weight(dv,dh,x)
