import numpy as np
import tensorflow as tf
import logging
import sys


logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class GlobalMLP(object):

    def __init__(self, params):
        super(GlobalMLP, self).__init__()

        self.LSTM_Layers = params['LSTM_layers']
        self.hidden_size = params['hidden_size']
        self.embedding_size = params['embedding_size']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

    def initialize_global_rnn(self):
        with tf.compat.v1.variable_scope("global_policy_step", reuse=tf.compat.v1.AUTO_REUSE):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True,
                                                               state_is_tuple=True))
            global_rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        return global_rnn

    def initialize_global_mlp(self):
        global_hidden_layer = tf.keras.layers.Dense(4 * self.hidden_size, activation='relu')
        global_output_layer = tf.keras.layers.Dense(self.m * self.embedding_size, activation='relu')
        return global_hidden_layer, global_output_layer

