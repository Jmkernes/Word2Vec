import time
import numpy as np
import tensorflow as tf

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, d_model=128, use_act=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.U = tf.keras.layers.Embedding(vocab_size, d_model, name='U')
        self.V = tf.keras.layers.Embedding(vocab_size, d_model, name='V')
        self.tanh = tf.keras.layers.Lambda(tf.math.tanh)
        self.use_act = use_act

    def call(self, pair):
        inp, tar = pair
        inp = tf.squeeze(inp)
        u = self.U(inp)
        v = self.V(tar)
        if self.use_act:
            u = self.tanh(u)
        logits = tf.einsum('ik,ijk->ij', u, v)
        return logits

def loss_fn(y_true, y_pred, mask):
    y_true = tf.cast(y_true, y_pred.dtype)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    loss = tf.boolean_mask(loss, mask)
    return tf.reduce_mean(loss)
