import unittest
import numpy as np
import tensorflow as tf
import time
from skipgrams import *
from model import *

class Tests(unittest.TestCase):


  def test_get_targ_context(self):
      vocab_size = 50
      unigrams = list(1/1.35**np.arange(12, 12+vocab_size, dtype=np.float32))
      skg = SkipGrams(unigrams)
      tar, context = skg.get_targ_context(tf.cast(tf.range(1,8), tf.int32), 7)
      ctxt = tf.constant([[0, 0, 0, 2, 3, 4],
                          [0, 0, 1, 3, 4, 5],
                          [0, 1, 2, 4, 5, 6],
                          [1, 2, 3, 5, 6, 7],
                          [2, 3, 4, 6, 7, 0],
                          [3, 4, 5, 7, 0, 0],
                          [4, 5, 6, 0, 0, 0]], dtype=context.dtype)
      self.assertTrue( tf.reduce_all( tf.equal(tar, tf.constant([1,2,3,4,5,6,7])) ) )
      self.assertTrue( tf.reduce_all(tf.equal(context, ctxt)) )

  def test_neg_samples(self):
      vocab_size = 50
      num_ns = 5
      unigrams = list(1/1.35**np.arange(12, 12+vocab_size, dtype=np.float32))
      skg = SkipGrams(unigrams)
      ctxt = tf.constant([[0, 0, 0, 2, 3, 4],
                          [0, 0, 1, 3, 4, 5],
                          [0, 1, 2, 4, 5, 6],
                          [1, 2, 3, 5, 6, 7],
                          [2, 3, 4, 6, 7, 0],
                          [3, 4, 5, 7, 0, 0],
                          [4, 5, 6, 0, 0, 0]], dtype=tf.int32)
      N = tf.shape(ctxt)[0]
      num_sampled = N*num_ns
      self.assertTrue( skg.get_neg_samples(ctxt, num_sampled).shape[0]==num_sampled )


  def test_replace_zero_with_negs(self):
      vocab_size = 50
      window_size = 7
      unigrams = list(1/1.35**np.arange(12, 12+vocab_size, dtype=np.float32))
      skg = SkipGrams(unigrams)
      ctxt = tf.constant([[0, 0, 0, 2, 3, 4],
                          [0, 0, 1, 3, 4, 5],
                          [0, 1, 2, 4, 5, 6],
                          [1, 2, 3, 5, 6, 7],
                          [2, 3, 4, 6, 7, 0],
                          [3, 4, 5, 7, 0, 0],
                          [4, 5, 6, 0, 0, 0]], dtype=tf.int32)
      N = ctxt.shape[0]
      ctxt, lbls = skg._replace_zero_with_negs(ctxt)
      self.assertTrue( ctxt.shape[0]==N )
      self.assertTrue( lbls.shape==ctxt.shape )

  def test_skipgrams_call(self):
      vocab_size = 50
      window_size = 7
      num_ns = 5
      unigrams = list(1/1.35**np.arange(12, 12+vocab_size, dtype=np.float32))
      skg = SkipGrams(unigrams)
      seq = tf.constant([22, 31, 15, 1, 39, 7, 4, 47, 33, 3, 42, 11, 30,
                         5, 1, 28, 19, 3],
                        dtype=tf.int64)
      (tar, ctxt), lbls, mask = skg(seq, window_size, num_ns)
      N = tar.shape[0]
      self.assertTrue (ctxt.shape[0]==N )
      self.assertTrue( ctxt.shape==lbls.shape )
      self.assertTrue( mask.shape==lbls.shape )
      self.assertTrue( lbls.shape[1]==(window_size+num_ns-1) )

  def test_word2vec(self):
     vocab_size = 100
     d_model = 32
     batch_size = 8
     word2vec = Word2Vec(vocab_size, d_model)
     inp = tf.random.uniform((batch_size, 1), 0, vocab_size, tf.int32)
     tar = tf.random.uniform((batch_size, 20), 0, vocab_size, tf.int32)
     labels = tf.random.uniform((batch_size, 20), 0, vocab_size, tf.int32)

     pred = word2vec((inp, tar))
     mask = tf.cast(tf.random.uniform(tf.shape(labels), 0, 2, tf.int32), tf.float32)
     self.assertTrue(pred.shape==(batch_size, 20))
     loss = loss_fn(labels, pred, mask)
     self.assertTrue(loss>=0)


if __name__=="__main__":
    unittest.main()
