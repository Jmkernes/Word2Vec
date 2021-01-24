import numpy as np
import tensorflow as tf


class SkipGrams:
    """ Takes as input a sequences of length (N,) and outputs
    1) the masked input sequence (N,) where words are dropped
        according to 1-sqrt(t/freq)
    2) the context tensor of shape (N, window_size-1+num_ns) of contexts
        plus no fewer than num_ns negative samples (edge cases are filled
        with negative samples to maintain tensor shape)
    3) the label tensor of 1's and 0's of the same shape as contexts.
    The class is callable.
    """
    def __init__(self, unigrams, distortion=0.75, threshold=1e-5):
        self.unigrams = unigrams
        self.distortion = distortion
        self.vocab_size = len(unigrams)
        self.drop_table = self.initialize_drop_table(self.unigrams, threshold)

    def initialize_drop_table(self, unigrams, threshold):
        drop_probs = tf.maximum(1-tf.math.sqrt(threshold/tf.constant(unigrams, tf.float64)), 0)
        drop_probs = tf.concat([tf.cast([1], drop_probs.dtype), drop_probs], 0) # always drop padding
        keys = tf.range(len(drop_probs))
        init = tf.lookup.KeyValueTensorInitializer(
                    keys, drop_probs, tf.int32, tf.float64)
        return tf.lookup.StaticHashTable(init, 0)

    def set_threshold(self, threshold):
        self.drop_table = self.initialize_drop_table(self.unigrams, threshold)

    def get_targ_context(self, x, window_size):
        x = tf.cast(x, tf.int32)
        N = int(tf.shape(x)[0])
        med = window_size//2
        x = tf.pad(x, [(med, med)])
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.window(window_size, 1, 1).flat_map(lambda w: w.batch(window_size))
        x = next(iter(x.batch(N)))
        inp = x[:, med]
        tar = tf.concat([x[:,:med], x[:,med+1:]], 1)
        return inp, tar

    def get_neg_samples(self, x, num_sampled):
        """Returns a tensor (N,D) of same 0-dimension of x, with D
        negative samples."""
        num_true = int(tf.shape(x)[1])
        neg_samples, _, _ = tf.random.fixed_unigram_candidate_sampler(
            true_classes = tf.cast(x, tf.int64),
            num_true = num_true,
            num_sampled = num_sampled,
            unique=False,
            range_max=self.vocab_size,
            distortion=self.distortion,
            num_reserved_ids=0,
            unigrams=self.unigrams,
            seed=None
        )
        return tf.cast(neg_samples, tf.int32)

    def _replace_zero_with_negs(self, context):
        """ Fills in the zeros of the context with negative samples. """
        ids = tf.where(context==0)
        n_zeros = len(ids)
        labels = tf.sparse.SparseTensor(
            ids, tf.ones(n_zeros), context.shape
        )

        neg_samples = self.get_neg_samples(context, n_zeros)
        neg_samples = tf.sparse.SparseTensor(
            ids, neg_samples, context.shape
        )

        # update the context and labels with banded portions
        labels = tf.cast(1-tf.sparse.to_dense(labels), context.dtype)
        context = tf.sparse.add(context, neg_samples)
        return context, labels


    def __call__(self, x, window_size=11, num_ns=5):
        # get the initial padded context and final targets
        N = int(tf.shape(x)[0])
        targ, context = self.get_targ_context(x, window_size)
        neg_samples = self.get_neg_samples(context, int(num_ns*N))
        neg_samples = tf.cast(tf.reshape(neg_samples, (N, -1)), context.dtype)

        # get the masks
        tar_sample = tf.random.uniform(tf.shape(targ), dtype=tf.float64)
        ctxt_sample = tf.random.uniform(tf.shape(context), dtype=tf.float64)
        tar_mask = tf.less(self.drop_table[tf.cast(targ, tf.int32)], tar_sample)[:, tf.newaxis]
        ctxt_mask = tf.less(self.drop_table[tf.cast(context, tf.int32)], ctxt_sample)
        mask = tf.logical_and(tar_mask, ctxt_mask)
        mask = tf.concat([mask, tf.cast(tf.ones_like(neg_samples), dtype=mask.dtype)], axis=1) # don't mask any neg_samples

        # Finally, concat the additional negative samples and labels
        labels = tf.concat([tf.ones_like(context), tf.zeros_like(neg_samples)], axis=1)
        context = tf.concat([context, neg_samples], axis=1)

        return (targ, context), labels, mask
