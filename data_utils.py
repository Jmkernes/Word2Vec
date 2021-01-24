import logging
import tensorflow as tf
import numpy as np
from skipgrams import SkipGrams
from functools import partial

def is_not_title(x):
    cond1 = tf.logical_not(tf.strings.regex_full_match(x,'\s[=].*[=]\s'))
    cond2 = tf.strings.length(x) > 1
    return tf.logical_and(cond1, cond2)

def sos_eos(x, med=5):
    pad_string = ' <pad>'*5+' <sos>'
    x = tf.strings.regex_replace(x, '\Q .\E', ' . <eos>'+pad_string)
    x = tf.strings.regex_replace(x, '\Q ?\E', ' ? <eos>'+pad_string)
    x = tf.strings.regex_replace(x, '\Q !\E', ' ! <eos>'+pad_string)
    x = tf.strings.regex_replace(x, '^', '<sos>')
    x = tf.strings.regex_replace(x, pad_string+' $', '')
    return x

def get_words_freqs(ds):
    print("Getting dataset size...")
    size = ds.reduce(0, lambda x,_: x+1)
    print(f"There are {size} total lines in the dataset.")
    all_tokens = []
    for i,x in enumerate(ds):
        all_tokens.append(x)
        print(f"Extracting all tokens: {100*(i+1)/size:.2f}% completed.", end='\r')
    words, _, freqs = tf.unique_with_counts(tf.concat(all_tokens, 0))
    print("Sorting words by frequency...")
    word_order = tf.argsort(freqs, direction='DESCENDING')
    words = tf.gather(words, word_order)
    freqs = tf.gather(freqs, word_order)
    return words, freqs

def get_keys_unigrams(words, freqs):
    print("Creating keys and unigrams...")
    keys = [b'<pad>', b'<sos>', b'<eos>', b'<unk>']
    unigrams = [0.,0.,0.,0.]
    for w,f in zip(list(words.numpy()), list(freqs.numpy())):
        if w==b'<pad>':
            unigrams[0] = f
        elif w==b'<sos>':
            unigrams[1] = f
        elif w==b'<eos>':
            unigrams[2] = f
        elif w==b'<unk>':
            unigrams[3] = f
        else:
            keys.append(w)
            unigrams.append(f)
    unigrams = np.array(unigrams)
    unigrams = unigrams/np.sum(unigrams)
    unigrams = unigrams[1:]
    return keys, unigrams

def create_vocab_tables(keys):
    print("Initializing lookup tables...")
    init = tf.lookup.KeyValueTensorInitializer(
    keys, np.arange(len(keys)), key_dtype=tf.string, value_dtype=tf.int64)
    vocab_table = tf.lookup.StaticVocabularyTable(init, 1)
    init = tf.lookup.KeyValueTensorInitializer(
        tf.range(len(keys)), keys, tf.int32, tf.string)
    inv_vocab_table = tf.lookup.StaticHashTable(init, '<unk>')
    return vocab_table, inv_vocab_table

def fit_vocab(ds, med=5):
    words, freqs = get_words_freqs(ds)
    keys, unigrams = get_keys_unigrams(words, freqs)
    vocab_table, inv_vocab_table = create_vocab_tables(keys)
    print("Finished.")
    return vocab_table, inv_vocab_table, unigrams, keys

class DataManager:
    def __init__(self, dataset, skipgram, vocab_table, inv_vocab_table, window_size, num_tokens=None):
        self.ds = dataset
        self.vocab_table = vocab_table
        self.inv_vocab_table = inv_vocab_table
        self.skg = skipgram
        self.window_size = window_size
        if num_tokens is None:
            self.num_tokens = self.find_num_tokens(self.ds)

    @classmethod
    def from_text_file(cls, file, window_size=11, threshold=1e-4):
        assert file.endswith('.txt')
        assert window_size%2
        print(f"Window size: {window_size}. Threshold: {threshold:.2e}")
        med = window_size//2

        raw_ds = tf.data.TextLineDataset('train.txt')
        text_ds = raw_ds.filter(is_not_title)
        text_ds = text_ds.map(tf.strings.lower)
        text_ds = text_ds.map(lambda x: sos_eos(x, med))
        text_ds = text_ds.map(tf.strings.split)

        vocab_table, inv_vocab_table, unigrams, keys = fit_vocab(text_ds)
        tokenized_ds = text_ds.map(vocab_table.lookup)
        skipgram = SkipGrams(unigrams, threshold)

        config = {'dataset':tokenized_ds, 'skipgram':skipgram, 'window_size':window_size,
         'vocab_table':vocab_table, 'inv_vocab_table':inv_vocab_table}
        return cls(**config)
        # return cls(tokenized_ds, skipgram, vocab_table, inv_vocab_table)

    def tokenize(self, x):
        x = tf.strings.split(tf.strings.lower(x))
        return self.vocab_table[x]

    def detokenize(self, x):
        x = self.inv_vocab_table[x]
        return tf.strings.reduce_join(x, separator=' ')

    def find_num_tokens(self, ds):
        print("Finding number of tokens...")
        tot_tokens = 0
        for x in ds:
            tot_tokens += x.shape[0]
        print(f"Total number of tokens: {tot_tokens}")
        return tot_tokens

    def find_ds_size(self, ds):
        return ds.reduce(0, lambda x,y: x+1)

    def _batched_generator(self, batch_size, num_ns, shuffle_buffer_size, window_size=None):
        if window_size is None:
            window_size = self.window_size
        else:
            print(f"WARNING: the built-in window_size is {self.window_size}."
            "You may still iterate over a different window size, but the end"
            "of sentence padding will not match. Therefore, there may be cross-"
            "sentence overlap. if window_size > built-in value.")

        ds_size = self.num_tokens//batch_size
        feature_dim = window_size-1+num_ns

        inp = tf.zeros((1,), dtype=tf.int32)
        ctxt = tf.zeros((1, feature_dim), dtype=tf.int32)
        lbl = tf.zeros((1, feature_dim), dtype=tf.int32)
        mask = tf.zeros((1, feature_dim), dtype=tf.bool)

        if shuffle_buffer_size is not None:
            self.ds = self.ds.shuffle(shuffle_buffer_size)

        for x in self.ds:
            (inp_new, ctxt_new), lbl_new, mask_new = self.skg(x, window_size, num_ns)

            inp = tf.concat([inp, inp_new], 0)
            ctxt = tf.concat([ctxt, ctxt_new], 0)
            lbl = tf.concat([lbl, lbl_new], 0)
            mask = tf.concat([mask, mask_new], 0)

            while tf.shape(inp)[0] > batch_size:
                x_b =inp[:batch_size]
                ctxt_b = ctxt[:batch_size]
                lbl_b = lbl[:batch_size]
                mask_b = mask[:batch_size]

                yield (x_b, ctxt_b), lbl_b, mask_b

                inp =inp[batch_size:]
                ctxt = ctxt[batch_size:]
                lbl = lbl[batch_size:]
                mask = mask[batch_size:]

    def batched_ds(self, batch_size, num_ns, shuffle_buffer_size=None, window_size=None):
        """ Usually would shuffle the final dataset, but that will be very expensive memory-wise.
        It's much better to shuffle the sequence data before computing skipgrams. But, this requires
        the awkward API where we add shuffle as a parameter """
        generator = partial(self._batched_generator, batch_size=batch_size, num_ns=num_ns,
                        window_size=window_size, shuffle_buffer_size=shuffle_buffer_size)
        window_size = self.window_size if window_size is None else window_size
        feature_dim = window_size-1+num_ns
        output_signature=(
                (tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, feature_dim), dtype=tf.int32)),
                tf.TensorSpec(shape=(batch_size, feature_dim), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, feature_dim), dtype=tf.bool)
        )
        ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return ds
