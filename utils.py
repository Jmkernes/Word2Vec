import time
import numpy as np
import tensorflow as tf
from functools import partial

def print_bar(step, tot, diff, loss):
    num_eq = int(10*(step+1)/tot)
    num_pd = 10-num_eq
    bar = '['+'='*num_eq+'>'+'.'*num_pd+']'
    time_left = (tot-step)*diff
    m = int(time_left)//60
    s = int(time_left)%60
    iter_message = f"{step+1:02d}/{tot}:"
    time_message = f"{1/diff:.2f} it/s. Est: {m:02d}m {s:02d}s"
    loss_message = f"Loss: {loss:.3f}"
    end = '\r' if step<tot-1 else '\n'
    print(iter_message, bar, time_message, loss_message, end=end)

def cossim(u, v):
    unorm = tf.linalg.norm(u)
    vnorm = tf.linalg.norm(v)
    return u@v/(unorm*vnorm)

def get_token_id(word, vocab_table):
    x = vocab_table[tf.constant(word)]
    oov = vocab_table[tf.constant('@#$%@#(@)%#$')]
    if tf.equal(x, oov):
        print(f"The word {word} is not in the dictionary!")
        return int(vocab_table[tf.constant('<unk>')])
    return int(x)

def find_analogy(wordA, wordB, wordC, U, vocab_table, inv_vocab_table, K=10):
    a, b, c = map(lambda x: get_token_id(x, vocab_table), (wordA, wordB, wordC))
    v = U[c]-U[a]+U[b]
    Unorm = np.linalg.norm(U, axis=1)
    probs = (U@v)/Unorm
    probs[[a,b,c]] = 0 # prevent from looking at self similarity
    topK = np.argpartition(probs, -K)[-K:]
    topK = sorted(topK, key=lambda x: -probs[x])
    words = inv_vocab_table[tf.cast(topK, tf.int32)].numpy()
    return words

def find_closest(word, U, vocab_table, inv_vocab_table, K=10):
    a = get_token_id(word, vocab_table)
    v = U[a]
    Unorm = np.linalg.norm(U, axis=1)
    probs = (U@v)/Unorm
    probs[a] = 0 # prevent from looking at self similarity
    topK = np.argpartition(probs, -K)[-K:]
    topK = sorted(topK, key=lambda x: -probs[x])
    words = inv_vocab_table[tf.cast(topK, tf.int32)].numpy()
    return words

def print_analogies(a, b, c, U, vocab_table, inv_vocab_table, K=3):
    """ prints out top K guesses for a->b as c->???. Takes a DataManager as input"""
    print(f"{a} is to {b}, as {c} is to ___?")
    for i, word in zip(range(K), find_analogy(a, b, c, U, vocab_table, inv_vocab_table, K)):
        print(f"\tOption {i+1}: {str(word)}")

def print_closest(x, U, vocab_table, inv_vocab_table, K=3):
    """ prints out top K closest cossim vectors. Takes a DataManager as input """
    print(f"The nearest neighbors to {x} are:")
    for i, word in zip(range(K), find_closest(x, U, vocab_table, inv_vocab_table, K)):
        print(f"\tNeighbor {i+1}: {word}")
