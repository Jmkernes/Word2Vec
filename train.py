import os
import time
import json
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skipgrams import SkipGrams
from data_utils import DataManager
from utils import *
from model import Word2Vec, loss_fn
from functools import partial

from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_integer('window_size', default=11, help='window size. if 11, will keep 5 skipgrams to the left and 5 to the right.')
flags.DEFINE_string('text_file', default='train.txt', help='Text to train on.')
flags.DEFINE_integer('batch_size', default=128, help='training batch size.')
flags.DEFINE_integer('num_ns', default=7, help='Number of negative samples per target, context pair.')
flags.DEFINE_integer('epochs', default=5, help='Number of training epochs.')
flags.DEFINE_float('threshold', default=5e-5, help='The drop probability threshold. Word frequencies above this will likely\
    be dropped, below kept. Aim for probs to be about the frequency of words at the top ~20% occurring.')
flags.DEFINE_string('weights_directory', default='embedding_weights', help='Name of output directory to save the vocab table,\
weights, and unigrams.')
flags.DEFINE_integer('d_model', default=128, help='Model embedding dimension.')
flags.DEFINE_float('learning_rate', default=1e-3, help='learning_rate')
flags.DEFINE_integer('buffer_size', default=5000, help='Buffer size for shuffling data set.\
    This shuffles the input sequences, not the computed skipgrams.')
flags.DEFINE_boolean('anneal_threshold', default=False, help='Every 1000 steps this will decrease the drop threshold by a factor\
    of 10, to a minimum value of 1e-5.')

def test_common_analogies(model, dm):
    U = model.layers[0].get_weights()[0]
    V = model.layers[1].get_weights()[0]
    W_emb = 0.5*(U+V)
    func = partial(print_analogies, U=W_emb, vocab_table=dm.vocab_table, inv_vocab_table=dm.inv_vocab_table, K=3)
    analogies = [('he', 'she', 'him'),('see', 'saw', 'hear'),('m', 'km', 'ft'),('possible', 'impossible', 'able'),
     ('week', 'day', 'year'), ('daughter', 'son', 'mother')]
    print('~'*20+' Analogy task '+'~'*20)
    for x in analogies:
        func(*x)
        print('-'*40)

def write_vocab_to_json(dm):
    probs = [0.]+list(dm.skg.unigrams)
    words = tf.strings.split(dm.detokenize(tf.range(dm.skg.vocab_size+1))).numpy()
    words = [x.decode() for x in words]
    d = dict(zip(words, probs))
    filename = FLAGS.weights_directory+'/unigrams.json'
    logging.info(f"Writing unigrams and probabilities to: {filename}")
    with open(filename, 'w') as file:
        file.write(json.dumps(d))


def main(argv):
    try:
        os.mkdir(FLAGS.weights_directory)
    except:
        pass

    # Load data
    logging.info(f"\n\nLoading data from {FLAGS.text_file}...\n\n")
    dm = DataManager.from_text_file(FLAGS.text_file, FLAGS.window_size, threshold=FLAGS.threshold)

    # metrics
    train_loss = tf.keras.metrics.Mean()
    VOCAB_SIZE = dm.skg.vocab_size+1
    history = {'loss':[]}

    # model
    logging.info(f"\n\nConstructing model. d_model: {FLAGS.d_model}. vocab_size: {VOCAB_SIZE}\n\n")
    tf.keras.backend.clear_session()
    model = Word2Vec(vocab_size=VOCAB_SIZE, d_model=FLAGS.d_model)
    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    @tf.function
    def train_step(inp, ctxt, lbl, mask):
        with tf.GradientTape() as tape:
            pred = model((inp, ctxt))
            loss = loss_fn(lbl, pred, mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

    # Checkpointing
    ckpt_path = FLAGS.weights_directory+'/checkpoints'
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(1))
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

    DS_SIZE = dm.num_tokens//FLAGS.batch_size
    train_ds = dm.batched_ds(FLAGS.batch_size, FLAGS.num_ns, FLAGS.buffer_size).prefetch(1)

    # Restore checkpoints
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}")
    else:
        logging.info("Initializing from scratch.")

    total_start = time.time()
    for epoch in range(FLAGS.epochs):
        logging.info(f"\n\n----- Epoch {epoch+1}/{FLAGS.epochs} -----")
        train_loss.reset_states()
        start = time.time()
        for step, ((inp, ctxt), lbl, mask) in enumerate(train_ds):

            train_step(inp, ctxt, lbl, mask)
            loss = train_loss.result().numpy()
            diff = (time.time()-start)/(step+1)
            history['loss'].append(loss)
            print_bar(step, DS_SIZE, diff, loss)
            ckpt.step.assign_add(1)

            # we start the drop threshold off high to get some training for
            # common words, then decrease it over training to learn rare words
            if FLAGS.anneal_threshold and (step+1)%500==0 and FLAGS.threshold > 1e-5:
                FLAGS.threshold /= 10
                dm.skg.set_threshold(FLAGS.threshold)

            if (step+1)%1000==0:
                logging.info(f"~~~~~~ Step: {step+1} ~~~~~~")
                test_common_analogies(model, dm)

            if (step+1)%5000==0:
                save_path = ckpt_manager.save()
                logging.info("Saved checkpoint for step {step} to: {save_path}")

    logging.info(f"\n\n\nCompleted training. Total time: {time.time()-total_start:.2f}s\n\n\n")
    filename = FLAGS.weights_directory+'/metrics.json'
    logging.info(f"\n\nSaving training metrics to: {filename}")
    with open(filename, 'w') as file:
        file.write(json.dumps(history))
    filename = FLAGS.weights_directory+'/weights.h5'
    logging.info(f"\n\nSaving model to: {filename}")
    model.save(filename)

if __name__=="__main__":
    app.run(main)
