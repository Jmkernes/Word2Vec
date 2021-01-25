# Word2Vec
<em>Implementation of the paper "Distributed representation of words and phrases and their compositionality":</em> [link](https://arxiv.org/pdf/1310.4546.pdf)

## Quick start
A pretrained model and its info is saved in the folder "Embeddings". To get started playing around with those learned embeddings, simply run the find_analogies script
```
python find_analogies.py
```
This will open a terminal based program that allows you to enter in analogies of the form
"A is to B as C is ___?"
and the model will return the top 3 (3 is the default it can be changed by changing K= in the source code) most likely words.
You also have the option of finding the 5 "closest" words to any given word. For example, entering Obama will return words like "president" "presidential" etc.

## Pretrained model
The embeddings folder consists of 

1) a tensorflow .savedmodel file w2v_model containing the saved word2vec model. This can be loaded using ```python tf.keras.models.load_model(FILE)```

2) a configuration json file called configs. Lower-level, this allows you to recreate the model architecture described in the .savedmodel file. Can be used in combo with saved weights to build an identical model then load in the weights.

3) a unigrams json file called unigrams. This is just the learned unigrams and their frequencies in the corpus. Can be used to rebuild the tokenizer and skipgram generator if needed.

## Training your own embeddings
The training was done on the wikitext2 training dataset. There is very minimal datacleaning done in the project. To train you own embeddings, do the following:
Tokenize your dataset, using spacing to delineate tokens. Example - "wow, it's hot!" would tokenize to "wow , it 's hot !". Save this in a .txt file.

You can run the train.py script, and use the flag --input_text=YOUR-FILE-HERE to start training. The shell script base_model contains a number of default parameters,
including batch size, window size, number of negative samples, epochs, the usual. If running in the terminal, you can view additional training progress via the command ```tensorboard --logdir logs```

## What is the model doing?
The model is basically skipgrams plus negative sampling with word dropout. The main task is given a target word in a sentence, to predict its surrounding context words within some window size.
Window size is the full window, so a size of 11 means to include the left 5 and right 5 words. To speed up computation, we use negative sampling, where the loss is sparse sigmoid cross entropy, i.e. predicting the probability of any class indepednently,
with respect to num_ns negative samples drawn randomly. The negative samples are drawn from a distribution $P(x) \sim f(x)^{3/4}$, proportional to the 3/4 power of any tokens frequency in the corpus.
We also implement token dropping according to the formula $P_\text{drop}(x) = 1-\sqrt{t/f(x)}$, where the threshold t is a hyperparameter with default $t=10^{-5}$ and $f$ is again token frequency.
Words with frequency above $t$ will usually be dropped, and below $t$ kept. You should aim to set this to be at around the 20th percentile of most common occuring tokens.
This helps learn rare words and can speed up training. However, if the threshold is too low the model discards too many words, so we give the option of annealing the threshold. You can choose a high value like $t=0.1$, then lower $t$ during training.

The word dropping is done via masking to maintain batch size and enable on-the-fly skigram generation. Any target or context tokens that are dropped are replaced with the pad toke nand asked out.
To keep memory down, skipgrams are generated on the fly during training. 

The program first pads sentences with window_size//2 pad tokens, then inserts <sos> and <eos> tags for each sentence. It reads in a line, then computes the skipgrams for that line, and saves the target, context, labels and mask to running variables.
We then pull batch_size sized pieces from the running tallies and yield those to the model for training, until the running variables are of size less than batch_size. We then continue to the next line and repeat.
In this way, the data is different every run due to the randomness of dropping tokens. It's slower, but much much much more memory efficient to generate skipgrams on the fly.
