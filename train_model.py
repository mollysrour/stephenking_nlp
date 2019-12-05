from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import os
import time
import nltk
import keras
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
import re
import random

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model

if __name__ == "__main__":
    path_to_file = 'StephenKing-Sorted.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    tokenizer = Tokenizer()
    corpus = text.split("\n")
    random.shuffle(corpus)
    random.shuffle(corpus)
    random.shuffle(corpus)
    proc_corpus = []
    for sent in corpus:
        sent = re.sub(r'\d+', '', sent)
        sent = re.sub(r'\_', '', sent)
        sent = re.sub(r'\"','', sent.strip())
        sent = re.sub(r'\$','', sent.strip())
        sent = re.sub(r'\'','', sent.strip())
        sent = re.sub(r'\¢|\®|\°|\±|\¼|\½|\Ç|\É|\Ô|\×|\Ü|\à|\á|\â|\ä|\ç|\è|\é|\ê|\ë|\ì|\í|\î|\ï|\ñ|\ó|\ô|\ö|\ø|\ú|\û|\ü|\ā|\ē|\ō|\ř|\Γ|\•|\™|\⅓|\♥', '', sent.strip())
        tok_sent = nltk.word_tokenize(sent)
        tok_sent = [i for i in tok_sent if i] # remove empty string tokens
        tok_sent = [i for i in tok_sent if len(i) > 0]
        proc_corpus.append(' '.join(w for w in tok_sent))    
    tokenizer.fit_on_texts(proc_corpus)
    total_words = len(tokenizer.word_index) + 1
    text = ' '.join(w for w in proc_corpus)
    # length of text is the number of characters in it
    print ('Length of text: {} characters'.format(len(text)))

        # Creating a mapping from unique characters to indices
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64
    # Length of the vocabulary in chars
    vocab_size = 74

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(vocab_size = len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE, seq_length=seq_length)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Directory where the checkpoints will be saved
    checkpoint_dir = 'training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    EPOCHS = 10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])