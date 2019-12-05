from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import json
import os
import time
import keras
import re
import random
import numpy as np

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


def get_model(vocab_size, embedding_dim, rnn_units, batch_size):
    #tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = 'training_checkpoints/ckpt_3'
    
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(checkpoint)

    model.build(tf.TensorShape([1, None]))
    
    return model

def generate_text(model, char2idx, idx2char, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 500

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.75

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def load_model():

    path_to_file = 'StephenKing-Processed.txt'
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Length of the vocabulary in chars
    vocab_size = 74

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = get_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    return model, char2idx, idx2char

def predicting(model, char2idx, idx2char, textinput):
    
    out = generate_text(model, char2idx, idx2char, start_string=textinput)

    json_dict = {"Input": textinput, "Output": out}
    output = json.dumps(json_dict)
    return output    
