# music_utils.py
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import backend as K # Keras is now part of TensorFlow
from tensorflow.keras.layers import RepeatVector
import sys
# music21 and other imports remain the same
from music21 import * 
import numpy as np
from grammar import *
from preprocess import *
from qa import *


def data_processing(corpus, values_indices, m = 60, Tx = 30):
    """
    Processes the corpus into input (X) and output (Y) one-hot encoded matrices
    for sequence-to-sequence model training.
    
    Arguments:
    corpus -- list of unique musical values (e.g., 'C,0.250')
    values_indices -- dict mapping unique values to their integer index
    m -- number of training examples to generate
    Tx -- length of each sequence (time steps)
    
    Returns:
    X -- (m, Tx, N_values) numpy array of one-hot encoded input sequences
    Y -- (Tx, m, N_values) numpy array of one-hot encoded target sequences (time-shifted X)
    N_values -- number of unique values in the corpus
    """
    # cut the corpus into semi-redundant sequences of Tx values
    Tx = Tx 
    N_values = len(set(corpus))
    np.random.seed(0)
    
    # Use dtype=bool for modern NumPy, which is equivalent to np.bool
    X = np.zeros((m, Tx, N_values), dtype=bool) 
    Y = np.zeros((m, Tx, N_values), dtype=bool)
    
    for i in range(m):
        # Sample a random starting index for a sequence of length Tx
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            
            # X[i, j, idx] is the input value at time step j
            # Y[i, j-1, idx] is the target value for the prediction at time step j-1
            # This implements a sequence-to-sequence model where Y is X shifted one step forward
            # X[i, 0] is ignored (or set to all zeros) and Y[i, Tx-1] is ignored
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    # The original code swaps axes for Y. This is unusual for Keras but kept for fidelity.
    # Keras models usually expect (m, Tx, N_values)
    Y = np.swapaxes(Y,0,1) 
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), N_values 

def next_value_processing(model, next_value, x, predict_and_sample, indices_values, abstract_grammars, duration, max_tries = 1000, temperature = 0.5):
    """
    Helper function to fix the first value.
    
    Arguments:
    next_value -- predicted and sampled value, index between 0 and 77
    x -- numpy-array, one-hot encoding of next_value
    predict_and_sample -- predict function
    indices_values -- a python dictionary mapping indices (0-77) into their corresponding unique value (ex: A,0.250,< m2,P-4 >)
    abstract_grammars -- list of grammars, on element can be: 'S,0.250,<m2,P-4> C,0.250,<P4,m-2> A,0.250,<P4,m-2>'
    duration -- scalar, index of the loop in the parent function
    max_tries -- Maximum numbers of time trying to fix the value
    
    Returns:
    next_value -- process predicted value
    """

    # fix first note: must not have < > and not be a rest
    if (duration < 0.00001):
        tries = 0
        while (next_value.split(',')[0] == 'R' or 
            len(next_value.split(',')) != 2):
            
            # give up after max_tries; random from input's first notes
            if tries >= max_tries:
                #print('Gave up on first note generation after', max_tries, 'tries')
                # np.random is exclusive to high
                rand = np.random.randint(0, len(abstract_grammars))
                next_value = abstract_grammars[rand].split(' ')[0]
            else:
                next_value = predict_and_sample(model, x, indices_values, temperature)

            tries += 1
            
    return next_value


def sequence_to_matrix(sequence, values_indices):
    """
    Convert a sequence (slice of the corpus) into a matrix (numpy) of one-hot vectors corresponding 
    to indices in values_indices
    
    Arguments:
    sequence -- python list
    
    Returns:
    x -- numpy-array of one-hot vectors 
    """
    sequence_len = len(sequence)
    # The length of the one-hot vector is determined by the size of the vocabulary
    N_values = len(values_indices) 
    x = np.zeros((1, sequence_len, N_values))
    for t, value in enumerate(sequence):
        if (not value in values_indices): print(value)
        x[0, t, values_indices[value]] = 1.
    return x

def one_hot(x):
    """
    Converts a probability vector (or tensor) x into a one-hot vector.
    This is typically used in the sampling step of a music generation model.

    Arguments:
    x -- tensor of shape (1, 1, N_values) representing probability distribution
    
    Returns:
    x -- tensor of shape (1, 1, N_values) representing the sampled one-hot vector
    """
    # Use tf.argmax instead of K.argmax (K is now discouraged for direct use)
    x = tf.argmax(x, axis=-1) 
    # Use tf.one_hot (N_values is hardcoded as 78 based on the original code's context)
    x = tf.one_hot(x, 78) 
    # Squeeze the extra dimension added by tf.argmax/tf.one_hot if necessary, then reshape
    x = tf.squeeze(x, axis=0) # Squeeze removes the first dimension (batch size of 1) if present
    x = RepeatVector(1)(x) # Use RepeatVector from the imported tf.keras.layers
    return x