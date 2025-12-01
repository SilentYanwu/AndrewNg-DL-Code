# data_utils.py
# Updated imports for TensorFlow 2.x
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from music21 import * # Assuming these utilities are defined in the same directory
from music_utils import * 
from preprocess import * # The data_processing function is assumed to be available from the previous step

# --- Global Initialization (Keep these for context) ---
chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
N_tones = len(set(corpus))
n_a = 64 # LSTM hidden state dimension
x_initializer = np.zeros((1, 1, 78)) # (batch_size, time_steps, vocab_size)
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
# ----------------------------------------------------


def load_music_utils():
    """
    Loads musical data, processes the corpus, and returns the training data.
    Assumes data_processing is available in the current scope.
    """
    chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    
    # Assumes data_processing is defined elsewhere (from the previous step)
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)  
    return (X, Y, N_tones, indices_tones)


def generate_music(inference_model, corpus = corpus, abstract_grammars = abstract_grammars, tones = tones, tones_indices = tones_indices, indices_tones = indices_tones, T_y = 10, max_tries = 1000, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. 
    Creates an audio stream to save the music and play it.
    
    Arguments:
    inference_model -- Keras model Instance, output of djmodel()
    ... (other arguments as defined)
    """
    
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0 
    num_chords = int(len(chords) / 3) 
    
    print("Predicting new values for different set of chords.")
    
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        for j in chords[i]:
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones += pred[-1]
            
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    return out_stream


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                      c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    # **TF2.x Change**: model.predict() is a standard method and works as before.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    # Use np.argmax on the numpy output for simplicity and speed.
    # The axis=-1 ensures it selects the maximum along the vocabulary dimension (78 classes).
    indices = np.argmax(pred, axis = -1)
    
    # **TF2.x Change**: Import to_categorical from tf.keras.utils
    results = to_categorical(indices, num_classes=78)
    
    return results, indices