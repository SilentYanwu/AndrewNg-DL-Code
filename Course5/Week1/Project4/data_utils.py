# data_utils.py
# Updated imports for TensorFlow 2.x
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from music21 import * 
from music_utils import * 
from preprocess import * 
import os,sys
# 添加路径修复代码
def fix_paths():
    """修复导入路径和文件路径"""
    # 将当前文件所在目录添加到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 切换到当前文件所在目录
    os.chdir(current_dir)

# 在导入本地文件/模型之前调用
fix_paths()

# --- Global Storage (Initialized as None) ---
# 我们不再在导入时直接加载数据，而是设为 None，在函数中按需加载
chords = None
abstract_grammars = None
corpus = None
tones = None
tones_indices = None
indices_tones = None

# MIDI 文件路径
MIDI_DATA_PATH = 'data/original_metheny.mid'

def _ensure_data_loaded():
    """
    内部辅助函数：确保全局数据已被加载。
    如果在 import 时加载失败，可以在运行时捕获错误。
    """
    global chords, abstract_grammars, corpus, tones, tones_indices, indices_tones
    
    if chords is None: # 只有当数据为空时才加载
        if not os.path.exists(MIDI_DATA_PATH):
            raise FileNotFoundError(f"找不到数据文件: {MIDI_DATA_PATH}。请在项目目录下创建 'data' 文件夹并放入 'original_metheny.mid'。")
            
        print(f"正在加载音乐数据: {MIDI_DATA_PATH} ...")
        chords, abstract_grammars = get_musical_data(MIDI_DATA_PATH)
        corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
        print("数据加载完成。")

def load_music_utils():
    """
    Loads musical data, processes the corpus, and returns the training data.
    """
    # 确保数据已加载
    _ensure_data_loaded()
    
    N_tones = len(set(corpus))
    
    # 调用 music_utils 中的 data_processing
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)  
    
    return (X, Y, N_tones, indices_tones)


def generate_music(inference_model, corpus_arg=None, abstract_grammars_arg=None, tones_arg=None, tones_indices_arg=None, indices_tones_arg=None, T_y = 10, max_tries = 1000, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. 
    """
    
    # 确保数据已加载
    _ensure_data_loaded()
    
    # 处理默认参数：如果调用时未传入，则使用全局加载的数据
    # 注意：Python 的默认参数是在定义时计算的，所以我们不能在函数定义头里写 default=corpus
    local_chords = chords 
    local_abstract_grammars = abstract_grammars_arg if abstract_grammars_arg is not None else abstract_grammars
    local_tones_indices = tones_indices_arg if tones_indices_arg is not None else tones_indices
    local_indices_tones = indices_tones_arg if indices_tones_arg is not None else indices_tones

    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0 
    num_chords = int(len(local_chords) / 3) 
    
    print("Predicting new values for different set of chords.")
    
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        for j in local_chords[i]:
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [local_indices_tones[p] for p in indices]
        
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

    # Save audio stream to file
    # 确保 output 文件夹存在
    if not os.path.exists('output'):
        os.makedirs('output')
        
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    return out_stream


def predict_and_sample(inference_model, x_initializer=None, a_initializer=None, c_initializer=None):
    """
    Predicts the next value of values using the inference model.
    """
    # 初始化器如果为 None，则生成默认的零矩阵
    # 注意：这里的维度 (1, 1, 78) 和 (1, 64) 是硬编码的，依赖于 main.py 的设置
    # 为了简化，这里假设传入的参数通常不为 None，或者我们在这里重新初始化
    if x_initializer is None:
        x_initializer = np.zeros((1, 1, 78))
    if a_initializer is None:
        a_initializer = np.zeros((1, 64))
    if c_initializer is None:
        c_initializer = np.zeros((1, 64))
    
    # TF2.x: model.predict
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    # Use np.argmax
    indices = np.argmax(pred, axis = -1)
    
    # TF2.x: to_categorical
    results = to_categorical(indices, num_classes=78)
    
    return results, indices