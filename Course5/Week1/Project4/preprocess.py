'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Parse, cleanup and process data.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest

# 假设 grammar.py 存在，并且其中包含 parse_melody 函数
from grammar import parse_melody 
# 假设 music_utils.py 存在
from music_utils import *

#----------------------------HELPER FUNCTIONS----------------------------------#

def __parse_midi(data_fn):
    ''' Helper function to parse a MIDI file into its measures and chords 
        - 集成了对所有已知 music21 兼容性错误的修复。
    '''
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    
    # ------------------ 旋律声部解析 ------------------
    # 尝试找到旋律部分 (原代码是 Part #5)
    try:
        melody_stream = midi_data[5]  
    except IndexError:
        print("警告: MIDI 文件没有 Part #5，使用第一个 Part 作为旋律部分。")
        melody_stream = midi_data.elements[0] 
        
    # 获取所有的 Voice 元素
    all_voices = melody_stream.getElementsByClass(stream.Voice)
    
    if len(all_voices) == 0:
        # 如果找不到 Voice，我们创建一个新的 Voice 对象，并将 Part 中的 Notes/Rests 插入进去
        melody_voice = stream.Voice()
        # 使用 .flatten().notesAndRests 获取所有音乐元素
        for element in melody_stream.flatten().notesAndRests:
             melody_voice.insert(element.offset, element)
        print("警告: 旋律 Part 中未找到 stream.Voice，使用 Part 的所有元素构建新的 Voice 声部。")
        
    elif len(all_voices) == 1:
        # 如果只有一个 Voice，直接使用它
        melody_voice = all_voices[0]
    else:
        # 如果有多个 Voice (>= 2)，将它们合并到第一个 Voice 中
        melody_voice = all_voices[0]
        for voice_to_merge in all_voices[1:]:
            for j in voice_to_merge:
                melody_voice.insert(j.offset, j)
    # ------------------ 旋律声部解析结束 ------------------

    # 修正四分音符长度为 0.0 的情况
    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature and add Electric Guitar.
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # The accompaniment parts. Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    
    # ------------------ 伴奏声部解析 ------------------
    valid_parts = []
    for i, j in enumerate(midi_data):
        if i in partIndices and i < len(midi_data):
            # 过滤 Metadata 对象，只对 stream.Stream 调用 .flatten()
            if isinstance(j, stream.Stream):
                valid_parts.append(j.flatten())
                
    comp_stream.append(valid_parts)
    # ------------------ 伴奏声部解析结束 ------------------

    # Full stream containing both the melody and the accompaniment. 
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)

    # ------------------ 提取 Solo Stream (使用 insert 修复) ------------------
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        
        # 1. 提取所有非音乐元素 (乐器、速度等)
        non_musical_elements = []
        non_musical_elements.extend(part.getElementsByClass(instrument.Instrument))
        non_musical_elements.extend(part.getElementsByClass(tempo.MetronomeMark))
        non_musical_elements.extend(part.getElementsByClass(key.KeySignature))
        non_musical_elements.extend(part.getElementsByClass(meter.TimeSignature))
        
        # 逐个插入非音乐元素
        for e in non_musical_elements:
            curr_part.insert(e.offset, e)

        
        # 2. 提取指定范围的音乐片段 (音符/和弦)
        solo_elements = part.getElementsByOffset(476, 548, 
                                                 includeEndBoundary=True)
        # 逐个插入音乐元素
        for e in solo_elements:
            curr_part.insert(e.offset, e)
        
        # 将 curr_part 扁平化，然后插入到 solo_stream
        cp = curr_part.flatten() 
        solo_stream.insert(cp)
    # ------------------ 提取 Solo Stream 结束 ------------------
    
    # Group by measure 
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    
    # 确保只处理 Note/Rest
    musical_elements = melody_stream.getElementsByClass(['Note', 'Rest'])
    
    # 注意: offset / 4 是因为每 4 个 quarterLength 是一个 measure
    offsetTuples = [(int(n.offset / 4), n) for n in musical_elements]
    measureNum = 0 
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get the stream of chords.
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Final assert and trimming (确保 measure 和 chord 数量匹配)
    if len(chords) > len(measures):
        # 只有在 measures 数量不为 0 时才删除，否则程序可能会崩溃
        if len(measures) > 0:
            del chords[len(chords) - 1]
    elif len(measures) > len(chords):
        # 只有在 chords 数量不为 0 时才删除
        if len(chords) > 0:
            del measures[len(measures) - 1]
        
    assert len(chords) == len(measures), f"措施数量({len(measures)})与和弦数量({len(chords)})不匹配！"

    return measures, chords

def __get_abstract_grammars(measures, chords):
    ''' Helper function to get the grammatical data from given musical data. '''
    # extract grammars
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        # 假设 parse_melody 存在于 grammar.py 中
        parsed = parse_melody(m, c) 
        abstract_grammars.append(parsed)

    return abstract_grammars

#----------------------------PUBLIC FUNCTIONS----------------------------------#

def get_musical_data(data_fn):
    ''' Get musical data from a MIDI file '''
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)
    return chords, abstract_grammars

def get_corpus_data(abstract_grammars):
    ''' Get corpus data from grammatical data '''
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))
    return corpus, values, val_indices, indices_val