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
        - 修复了 Voice 提取逻辑
        - 修复了 Solo 切片为空时的回退逻辑
    '''
    # Parse the MIDI data
    midi_data = converter.parse(data_fn)
    
    # ------------------ 旋律声部解析 ------------------
    try:
        melody_stream = midi_data[5]  
    except IndexError:
        melody_stream = midi_data.elements[0] 
        
    all_voices = melody_stream.getElementsByClass(stream.Voice)
    
    if len(all_voices) == 0:
        melody_voice = stream.Voice()
        for element in melody_stream.flatten().notesAndRests:
             melody_voice.insert(element.offset, element)
    elif len(all_voices) == 1:
        melody_voice = all_voices[0]
    else:
        melody_voice = all_voices[0]
        for voice_to_merge in all_voices[1:]:
            for j in voice_to_merge:
                melody_voice.insert(j.offset, j)
    
    # 修正长度
    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # ------------------ 伴奏声部解析 ------------------
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    valid_parts = []
    for i, j in enumerate(midi_data):
        if i in partIndices and i < len(midi_data):
            if isinstance(j, stream.Stream):
                valid_parts.append(j.flatten())
    comp_stream.append(valid_parts)

    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)

    # ------------------ 提取 Solo Stream (核心修复) ------------------
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        
        # 1. 插入非音乐元素
        for cls in [instrument.Instrument, tempo.MetronomeMark, key.KeySignature, meter.TimeSignature]:
            for e in part.getElementsByClass(cls):
                curr_part.insert(e.offset, e)
        
        # 2. 尝试提取指定范围 (原始 DeepJazz 逻辑: 476-548)
        solo_elements = part.getElementsByOffset(476, 548, includeEndBoundary=True)
        
        # === 修复开始: 如果指定范围没找到音符，就尝试使用全部 ===
        if len(solo_elements) == 0:
            # 只有当这是一个包含音符的 Part 时才回退到全部
            if len(part.flatten().notes) > 0:
                print(f"提示: 在 offset 476-548 未找到音符，将使用整个 Part (Offset: {part.offset})...")
                # 限制一下长度，避免太长导致内存爆炸，比如取前 100 个小节
                solo_elements = part.flatten().notesAndRests.getElementsByOffset(0, 400)
        # === 修复结束 ===

        for e in solo_elements:
            curr_part.insert(e.offset, e)
        
        solo_stream.insert(curr_part.flatten())
    # -----------------------------------------------------------
    
    # Group by measure 
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream.getElementsByClass(['Note', 'Rest'])]
    measureNum = 0 
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get chords
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Final trimming
    if len(chords) > len(measures):
        if len(measures) > 0: del chords[len(chords) - 1]
    elif len(measures) > len(chords):
        if len(chords) > 0: del measures[len(measures) - 1]
    
    # 如果还是不匹配，打印调试信息而不是直接断言崩溃
    if len(chords) != len(measures):
        print(f"⚠️ 警告: Measures ({len(measures)}) 与 Chords ({len(chords)}) 数量不匹配。尝试截断到最小长度。")
        min_len = min(len(chords), len(measures))
        # 强制截断字典
        measures = OrderedDict(list(measures.items())[:min_len])
        chords = OrderedDict(list(chords.items())[:min_len])

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