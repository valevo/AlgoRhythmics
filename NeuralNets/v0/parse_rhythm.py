# -*- coding: utf-8 -*-

#%%

import numpy as np
import music21 as m21
from music21 import corpus
from fractions import Fraction

from tqdm import tqdm


def clean_score(score):
    '''
    - Transposes to key of C
    - Sets simultaneous notes to a chord object (chordifies parts, any problems?)
    - Fixes first bar to be of correct length and fill with rest (eg in Bach chorales)
    - Set time signature denominator to 4?
    - etc
    '''
    key = score.analyze('key')
    diff = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    
    new_score = m21.stream.Score()
    
    for part in score.parts:
        # check if first bar adds up...
        m0 = part.getElementsByClass(m21.stream.Measure)[0]
        if m0.duration != m0.barDuration:
            # need to pad out the left side with a rest...
            shift = m0.paddingLeft
            r = m21.note.Rest(quarterLength=shift)
            m0.insertAndShift(0, r)
            for element in m0.elements:
                if element not in m0.notesAndRests:
                    element.offset = 0.0
            
            for m in part.getElementsByClass(m21.stream.Measure)[1:]:
                m.offset += shift
        
        new_score.insert(part.offset, part.chordify())
    
    return new_score.transpose(diff)



def rhythm_word(score, ts=None):
    '''
    Splits the score into single beat length 'words' that contain offset and tie information from that beat.
    - Includes START token with Time Signature information
    - Gets the length of the beat from the Time Signature
    - Robust against odd time signatures and tied notes
    '''
    
    data = []
    
    if not ts:
        try:
            ts = score.recurse().getElementsByClass("TimeSignature")[0]
        except:
            ts = m21.meter.TimeSignature('4/4')
    
    beat_length = Fraction(ts.beatDuration.quarterLength)
    
    for i in np.arange(score.duration.quarterLength, step=beat_length):
        w = score.flat.getElementsByOffset(i, i+beat_length, includeEndBoundary=False)
        word = []
        for x in w.flat.notesAndRests:
            if x.tie:
                if x.tie.type != 'start':
                    continue
    
            word.append(x.offset - i)
    
        data.append(word)
    
    return data


def corpus_to_beats(files=None, scores=None):
    if not files and not scores:
        raise ValueError("Both `files` and `scores` are None!")
    if files:
        scores = (corpus.parse(p) for p in files)
    
    cleaned = (clean_score(s) for s in scores)
        
    for s in tqdm(cleaned, total=len(files) if files else len(scores)):
        yield [rhythm_word(p) for p in s.parts]
        
    
    
