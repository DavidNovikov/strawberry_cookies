#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:46:22 2024

@author: samuelge
"""

from music21 import converter, instrument, note, chord
import numpy as np
import os

def extractNote(element):
    return int(element.pitch.ps)


def extractDuration(element):
    return element.duration.quarterLength


def get_notes(notes_to_parse):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))

        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start": start, "pitch": notes, "dur": durations}


def midi_to_piano_roll(midi_path, output_dir, resolution=0.25, lowerBoundNote=21, upperBoundNote=109):
    mid = converter.parse(midi_path)
    instruments = instrument.partitionByInstrument(mid)
    data = {}
    
    try:
        for i, instrument_i in enumerate(instruments.parts):
            notes_to_parse = instrument_i.recurse()
            notes_data = get_notes(notes_to_parse)
            if len(notes_data["start"]) == 0:
                continue
            instrument_name = instrument_i.partName if instrument_i.partName else f"instrument_{i}"
            data[instrument_name] = notes_data
    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0"] = get_notes(notes_to_parse)

    for instrument_name, values in data.items():
        pitches = values["pitch"]
        durs = values["dur"]
        starts = values["start"]

        # Determine the maximum length based on the last note's start time and duration
        song_length = max([start + dur for start, dur in zip(starts, durs)]) / resolution
        song_length = int(np.ceil(song_length))
        
        # Initialize the piano roll matrix
        matrix = np.zeros((upperBoundNote - lowerBoundNote, song_length))

        for dur, start, pitch in zip(durs, starts, pitches):
            dur = int(dur / resolution)
            start = int(start / resolution)
            for j in range(start, start + dur):
                if 0 <= pitch - lowerBoundNote < upperBoundNote - lowerBoundNote:
                    matrix[pitch - lowerBoundNote, j] = 1  # Using 1s instead of 255 for binary representation

        # Save the matrix as .npz
        filename = os.path.join(output_dir, f"{os.path.basename(midi_path).replace('.midi', '')}_{instrument_name}.npz")
        np.savez_compressed(filename, piano_roll=matrix)