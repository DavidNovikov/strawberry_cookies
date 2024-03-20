#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:09:39 2024

@author: samuelge
"""

import numpy as np
import os

def npz_to_segments(npz_path, output_dir, delta_t, sample_length):
    # Load the piano roll
    data = np.load(npz_path)
    piano_roll = data['piano_roll']
    
    # The total number of time steps (columns) in the piano roll
    total_time_steps = piano_roll.shape[1]
    
    # Calculate the number of segments that can be created, ensuring no trailing zeros
    num_segments = 1 + (total_time_steps - sample_length) // delta_t
    
    # Check if there's enough data for at least one segment
    if num_segments <= 0:
        print("Not enough data for a single segment. Consider reducing sample_length or delta_t.")
        return
    
    # Generate and save segments
    for i in range(num_segments):
        start = i * delta_t
        end = start + sample_length
        if end > total_time_steps:
            break  # Avoid creating segments with trailing zeros
        segment = piano_roll[:, start:end]
        
        # Save the segment
        segment_filename = os.path.join(output_dir, f"{os.path.basename(npz_path).replace('.npz', '')}_segment_{i}.npz")
        np.savez_compressed(segment_filename, piano_roll=segment)
        
        
# single file example
"""
if __name__ == "__main__":
    npz_path = "npz/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1_Cui05-08.mid.npz"
    output_dir = "samples"
    delta_t = 25
    sample_length = 88
    npz_to_segments(npz_path=npz_path, output_dir=output_dir, delta_t=delta_t, sample_length=sample_length)

# Full run
"""
if __name__ == "__main__":
    npz_dir = "npz/"
    if not os.path.exists('samples'):
        os.makedirs('samples')
    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.endswith(".npz") or file.endswith(".npy"):
                output_dir = "samples"
                delta_t = 20
                sample_length = 88
                npz_to_segments(npz_path=npz_dir+file, output_dir=output_dir, delta_t=delta_t, sample_length=sample_length)
