#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:37:51 2024

@author: samuelge
"""

import os
from midi2npz import midi_to_piano_roll  # Ensure this import works based on your project structure

def generate_npz_for_directory(maestro_dir, npz_dir):
    # Create the npz directory if it doesn't exist
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)

    # Walk through the maestro directory
    for root, dirs, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                # You might want to customize output naming or structure
                output_npz_path = os.path.join(npz_dir, os.path.splitext(file)[0] + ".npz")
                print(f"Processing {midi_path} -> {output_npz_path}")
                # Convert to piano roll and save as .npz
                midi_to_piano_roll(midi_path, npz_dir)

if __name__ == "__main__":
    maestro_dir = "maestro-v3.0.0"
    npz_dir = "npz"
    generate_npz_for_directory(maestro_dir, npz_dir)