#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:37:51 2024

@author: samuelge
"""

import os
# Ensure this import works based on your project structure
from midi2npz import midi_to_piano_roll
import threading

def run_though_list(list_of_files, npz_dir, print_lock):
    num_invalid_songs = 0
    for midi_path, file in list_of_files:
        output_npz_path = os.path.join(
                    npz_dir, os.path.splitext(file)[0] + ".npz")
        print(f"Processing {midi_path} -> {output_npz_path}")
        # Convert to piano roll and save as .npz
        try:
            midi_to_piano_roll(midi_path, npz_dir)
        except:
            num_invalid_songs += 1
    
    print_lock.acquire()
    print('*'*20, '\n', 'num invalid songs: ', num_invalid_songs, '\n', '*'*20)
    print_lock.release()


def generate_npz_for_directory(maestro_dir, npz_dir):
    # Create the npz directory if it doesn't exist
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)

    num_threads = 10
    lists = []
    threads = []
    idx = 0
    
    print_lock = threading.Lock()
    for i in range(num_threads):
        lists.append([])
    # Walk through the maestro directory
    for root, dirs, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                # You might want to customize output naming or structure
                lists[idx].append([midi_path, file])
                idx = (idx + 1) % num_threads
    for i in range(num_threads):
        newT = threading.Thread(target=run_though_list, args=(lists[i], npz_dir, print_lock))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()    

if __name__ == "__main__":
    # maestro_dir = "midi_files"
    maestro_dir = "maestro-v3.0.0"
    npz_dir = "npz_maestro_remove_poor_resolution_0416_drop_high_feq"
    generate_npz_for_directory(maestro_dir, npz_dir)