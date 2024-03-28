#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Mar 28 10:56:13 2024

@author: samuelg
"""

import numpy as np
from img2midi import image2midi
import matplotlib.pyplot as plt
import os


def npz2midi(file_path):
    npz = np.load(file_path)
    data = npz['piano_roll']
    
    sample_path = f"example.png"
    plt.imsave(sample_path, data, cmap='gray')
    
    image2midi(sample_path)
    
if __name__ == "__main__":
    npz2midi('npz_maestro_remove_poor_resolution_0416_drop_high_feq/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav_Piano.npz')