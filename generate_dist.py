#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:40:55 2024

@author: samuelge
"""
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def generate_dist(dir_name, imgs):
    hist = []
    sparcity = []
    sample = cv2.imread(dir_name + imgs[0])
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)/255
    total_pixels = sample.shape[0] * sample.shape[1]
    print(total_pixels)
    for im in imgs:
        img = cv2.imread(dir_name + im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
        dist = img.sum(axis=0)
        hist.append(dist)
        filled = sum(sum(img))
        sparcity.append(filled/total_pixels)
    return np.asarray(hist).flatten('F'), sparcity

dir_name = "png_files/SONGS/"
imgs = os.listdir(dir_name)

#x, sparcity = generate_dist(dir_name, imgs)

#bins = np.arange(0, np.max(x) + 1, 1)  # Bin edges from 0 to maximum value of x with bin size 1
bins = np.arange(0, 88, 1)  # Bin edges from 0 to maximum value of x with bin size 1

hist_values, bin_edges, _ = plt.hist(x, bins=bins, edgecolor='black')

from scipy.stats import kstest

# Define your histogram distribution (you already have this)
count = []
# Load sample from dist
for i in range(1000):
    img = cv2.imread(dir_name + imgs[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255

    # Generate sample data
    sample_data = img.sum(axis=0)

    # Perform Kolmogorov-Smirnov test
    # Assuming 'hist_values' and 'bin_edges' are already defined
    D, p_value = kstest(sample_data, lambda x: np.interp(x, bins[:-1], np.cumsum(hist_values))/np.sum(hist_values))
    count.append(D)
    print("Kolmogorov-Smirnov test statistic:", D)
    #print("p-value:", p_value)
    
random = np.random.rand(88)
zeros = np.zeros(88)

D, p_value = kstest(random, lambda x: np.interp(x, bins[:-1], np.cumsum(hist_values))/np.sum(hist_values))

print("\nKolmogorov-Smirnov test statistic random:", D) 
 
D, p_value = kstest(zeros, lambda x: np.interp(x, bins[:-1], np.cumsum(hist_values))/np.sum(hist_values))

print("\nKolmogorov-Smirnov test statistic zero:", D)

