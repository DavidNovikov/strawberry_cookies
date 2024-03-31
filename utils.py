# https://github.com/salu133445/musegan/blob/main/v2/musegan/utils/metrics.py#L17

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np


def load_npz_files(folder_path):
    """
    Load all npz files in a folder.

    Parameters:
    - folder_path (str): Path to the folder containing npz files.

    Returns:
    - list of numpy.ndarray: List of piano roll images represented as numpy matrices.
    """
    piano_rolls = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)
            with np.load(file_path) as data:
                piano_rolls.append(data['piano_roll'])
    return piano_rolls


def get_qualified_note_rate(pianoroll, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
    pianoroll = pianoroll.T
    padded = np.pad(pianoroll.astype(int), ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded, axis=0)
    flattened = diff.T.reshape(-1,)
    
    onsets = (flattened > 0).nonzero()[0]
    offsets = (flattened < 0).nonzero()[0]
    num_qualified_note = (offsets - onsets >= threshold).sum()
    
    return num_qualified_note / len(onsets)


def get_polyphonic_ratio(pianoroll, threshold=2):
    """Return the ratio of the number of time steps where the number of pitches
    being played is larger than `threshold` to the total number of time steps"""
    return np.sum(np.sum(pianoroll, 1) >= threshold) / pianoroll.shape[0]

def get_num_pitch_used(pianoroll):
    """Return the number of unique pitches used in a piano-roll."""
    return np.sum(np.sum(pianoroll, 1) > 0)

def percentage_of_empty_bars(piano_roll, threshold = 2):
    """
    Calculate the percentage of empty consecutive columns of size bigger than the threshold.

    Parameters:
    - piano_roll (numpy.ndarray): The piano roll image represented as a numpy matrix.
    - threshold (int): The threshold for the size of consecutive empty columns.

    Returns:
    - float: The percentage of empty bars.
    """

    # Count the total number of columns
    total_columns = piano_roll.shape[1]

    # Count the number of empty consecutive columns of size bigger than the threshold
    empty_bars = 0
    consecutive_empty_columns = 0
    for column in range(total_columns):
        if np.all(piano_roll[:, column] == 0):
            consecutive_empty_columns += 1
        else:
            if consecutive_empty_columns > threshold:
                empty_bars += consecutive_empty_columns
            consecutive_empty_columns = 0

    # Check if the last segment exceeds the threshold
    if consecutive_empty_columns > threshold:
        empty_bars += consecutive_empty_columns

    # Calculate the percentage
    percentage_empty_bars = empty_bars / total_columns

    return percentage_empty_bars


def COWS(folder_path):
    """
    Plot the number of notes being played in each column of the piano roll for all songs in the folder.

    Parameters:
    - folder_path (str): Path to the folder containing npz files.

    Returns:
    - None (plots the graph).
    """
    piano_rolls = load_npz_files(folder_path)

    notes_counts = []
    for piano_roll in piano_rolls:
        notes_counts.extend(np.sum(piano_roll, axis=0))

    unique_notes_counts, counts = np.unique(notes_counts, return_counts=True)
    
    counts = counts / counts.sum()

    plt.bar(unique_notes_counts, counts)
    plt.title('Number of Notes Per Column Across All Piano Rolls')
    plt.xlabel('Number of Notes')
    plt.xlim(right=unique_notes_counts.max()*1.2)
    plt.ylabel('Frequency')
    # plt.grid(True)
    plt.savefig(f'graph_COWS_{folder_path.replace("/", "")}')
    plt.close()


def plot_threshold_effect(func, folder_path, thresholds):
    """
    Plot the effect of different thresholds on the given function for piano roll/ image
    stored in npz files in the specified folder.

    Parameters:
    - func (function): The function to be plotted.
    - folder_path (str): Path to the folder containing npz files.
    - thresholds (list): List of thresholds to be tested.

    Returns:
    - None (plots the graph).
    """
    piano_rolls = load_npz_files(folder_path)
    percentages = []

    for piano_roll in piano_rolls:
        piano_roll_percentages = []
        for threshold in thresholds:
            piano_roll_percentages.append(func(piano_roll, threshold))
        percentages.append(piano_roll_percentages)

    percentages = np.array(percentages)
    mean_percentages = np.mean(percentages, axis=0)

    plt.plot(thresholds, mean_percentages, marker='o')
    plt.title('Effect of Threshold on {}'.format(func.__name__))
    plt.xlabel('Threshold')
    plt.ylabel('Average Percentage of Empty Bars')
    plt.grid(True)
    plt.savefig(f'graph_{func.__name__}_{folder_path.replace("/", "")}')
    plt.close()

def plot_function_output(func, folder_path):
    """
    Plot the output of a function for piano roll images stored in npz files in the specified folder.

    Parameters:
    - func (function): The function whose output is to be plotted.
    - folder_path (str): Path to the folder containing npz files.

    Returns:
    - None (plots the graph).
    """
    piano_rolls = load_npz_files(folder_path)

    outputs = []
    for piano_roll in piano_rolls:
        outputs.append(func(piano_roll))

    plt.hist(outputs)
    plt.title('Output of {} on Piano Rolls'.format(func.__name__))
    plt.xlabel('Piano Roll')
    plt.ylabel('Function Output')
    plt.grid(True)
    plt.savefig(f'graph_{func.__name__}_{folder_path.replace("/", "")}')
    plt.close()




if __name__ == '__main__':

    train_folder_path = "samples_for_analysis/"
    plot_threshold_effect(get_qualified_note_rate, train_folder_path, range(1,10))
    plot_threshold_effect(get_polyphonic_ratio, train_folder_path, range(1, 10))
    plot_threshold_effect(percentage_of_empty_bars, train_folder_path, range(1, 10))
    plot_function_output(get_num_pitch_used, train_folder_path)
    COWS(train_folder_path)

    our_folder_path = "our_music_for_data_analysis/samples_our_music/"
    plot_threshold_effect(get_qualified_note_rate, our_folder_path, range(1,10))
    plot_threshold_effect(get_polyphonic_ratio, our_folder_path, range(1, 10))
    plot_threshold_effect(percentage_of_empty_bars, our_folder_path, range(1, 10))
    plot_function_output(get_num_pitch_used, our_folder_path)
    COWS(our_folder_path)

