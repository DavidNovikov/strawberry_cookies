# https://github.com/salu133445/musegan/blob/main/v2/musegan/utils/metrics.py#L17

import numpy as np
import matplotlib.pyplot as plt

def get_qualified_note_rate(pianoroll, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
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
    return np.sum(np.sum(pianoroll, 0) > 0)


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


def plot_threshold_effect(func, piano_roll, thresholds):
    """
    Plot the effect of different thresholds on the given function.

    Parameters:
    - func (function): The function to be plotted.
    - piano_roll (numpy.ndarray): The piano roll image represented as a numpy matrix.
    - thresholds (list): List of thresholds to be tested.

    Returns:
    - None (plots the graph).
    """
    percentages = []
    for threshold in thresholds:
        percentages.append(func(piano_roll, threshold))

    plt.plot(thresholds, percentages, marker='o')
    plt.title('Effect of Threshold on {}'.format(func.__name__))
    plt.xlabel('Threshold')
    plt.ylabel('Percentage of Empty Bars')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    tensor = np.zeros((88, 88))
