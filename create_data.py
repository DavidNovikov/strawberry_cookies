import os
import midi2img


def create_data():
    # Define the directory containing MIDI files
    midi_dir = "C:\\Users\\esade\\PycharmProjects\\DL4CV\\midi_files"  # path_to_directory_containing_midifiles

    # Get a list of all files in the directory
    files = os.listdir(midi_dir)

    # Filter out only the MIDI files
    midi_files = ["C:\\Users\\esade\\PycharmProjects\\DL4CV\\midi_files\\" + f for f in files if f.endswith(".mid")]

    # Output directory for saving images
    output_dir = "C:\\Users\\esade\\PycharmProjects\\DL4CV\\png_files\\"  # output_images

    for i in range(len(midi_files)):
        # Convert MIDI files to images
        midi2img.midi2image(midi_files[i], output_dir)


if __name__ == "__main__":
    create_data()

