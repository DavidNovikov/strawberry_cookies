import os
import midi2img


def create_data(num_to_extract=-1):
    # Define the directory containing MIDI files
    # path_to_directory_containing_midifiles
    os_name = os.name
    midi_dir = 'midi_files\\' if os_name == 'nt' else 'midi_files/'
    png_dir = 'png_files\\SONGS\\' if os_name == 'nt' else 'png_files/SONGS/'

    # Get a list of all files in the directory
    files = os.listdir(midi_dir)

    # Filter out only the MIDI files
    midi_files = [midi_dir + f for f in files if f.endswith(".mid")]
    midi_files.sort()

    # Output directory for saving images
    output_dir = png_dir  # output_images

    total_num_of_files = len(midi_files)
    num_to_extract = total_num_of_files if num_to_extract == - \
        1 else min(num_to_extract, total_num_of_files)
    print('extracting ', num_to_extract, ' midi files to pngs')
    for i in range(num_to_extract):
        # Convert MIDI files to images
        print(midi_files[i])
        midi2img.midi2image(i, midi_files[i], output_dir)


if __name__ == "__main__":
    # create_data()
    create_data()
    # create_data(100000)
