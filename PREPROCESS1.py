import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

KERN_DATASET_PATH = "Dataset"           #dataset directory read all files of required type from this dataset
SAVE_DIR = "Dataset"                    #The directory where dataset is saved
SINGLE_FILE_DATASET = "file_dataset"    #A single file where the preprocessed data will be written
MAPPING_PATH = "mapping.json"           #Mapping of converted preprocessed(time series data) to vector
SEQUENCE_LENGTH = 64

ACCEPTABLE_DURATIONS = [            
    0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4                 #Acceptable pitch durations
]

# Function to load songs in the kern format using music21
def load_songs_in_kern(dataset_path):          
    songs = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):       #return the song if it has acceptable duration
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose_song(song):                                #return the transpose of a song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    transposed_song = song.transpose(interval)
    return transposed_song

#Function to enocde the song
def encode_song(song, time_step=0.25):               
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

#Function to preprocess the data using all functions we defined above
def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        song = transpose_song(song)
        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")


def load_song(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

#Function to create a single file dataset onto which we write the preprocessed data
def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load_song(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs


def create_mapping(songs, mapping_path):
    mappings = {}
    songs = songs.split()
    vocabulary = list(set(songs))
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs

#To generate the training sequence(in time series form)
def generate_training_sequences(sequence_length):
    songs = load_song(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)
    print(f"There are {len(inputs)} sequences.")
    return inputs, targets

#Defining the main function
def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
