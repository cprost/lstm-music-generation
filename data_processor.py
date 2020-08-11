import pathlib
import numpy as np
import music21 as m21
import pickle

def get_song_pitches():
    '''
    Get individual notes and chords for each song, encoded along with the song's
    tempo, key, and time signature.
    '''
    song_data = []
    pitches = []
    tempos = []

    midi_dir = pathlib.Path('midi/')

    for midi_file in midi_dir.glob('*.mid'):
        print('Reading file: {}'.format(midi_file))

        midi_data = m21.converter.parse(midi_file)

        instrument_stream = m21.instrument.partitionByInstrument(midi_data)
        note_gen = None

        if instrument_stream is None:
            note_gen = midi_data.recurse()
        else:
            note_gen = instrument_stream.parts[0].recurse()

        key = None
        signature = None
        song = []

        for elem in note_gen:
            # every midi file has a tempo, key, and time signature in the header
            # all other subsequent contents will be notes, chords, or rests
            # IGNORING TEMPO FOR NOW

            if isinstance(elem, m21.tempo.MetronomeMark):
                tempos.append(elem.number)

            elif isinstance(elem, m21.key.Key):
                # may remove this from dataset, chord progression more important
                key = str(elem.sharps)  # if -ve then number of flats

            elif isinstance(elem, m21.meter.TimeSignature):
                signature = elem.ratioString

            elif isinstance(elem, (m21.note.Note, m21.note.Rest)):
                note = None
                # drop inexpressible rests
                if isinstance(elem, m21.note.Rest) and elem.duration.type is not 'inexpressible':
                    note = 'rest'
                    dur = str(elem.duration.type)

                elif isinstance(elem, m21.note.Note):
                    note = str(elem.pitch)
                    dur = str(elem.duration.type)

                full_note = (note, dur, key, signature)
                song.append(full_note)
                pitches.append(full_note)

            elif isinstance(elem, m21.chord.Chord):
                chord = '.'.join(str(note) for note in elem.normalOrder)
                dur = str(elem.duration.type)
                full_note = (chord, dur, key, signature)
                song.append(full_note)
                pitches.append(full_note)

        song_data.append(song)

    pitches = set(pitches)  # get unique pitches

    with open('data/midi_data', 'wb') as file:
        # np.save does not preserve object shape and order, use pickle instead
        pickle.dump(song_data, file)
        pickle.dump(pitches, file)

    return song_data, pitches, tempos


def create_sequences(song_data, pitches, pitch_count, seq_len=50): 
    data_in = []  # list of sequence lists
    data_out = []  # list of individual notes immediately after each data_in[] sequence

    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    # keras one-hot encoding only allows int input, keep as dict instead 
    pitch_to_int = dict((pitch, number) for number, pitch in enumerate(pitches))

    print(pitch_count)
    print(len(pitches))

    for song in song_data:
        song_notes = len(song)
        # summer += song_notes
        
        for i in range(0, song_notes - seq_len):
            # get seq_len total pitches as integer input sequence
            seq_in = song[i : i+seq_len]
            seq_in = [pitch_to_int[pitch] for pitch in seq_in]

            # get the next 1 token after the seq_len tokens as the output
            seq_out = song[i + seq_len]
            seq_out = pitch_to_int[seq_out]

            data_in.append(seq_in)
            data_out.append(seq_out)
    
    seq_count = len(data_in)  # sanity check
    print(seq_count, 'total sequences')

    data_in = np.divide(data_in, pitch_count)

    return data_in, data_out

if __name__ == '__main__':
    # song_data, pitches, tempos = get_song_pitches()

    with open('data/midi_data', 'rb') as file:
        song_data = pickle.load(file)
        pitches = pickle.load(file)

    # # get amount of pitch names
    pitch_count = len(pitches)
    song_count = len(song_data)

    print('\nTotal unique pitches: {}'.format(pitch_count))
    print('Total songs: {}'.format(song_count))

    create_sequences(song_data, pitches, pitch_count)