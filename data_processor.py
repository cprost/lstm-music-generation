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

song_data, pitches, tempos = get_song_pitches()

# get amount of pitch names
pitch_count = len(pitches)
song_count = len(song_data)

print('\nTotal unique pitches: {}'.format(pitch_count))