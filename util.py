import numpy as np
from params import *
import pretty_midi
import music21
import copy


def result_sampling(rolls):
    num = rolls.shape[0]
    new_rolls = []

    for i in range(num):
        roll = rolls[i]
        timesteps = roll.shape[0]
        new_roll = np.zeros((roll.shape[0],89))
        for step in range(timesteps):
            melody_note = np.argmax(roll[step,:melody_output_dim])
            melody_start = roll[step,melody_output_dim] > 0.5
            bass_note = np.argmax(roll[step, melody_output_dim+melody_note_start_dim:melody_output_dim+melody_note_start_dim + bass_output_dim])
            bass_start = roll[step,melody_output_dim+melody_note_start_dim + bass_output_dim] > 0.5

            new_roll[step,melody_note] = 1
            new_roll[step, melody_output_dim] = melody_start
            new_roll[step, bass_note+melody_output_dim+melody_note_start_dim] = 1
            new_roll[step,melody_output_dim+melody_note_start_dim + bass_output_dim] = bass_start
        new_rolls.append(new_roll)
    return np.array(new_rolls)





def roll_to_pretty_midi(rolls,pm_old):

    melody_notes = []
    bass_notes = []
    step_time = 60 / TEMPO / 4

    previous_m_pitch = -1
    previous_b_pitch = -1
    previous_m_start = False
    previous_b_start = False



    for timestep in range(rolls.shape[0]):
        melody_pitch = np.where(rolls[timestep, :melody_dim] != 0)[0]
        melody_start = rolls[timestep, melody_dim] != 0
        bass_pitch = np.where(rolls[timestep, melody_dim + 1:melody_dim + bass_dim + 1] != 0)[0]
        bass_start = rolls[timestep, melody_dim + 1 + bass_dim] != 0

        # not the emtpy pitch
        if len(melody_pitch) > 0:
            melody_pitch = melody_pitch[0]

            if previous_m_pitch != -1:
                ## set the end

                if melody_pitch == melody_dim - 1 or melody_start or melody_pitch != previous_m_pitch or timestep == \
                        rolls.shape[0] - 1:
                    if previous_m_start:
                        m_end_time = timestep * step_time
                        melody_notes.append(pretty_midi.Note(velocity=100, pitch=previous_m_pitch + 24,
                                                             start=m_start_time, end=m_end_time))
                        previous_m_start = False

            ## set the start
            if melody_pitch != melody_dim - 1:

                if timestep == 0 or melody_start or rolls[timestep - 1, melody_pitch] == 0:
                    m_start_time = timestep * step_time
                    previous_m_start = True
                    if previous_m_pitch != -1:
                        while melody_pitch - previous_m_pitch > 12:
                            melody_pitch -= 12
                        while melody_pitch - previous_m_pitch < -12:
                            melody_pitch += 12
                    previous_m_pitch = melody_pitch

        if len(bass_pitch) > 0:
            bass_pitch = bass_pitch[0]

            if previous_b_pitch != -1:
                ## set the end
                if bass_pitch == bass_dim - 1 or bass_start or bass_pitch != previous_b_pitch or timestep == \
                        rolls.shape[0] - 1:
                    if previous_b_start:
                        b_end_time = timestep * step_time
                        #                         print(f'bass pitch is {previous_b_pitch + 24}')
                        bass_notes.append(pretty_midi.Note(velocity=100, pitch=previous_b_pitch + 36,
                                                           start=b_start_time, end=b_end_time))
                        previous_b_start = False

            ## set the start
            if bass_pitch != bass_dim - 1:

                if timestep == 0 or bass_start or rolls[timestep - 1, bass_pitch + melody_dim + 1] == 0:
                    b_start_time = timestep * step_time
                    previous_b_start = True

                    previous_b_pitch = bass_pitch

    if pm_old:
        pm_new = copy.deepcopy(pm_old)
        pm_new.instruments[0].notes = melody_notes
        pm_new.instruments[1].notes = bass_notes
        pm_new.instruments = pm_new.instruments[:2]
        return pm_new

    else:
        pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
        piano = pretty_midi.Instrument(program=1)
        piano.notes = melody_notes
        bass = pretty_midi.Instrument(program=33)
        bass.notes = bass_notes
        pm.instruments.append(piano)
        pm.instruments.append(bass)

        return pm

def show_score(pm):
    pm.write('./temp.mid')
    stream = music21.converter.parse('./temp.mid')
    stream.show()


def setup_musescore(musescore_path=None):
    if not is_ipython(): return

    import platform
    from music21 import environment
    from pathlib import Path

    system = platform.system()
    if system == 'Linux':
        import os
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # https://musescore.org/en/node/29041

    existing_path = environment.get('musicxmlPath')
    if existing_path: return
    if musescore_path is None:
        if system == 'Darwin':
            app_paths = list(Path('/Applications').glob('MuseScore *.app'))
            if len(app_paths): musescore_path = app_paths[-1] / 'Contents/MacOS/mscore'
        elif system == 'Linux':
            musescore_path = '/usr/bin/musescore'

    if musescore_path is None or not Path(musescore_path).exists():
        print(
            'Warning: Could not find musescore installation. Please install musescore (see README) and/or update music21 environment paths')
    else:
        environment.set('musicxmlPath', musescore_path)
        environment.set('musescoreDirectPNGPath', musescore_path)


def is_ipython():
    try:
        get_ipython
    except:
        return False
    return True


def is_colab():
    try:
        import google.colab
    except:
        return False
    return True

#
# def draw_scores(pm):
#
