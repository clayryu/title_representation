from music21 import environment
from music21 import converter
import muspy
import os, io, sys
from music21 import abcFormat
from pathlib import Path

def noop(x):
  pass

class MuteWarn:
    def __enter__(self):
        self._init_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._init_stdout


abcFormat.translate.environLocal.warn = noop
us=environment.UserSettings()
us['musescoreDirectPNGPath'] = '/usr/bin/mscore'

os.putenv("QT_QPA_PLATFORM", "offscreen")
os.putenv("XDG_RUNTIME_DIR", environment.Environment().getRootTempDir())


def save_score_image_from_abc(abc, file_name):
  assert isinstance(abc, str)
  with MuteWarn():
    convert = converter.parse(abc)
    convert.write('musicxml.png', fp=file_name)

def save_wav_from_abc(abc, file_name):
  assert isinstance(abc, str)
  with MuteWarn():
    muspy.read_abc_string(abc).write_audio(file_name, rate=16000)


def save_abc(abc_str, abc_fn):
  with open(abc_fn, 'w') as f:
    f.write(abc_str)

class Note2ABC:
  def __init__(self) -> None:
    self.abc_vocab = self.get_abc_pitchs_w_sharp()

  def get_abc_pitchs_w_sharp(self):
    abc_octave = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
    abc_notation = []

    for num in range(4):
      if num == 0:
        octave = [p + ',' for p in abc_octave]
      elif num == 1:
        octave = abc_octave
      elif num == 2:
        octave = [p.lower() for p in abc_octave]
      else:
        octave = [p.lower() + "'" for p in abc_octave]
      abc_notation.extend(octave)
    return abc_notation

  def pitch2abc(self, midi_pitch: int):
    return self.abc_vocab[midi_pitch-36]

  def duration2abc(self, duration):
    if duration == 0.5:
      return '/'
    elif duration == 1:
      return ''
    elif duration == 0.75:
      return '3/4'
    elif duration == 0.25:
      return '1/4'
    elif duration == 1.5:
      return '3/2'
    else:
      return str(int(duration))

  def __call__(self, pitch_dur_str: str):
    pitch, dur = pitch_dur_str.split('//')
    midi_pitch = int(pitch.replace('pitch',''))
    dur = float(dur.replace('dur',''))

    pitch_str = self.pitch2abc(midi_pitch)
    dur_str = self.duration2abc(dur)

    return pitch_str + dur_str





class LanguageModelDecoder:
  def __init__(self, vocab, save_dir='./'):
    self.vocab = vocab
    self.converter = Note2ABC()
    self.save_dir = Path(save_dir)

  # def convert_str_to_note(self, pitch_dur_str):
  #   pitch, dur = pitch_dur_str.split('//')
  #   midi_pitch = int(pitch.replace('pitch',''))
  #   dur = float(dur.replace('dur',''))

  #   pitch_str = pitch2abc(midi_pitch)
  #   dur_str = duration2abc(dur)
  #   return pitch_str + dur_str 
  def decode(self, model_pred, meta_string='X:1\nT:Title\nM:4/4\nL:1/8\nK:C\n'):
    list_of_string = self.vocab.decode(model_pred)
    abc_string = [self.converter(x) if '//' in x else x for x in list_of_string]
    
    abc_decoded = ''.join(abc_string)
    abc_decoded = meta_string + abc_decoded

    return abc_decoded

  def __call__(self, model_pred, file_code='abc_decoded_0', save_image=True, save_audio=True, meta_string='X:1\nT:Title\nM:4/4\nL:1/8\nK:C\n',):
    # list_of_string = [self.vocab[token] for token in model_pred.tolist()[1:]]
    abc_decoded = self.decode(model_pred, meta_string)
    if save_image:
      save_score_image_from_abc(abc_decoded, self.save_dir / f'{file_code}.png')
    if save_audio:
      save_wav_from_abc(abc_decoded, self.save_dir / f'{file_code}.wav')
    save_abc(abc_decoded, self.save_dir / f'{file_code}.abc')
    return abc_decoded
