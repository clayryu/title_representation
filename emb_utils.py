from collections import defaultdict
from pathlib import Path
import torch
from torch.nn.utils.rnn import pack_sequence
import json
import time
import random

from pyabc import pyabc

def convert_token(token):
  if isinstance(token, pyabc.Note):
    return f"{token.midi_pitch}//{token.duration}"
  if isinstance(token, pyabc.Rest):
    return f"{0}//{token.duration}"
  text = token._text
  if '"' in text:
    text = text.replace('"', '')
  if text == 'M':
    return None
  if text == '\n':
    return None
  return text

def is_valid_tune(tune):
  header = tune.header
  if 'key' not in header:
    return False
  if 'meter' not in header:
    return False

  for token in tune.tokens:
    if isinstance(token, pyabc.BodyField):
      '''
      중간에 key나 meter가 바뀌는 경우 사용하지 않음
      '''
      return False
    if isinstance(token, pyabc.InlineField):
      '''
      중간에 key나 meter가 바뀌는 경우 사용하지 않음
      '''
      return False
    token_text = convert_token(token)
    if token_text == '|:1':
      return False
    if token_text == ':||:4':
      return False
    # TODO: 파트가 여러개인 경우 처리하는 부분이 필요함
  return True

def prepare_abc_title(paths: list):
    delete_list = ['Z:', 'F:', 'W:'] # F: 가 들어간 abc notation이 있는지 확인 -> 일단은 없음
    tune_list = []
    error_list = []
    filtered_tunes = []
    filtered_tunes_list = []
    title_in_text = []
    
    for path in paths:
        f = open(path)
        abc = f.readlines()
        length = len(abc)

        for line in reversed(abc):
          length -= 1
          if line[:2] in delete_list: # 지워야할 헤더 항목과 각 라인의 앞 부분이 일치하면 pop
            abc.pop(length)

        abc = ''.join(abc)
        abc = abc.replace('\\\n', '\n') # escape 문자로 \ 하나를 더 붙인 부분을 그냥 줄바꿈 기호로 치환

        try: # TODO: 같은 tunes에 묶인 tune을 필요시 구별해서 묶어야함
          tunes = pyabc.Tunes(abc=abc)
          filtered_tunes = []
          for tune in tunes.tunes:
            # tune = pyabc.Tune(abc=abc)
            if 'rhythm' not in tune.header:
              tune.header['rhythm'] = 'Unspecified'
            if 'unit note length' not in tune.header:
              tune.header['rhythm'] = '1/8'
            if is_valid_tune(tune):
              tune_list.append(tune)
              filtered_tunes.append(tune)
          
        except:
          error_list.append(path.name)
          
        if len(filtered_tunes) > 0:
          filtered_tunes_list.append(filtered_tunes)
          title_in_text.append(filtered_tunes[0].title)
          
    return tune_list, error_list, filtered_tunes_list, title_in_text
  
def pack_collate_title(raw_batch:list):
    
    melody = [pair[0] for pair in raw_batch]
    title = [pair[1] for pair in raw_batch] #pair[1] 
    
    packed_melody = pack_sequence(melody, enforce_sorted=False)
    
    return packed_melody, torch.stack(title, dim=0)
  