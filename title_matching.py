import torch
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import argparse
import wandb
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from data_utils import ABCset, MeasureNumberSet, pack_collate, PitchDurSplitSet, FolkRNNSet, MeasureOffsetSet, read_yaml, MeasureEndSet, get_emb_total_size
from emb_trainer import EmbTrainer, EmbTrainerMeasure, EmbTrainerMeasureMRR
from emb_loss import get_batch_contrastive_loss
from emb_utils import pack_collate_title
from trainer import Trainer, TrainerMeasure, TrainerPitchDur

import data_utils
import model_zoo
import emb_model
import emb_data_utils
import emb_data_utils_foraimg
import vocab_utils

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, default='abc_dataset/folk_rnn_abc_key_cleaned_title/',
                      help='directory path to the dataset')
  parser.add_argument('--yml_path', type=str, default='yamls/measure_note_xl.yaml',
                      help='yaml path to the config')

  parser.add_argument('--batch_size', type=int, default=512)
  parser.add_argument('--num_iter', type=int, default=100000)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--scheduler_factor', type=float, default=0.3)
  parser.add_argument('--scheduler_patience', type=int, default=3)
  parser.add_argument('--grad_clip', type=float, default=1.0)
  parser.add_argument('--num_epochs', type=float, default=500)

  # parser.add_argument('--hidden_size', type=int, default=256)
  # parser.add_argument('--num_layers', type=int, default=3)
  # parser.add_argument('--dropout', type=float, default=0.1)

  parser.add_argument('--aug_type', type=str, default='stat')

  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--num_workers', type=int, default=2)
  parser.add_argument('--device', type=str, default='cuda')

  parser.add_argument('--num_epoch_per_log', type=int, default=2)
  parser.add_argument('--num_iter_per_valid', type=int, default=5000)

  parser.add_argument('--model_type', type=str, default='pitch_dur')
  parser.add_argument('--abc_model_name', type=str, default='measurenote')
  parser.add_argument('--ttl_model_name', type=str, default='ttlemb')
  parser.add_argument('--save_dir', type=Path, default=Path('experiments/'))

  parser.add_argument('--no_log', action='store_true')

  return parser

def make_experiment_name_with_date(args):
  current_time_in_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return f'{current_time_in_str}-{args.model_type}_{args.batch_size}_{args.num_epochs}_{args.lr}'

if __name__ == '__main__':
  args = get_argument_parser().parse_args()
  torch.manual_seed(args.seed)

  if 'folk_rnn/data_v3' in args.path:
    score_dir = FolkRNNSet(args.path)
    vocab_path = Path(args.path).parent /  f'{args.model_type}_vocab.json'
  else:
    score_dir = Path(args.path)
    #score_dir_origin
    vocab_path = Path(args.path) / f'{args.model_type}_vocab.json'
  
  score_dir = Path(args.path)
  
  
  path = Path('pre_trained/measure_note_xl/')
  if path.is_dir():
    yaml_path = list(path.glob('*.yaml'))[0]
    vocab_path = list(path.glob('*vocab.json'))[0]
    checkpoint_list = list(path.glob('*.pt'))
    checkpoint_list.sort(key= lambda x: int(x.stem.split('_')[-2].replace('iter', '')))
    checkpoint_path = checkpoint_list[-1]
    config = data_utils.read_yaml(yaml_path)
    data_param = config.data_params
    model_name = config.nn_params.model_name
    vocab_name = config.nn_params.vocab_name
    net_param = config.nn_params
    vocab = getattr(vocab_utils, vocab_name)(json_path= vocab_path)
    config = data_utils.get_emb_total_size(config, vocab)
    #model = getattr(model_zoo, model_name)(vocab.get_size(), net_param)
    #checkpoint = torch.load(checkpoint_path, map_location= 'cpu')
    #model.load_state_dict(checkpoint['model'])
    
    dataset_name_ttl = "ABCsetTitle"
    #dataset_abc = getattr(emb_data_utils, dataset_name_ttl)(score_dir, vocab_path, make_vocab=False, key_aug=data_param.key_aug, vocab_name=net_param.vocab_name)
    #dataset_abc.vocab = vocab
    
    model_abc = getattr(model_zoo, model_name)(vocab.get_size(), net_param)
    model_abc.load_state_dict(torch.load('pre_trained/measure_note_xl/pitch_dur_iter99999_loss0.9795.pt')['model'])
    
    model_abc = getattr(model_zoo, model_name)(vocab.get_size(), net_param)
    checkpoint = torch.load(checkpoint_path, map_location= 'cpu')
    model_abc.load_state_dict(checkpoint['model'])

    model_abc_trans = getattr(emb_model, 'ABC_measnote_emb_Model')(model_abc.emb, model_abc.rnn, model_abc.measure_rnn, model_abc.final_rnn)
    model_ttl = getattr(emb_model, 'TTLembModel')()
    
    # transfer parameter
    measnote_check = torch.load('measurenote_last.pt')
    model_abc_trans.load_state_dict(measnote_check['model'])
    
    ttl_check = torch.load('ttlemb_last.pt')
    model_ttl.load_state_dict(ttl_check['model'])
    
    session_dir = Path('abc_dataset/AIMG2022/')
    session_abcs = list(session_dir.rglob('*.abc'))
    session_abcs.sort(key=lambda x: int(x.stem))
    
    ttl_list = []
    ttl_dict = {}
    with open ('abc_dataset/ttl_list.json', 'r') as f:
      titles = f.readlines()
      for idx, title in enumerate(titles):
        ttl_list.append(title)
        ttl_dict[idx] = title
      
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ttl2emb = model.encode(ttl_list, device='cuda')
    ttl2emb_torch = [torch.from_numpy(ttl) for ttl in ttl2emb]
    
    ttl_input = torch.stack(ttl2emb_torch, dim=0)
    model_ttl.eval()
    ttl_emb_output = model_ttl(ttl_input)
    
        
    dataset_abc = getattr(emb_data_utils_foraimg, dataset_name_ttl)(session_dir, vocab_path, make_vocab=False, key_aug=data_param.key_aug, vocab_name=net_param.vocab_name)
    dataset_abc.vocab = vocab
    
    data_loader = DataLoader(dataset_abc, batch_size=50, collate_fn=pack_collate_title, shuffle=False) #collate_fn=pack_collate)
    model_abc_trans.eval()
    for batch in data_loader:
      melody, title, measure_numbers = batch
      abc_emb_output = model_abc_trans(melody, measure_numbers)
    
    cos_sim = cosine_similarity(abc_emb_output.detach().cpu().numpy(), ttl_emb_output.detach().cpu().numpy()) 
    sorted_cos_idx = np.argsort(cos_sim, axis=-1)
    
    abc2ttl_dict = {'tune':[], 'title':[]}
    title_idx_list = []
    for idx in range(50):
      abc2ttl_dict['tune'].append(session_abcs[idx].stem)
      for i in range(1000):
        if sorted_cos_idx[idx][-1-i] not in title_idx_list:
          best_idx = sorted_cos_idx[idx][-1-i] 
          title_idx_list.append(best_idx)
          break
      abc2ttl_dict['title'].append(ttl_dict[best_idx])
    
    abc2ttl_df = pd.DataFrame(abc2ttl_dict)
    abc2ttl_df.to_csv('abc2ttl.csv', index=False)
      
      

    