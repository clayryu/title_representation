import torch
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_utils import ABCset, MeasureNumberSet, pack_collate, PitchDurSplitSet, FolkRNNSet, MeasureOffsetSet, read_yaml, MeasureEndSet, ABCsetTitle, get_emb_total_size
from emb_model import ABC_measn_emb_Model, TTLembModel
from emb_trainer import EmbTrainer
from emb_loss import get_batch_contrastive_loss
from emb_utils import pack_collate_title

import data_utils
import model_zoo
import emb_model

from trainer import Trainer, TrainerMeasure, TrainerPitchDur
import argparse
import wandb
import datetime

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, default='abc_dataset/folk_rnn_abc_key_cleaned/',
                      help='directory path to the dataset')
  parser.add_argument('--yml_path', type=str, default='yamls/measure_note_xl.yaml',
                      help='yaml path to the config')

  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--num_iter', type=int, default=100000)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--scheduler_factor', type=float, default=0.3)
  parser.add_argument('--scheduler_patience', type=int, default=3)
  parser.add_argument('--grad_clip', type=float, default=1.0)

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
  parser.add_argument('--model_name', type=str, default='pitch_dur')
  parser.add_argument('--save_dir', type=Path, default=Path('experiments/'))

  parser.add_argument('--no_log', action='store_true')

  return parser

def make_experiment_name_with_date(args):
  current_time_in_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return f'{current_time_in_str}-{args.model_type}_{args.batch_size}_{args.num_epochs}_{args.lr}_{args.hidden_size}'

if __name__ == '__main__':
  args = get_argument_parser().parse_args()
  torch.manual_seed(args.seed)

  config = read_yaml(args.yml_path)
  net_param = config.nn_params
  data_param = config.data_params
  model_name = net_param.model_name
  if hasattr(net_param, 'dataset_name'):
    dataset_name = net_param.dataset_name
  elif model_name == "PitchDurModel":
    dataset_name = "PitchDurSplitSet"
  elif model_name in ["MeasureHierarchyModel", "MeasureNoteModel", "MeasureNotePitchFirstModel"]:
    dataset_name = "MeasureNumberSet"
    dataset_name_ttl = "ABCsetTitle"
  elif model_name == "MeasureInfoModel":
    dataset_name = "MeasureOffsetSet"
  elif model_name == "LanguageModel":
    dataset_name = ABCset
  else:
    raise NotImplementedError
  
  if 'folk_rnn/data_v3' in args.path:
    score_dir = FolkRNNSet(args.path)
    vocab_path = Path(args.path).parent /  f'{args.model_type}_vocab.json'
  else:
    score_dir = Path(args.path)
    vocab_path = Path(args.path) / f'{args.model_type}_vocab.json'
  
  score_dir = Path(args.path)

  dataset_abc = getattr(data_utils, dataset_name)(score_dir, vocab_path, key_aug=data_param.key_aug, vocab_name=net_param.vocab_name, num_limit=100)
  dataset_ttl = getattr(data_utils, dataset_name_ttl)(score_dir, vocab_path, vocab_name=net_param.vocab_name, num_limit=100)
  
  config = data_utils.get_emb_total_size(config, dataset_abc.vocab)
  net_param = config.nn_params
  
  model_abc = getattr(model_zoo, model_name)(dataset_abc.vocab.get_size(), net_param)
  model_abc.load_state_dict(torch.load('pretrained/pitch_dur_iter99999_loss0.9795.pt')['model'])

  model_abc_trans = getattr(emb_model, 'ABC_measn_emb_Model')(model_abc.emb, model_abc.rnn, emb_size=128)
  model_ttl = TTLembModel().to(args.device)

  wandb.watch(model1)
  wandb.watch(model2)
  
  optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
  optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

  loss_fn = get_batch_contrastive_loss

  trainset, validset = torch.utils.data.random_split(dataset_ttl, [int(len(dataset_ttl)*0.9), len(dataset_ttl) - int(len(dataset_ttl)*0.9)], generator=torch.Generator().manual_seed(42))

  train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=pack_collate_title, shuffle=True) #collate_fn=pack_collate)
  valid_loader = DataLoader(validset, batch_size=args.batch_size, collate_fn=pack_collate_title, shuffle=False) #collate_fn=pack_collate)

  experiment_name = make_experiment_name_with_date(args)
  save_dir = args.save_dir / experiment_name
  save_dir.mkdir(parents=True, exist_ok=True)

  trainer = EmbTrainer(model1, model2, optimizer1, optimizer2, loss_fn, train_loader, valid_loader, device=args.device)
  
  if not args.no_log:
    wandb.init(project="irish-maler", entity="maler", config={**vars(args), **config})
    # wandb.config.update({**vars(args), **config})
    wandb.watch(model_abc)
  
  trainer.train_by_num_epoch(args.num_epochs)
  
  