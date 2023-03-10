import torch
import shutil
import copy
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_utils import ABCset, MeasureNumberSet, pack_collate, PitchDurSplitSet, FolkRNNSet, MeasureOffsetSet, read_yaml, MeasureEndSet, get_emb_total_size
from emb_trainer import EmbTrainer, EmbTrainerMeasure, EmbTrainerMeasureMRR
from emb_loss import ContrastiveLoss, ContrastiveLoss_euclidean, clip_crossentropy_loss
from emb_utils import pack_collate_title_sampling_textttl, pack_collate_title_sampling_train, pack_collate_title_sampling_valid
from torch.nn import CosineEmbeddingLoss

import data_utils
import model_zoo
import emb_model
import emb_data_utils
import vocab_utils

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

  parser.add_argument('--batch_size', type=int, default=3000)
  parser.add_argument('--num_iter', type=int, default=100000)
  parser.add_argument('--lr', type=float, default=0.00001)
  parser.add_argument('--lr_scheduler_type', type=str, default='Plateau')
  parser.add_argument('--scheduler_factor', type=float, default=0.7)
  parser.add_argument('--scheduler_patience', type=int, default=9000)
  parser.add_argument('--grad_clip', type=float, default=1.0)
  parser.add_argument('--num_epochs', type=float, default=9000)

  parser.add_argument('--abc_hidden_size', type=int, default=144)
  parser.add_argument('--ttl_hidden_size', type=int, default=256)
  parser.add_argument('--output_emb_size', type=int, default=256)
  parser.add_argument('--loss_margin', type=float, default=0.1)
  parser.add_argument('--dataset_tune_length', type=int, default=100)
  parser.add_argument('--topk', type=int, default=0)
  # 'contrastive_euclidean', 'contrastive_cosine'
  parser.add_argument('--loss_type', type=str, default='contrastive_cosine')
  # you should check emb_utils.py to configure the sampling size of the token manually
  # parser.add_argument('--L2_regularization')

  parser.add_argument('--aug_type', type=str, default='stat')

  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--num_workers', type=int, default=2)
  parser.add_argument('--device', type=str, default='cuda')

  parser.add_argument('--num_epoch_per_log', type=int, default=2)
  parser.add_argument('--num_iter_per_valid', type=int, default=5000)

  parser.add_argument('--model_type', type=str, default='cnn_reducedemb')
  parser.add_argument('--abc_model_name', type=str, default='ABC_cnn_emb_Model')
  parser.add_argument('--ttl_model_name', type=str, default='TTLembModel')
  parser.add_argument('--name_of_model_to_save', type=str, default='ST_margin0.1_lr0.00003_150tk_ker20333_abc144_reel')

  # ST_embedding_withsongtitle, OA_embedding_withsongtitle, OA_embedding_withsongtitle_genre, OA_embedding_titleonly
  parser.add_argument('--pretrnd_ttl_emb_type', type=str, default='ST_titleonly_6283')
  parser.add_argument('--dataset_name_ttl', type=str, default='ABCsetTitle_vartune')
  # 'list300_ttl-ttlsong.pkl'
  parser.add_argument('--ambiguous_title', type=str, default=None)
  parser.add_argument('--ambiguous_title_delnum', type=int, default=100)
  # 'not_english_title_azure.pkl'
  parser.add_argument('--language_detect', type=str, default=None)

  parser.add_argument('--save_dir', type=Path, default=Path('experiments/'))
  parser.add_argument('--no_log', action='store_true')

  return parser

def make_experiment_name_with_date(args):
  current_time_in_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return f'{current_time_in_str}-{args.model_type}_{args.batch_size}_{args.num_epochs}_{args.lr}'

if __name__ == '__main__':
  #torch.backends.cudnn.enabled = False # 
  
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
  
  dataset_abc = getattr(emb_data_utils, args.dataset_name_ttl)(score_dir, vocab_path, make_vocab=False, key_aug=data_param.key_aug, vocab_name=net_param.vocab_name, args=args, pre_vocab=vocab)
  #dataset_abc.vocab = vocab
  #dataset_abc.vocab_dictionary = vocab_utils.Vocab_dictionary(dataset_abc.vocab)
  #dataset_abc.vocab_tok2idx = copy.deepcopy(vocab.tok2idx)
  
  #rnn_gen_model = getattr(model_zoo, model_name)(vocab.get_size(), net_param)
  #rnn_gen_model.load_state_dict(torch.load('pre_trained/measure_note_xl/pitch_dur_iter99999_loss0.9795.pt')['model'])
  
  rnn_gen_model = getattr(model_zoo, model_name)(vocab.get_size(), net_param)
  checkpoint = torch.load(checkpoint_path, map_location= 'cpu')
  rnn_gen_model.load_state_dict(checkpoint['model'])

  model_rnn_trans = getattr(emb_model, 'ABC_measnote_emb_Model')(rnn_gen_model.emb, rnn_gen_model.rnn, rnn_gen_model.measure_rnn, rnn_gen_model.final_rnn, emb_size=args.output_emb_size)
  model_cnn_trans = getattr(emb_model, args.abc_model_name)(trans_emb=rnn_gen_model.emb, emb_size=args.output_emb_size)
  model_cnn_reducedemb = getattr(emb_model, args.abc_model_name)(trans_emb=None, vocab_size=vocab.get_size(), net_param=net_param, emb_size=args.output_emb_size, hidden_size=args.abc_hidden_size, emb_ratio=1)
  ttl_in_embedding_size = 768 if args.pretrnd_ttl_emb_type[:2] == 'ST' else 1536
  model_ttl = getattr(emb_model, args.ttl_model_name)(in_embedding_size=ttl_in_embedding_size, hidden_size=args.ttl_hidden_size ,emb_size=args.output_emb_size)
  
  # load pretrained model
  # model_cnn_reducedemb.load_state_dict(torch.load('/home/clay/userdata/title_generation/measurenote_last copy.pt')['model'])
  # model_ttl.load_state_dict(torch.load('/home/clay/userdata/title_generation/ttlemb_last copy.pt')['model'])
  
  '''
  # freeze all parameters except proj
  for para in rnn_gen_model_trans.parameters():
    para.requires_grad = False
  for name, param in rnn_gen_model_trans.named_parameters():
    if name in ['proj.weight', 'proj.bias']:
      param.requires_grad = True
  '''
  if args.model_type == 'rnn_trans': 
    model = model_rnn_trans
  elif args.model_type == 'cnn_trans':
    model = model_cnn_trans
  elif args.model_type == 'cnn_reducedemb':
    model = model_cnn_reducedemb
  
  optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr)
  optimizer2 = torch.optim.Adam(model_ttl.parameters(), lr=args.lr)

  if args.loss_type == 'contrastive_euclidean':
    loss_fn = ContrastiveLoss_euclidean(margin=args.loss_margin)
  elif args.loss_type == 'contrastive_cosine':
    loss_fn = ContrastiveLoss(margin=args.loss_margin)
  elif args.loss_type == 'clip':
    loss_fn = clip_crossentropy_loss

  trainset, validset = torch.utils.data.random_split(dataset_abc, [int(len(dataset_abc)*0.9), len(dataset_abc) - int(len(dataset_abc)*0.9)], generator=torch.Generator().manual_seed(42))

  train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=pack_collate_title_sampling_textttl, shuffle=True, num_workers=0) #collate_fn=pack_collate)
  valid_loader = DataLoader(validset, batch_size=args.batch_size, collate_fn=pack_collate_title_sampling_textttl, shuffle=False, num_workers=0) #collate_fn=pack_collate)

  experiment_name = make_experiment_name_with_date(args)
  save_dir = args.save_dir / experiment_name
  save_dir.mkdir(parents=True, exist_ok=True)

  trainer = EmbTrainerMeasureMRR(model, model_ttl, loss_fn, train_loader, valid_loader, args)
  
  
  if not args.no_log:
    wandb.init(project="title-embedding-mrr", entity="clayryu", config={**vars(args), **config})
    # wandb.config.update({**vars(args), **config})
    wandb.watch(model)
    wandb.watch(model_ttl)
  
    trainer.train_by_num_epoch(args.num_epochs)
  print(f'mean of val_acc : {sum(trainer.validation_acc) / len(trainer.validation_acc)}')
  
  