import torch
import argparse
from pathlib import Path

import model_zoo
import data_utils
import vocab_utils
from decoding import LanguageModelDecoder
from tqdm.auto import tqdm


def inference(args):

  path = Path(args.path)
  if path.is_dir():
    yaml_path = list(path.glob('*.yaml'))[0]
    vocab_path = list(path.glob('*vocab.json'))[0]
    checkpoint_list = list(path.glob('*.pt'))
    checkpoint_list.sort(key= lambda x: int(x.stem.split('_')[-2].replace('iter', '')))
    checkpoint_path = checkpoint_list[-1]

    config = data_utils.read_yaml(yaml_path)
    model_name = config.nn_params.model_name
    vocab_name = config.nn_params.vocab_name
    net_param = config.nn_params

    vocab = getattr(vocab_utils, vocab_name)(json_path= vocab_path)
    config = data_utils.get_emb_total_size(config, vocab)
    model = getattr(model_zoo, model_name)(vocab.get_size(), net_param)

    checkpoint = torch.load(checkpoint_path, map_location= 'cpu')

    model.load_state_dict(checkpoint['model'])

  else:
    pass

  model.eval()
  decoder =  LanguageModelDecoder(vocab, args.save_dir)
  args.save_dir.mkdir(parents=True, exist_ok=True)

  header =  {'key':'C Major', 'meter':'4/4', 'unit note length':'1/8', 'rhythm':'reel'}
  for i in tqdm(range(args.num_samples)):
    if i % 3 == 0:
      header['key'] = 'C Major'
    elif i % 3 == 1:
      header['key'] = 'C minor'
    else:
      header['key'] = 'C Dorian'

    out = model.inference(vocab, manual_seed=i, header = header)
    file_name = f'model_{yaml_path.stem}_seed_{i}_key_{header["key"]}'
    meta_string = f'X:1\nT:Title\nM:{header["meter"]}\nL:{header["unit note length"]}\nK:{header["key"]}\n'

    try:
      decoder(out, file_name, save_image=True, save_audio=True, meta_string=meta_string)
    except Exception as e:
      print(f"decoding failed: {e}")


def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=Path, default=Path('experiments/20221007-235249-MeasureHierarchyModel_256_32_110_0.001'))
  parser.add_argument('--num_samples', type=int, default=10)
  parser.add_argument('--save_dir', type=Path, default=Path('generated'))
  return parser


if __name__ == "__main__":
  args = get_parser().parse_args()
  inference(args)