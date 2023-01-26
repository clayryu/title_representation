from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch.nn as nn
import torch

class MultiEmbeddingreduced(nn.Module):
  def __init__(self, vocab_sizes: dict, vocab_param, ratio) -> None:
    super().__init__()
    self.layers = []
    embedding_sizes = self.get_embedding_size(vocab_sizes, vocab_param)
    # if isinstance(embedding_sizes, int):
    #   embedding_sizes = [embedding_sizes] * len(vocab_sizes)
    for vocab_size, embedding_size in zip(vocab_sizes.values(), embedding_sizes):
      if int(embedding_size * ratio) != 0:
        self.layers.append(nn.Embedding(vocab_size, int(embedding_size * ratio)))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    # num_embeddings = torch.tensor([x.num_embeddings for x in self.layers])
    # max_indices = torch.max(x, dim=0)[0].cpu()
    # assert (num_embeddings > max_indices).all(), f'num_embeddings: {num_embeddings}, max_indices: {max_indices}'
    return torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)

  def get_embedding_size(self, vocab_sizes, vocab_param):
    embedding_sizes = [getattr(vocab_param, vocab_key) for vocab_key in vocab_sizes.keys()]
    return embedding_sizes

class ABC_meas_emb_Model(nn.Module):
  def __init__(self, trans_emb, trans_rnn, emb_size=128):
    super().__init__()
    self.hidden_size = trans_rnn.hidden_size
    self.emb = trans_emb
    self.rnn = trans_rnn
    self.proj = nn.Linear(self.hidden_size, emb_size)
    
  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      hidden, last_hidden = self.rnn(emb)
      hidden_emb = last_hidden.data[-1] # 1 x 256
      final_emb = self.proj(hidden_emb) # 1 x 128
      
    return final_emb
  
class ABC_measnote_emb_Model(nn.Module):
  def __init__(self, trans_emb=None, trans_rnn=None, trans_measure_rnn=None, trans_final_rnn=None, emb_size=256):
    super().__init__()
    self.emb_size = emb_size
    self.emb = trans_emb
    self.rnn = trans_rnn
    self.measure_rnn = trans_measure_rnn
    self.final_rnn = trans_final_rnn
    self.emb_rnn = nn.GRU(input_size=512, hidden_size=128, num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
    self.hidden_size = self.emb_rnn.hidden_size * 2 * self.emb_rnn.num_layers 
    self.proj = nn.Linear(self.hidden_size, self.emb_size)
    
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      return emb
    else:
      pass
    
  def forward(self, input_seq, measure_numbers):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([hidden.data, measure_hidden.data], dim=-1), hidden.batch_sizes, hidden.sorted_indices, hidden.unsorted_indices)
      
      final_hidden, _ = self.final_rnn(cat_hidden)
      emb_hidden, last_emb_hidden = self.emb_rnn(final_hidden)
      # last_emb_hidden.data.shape # torch.Size([4, 32, 128])
      extr = last_emb_hidden.data.transpose(0,1) # torch.Size([32, 4, 128])
      extr_batch = extr.reshape(len(input_seq.sorted_indices),-1) # torch.Size([32, 512])
      batch_emb = self.proj(extr_batch) # torch.Size([32, 128])
      # batch_emb = batch_emb[emb_hidden.unsorted_indices] # for title matching, we need to sort the batch_emb
      return batch_emb
    else:
      raise NotImplementedError
  
  '''
  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      hidden, last_hidden = self.rnn(emb)
      hidden_emb = last_hidden.data[-1] # 1 x 256
      final_emb = self.proj(hidden_emb) # 1 x 128
      
    return final_emb
  '''
  
class TTLembModel(nn.Module): 
    def __init__(self, in_embedding_size=384, hidden_size=256, emb_size=256):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_embedding_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(hidden_size, hidden_size//2),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(hidden_size//2, emb_size)
                                    )
        
    def forward(self, x):
        '''
        x (torch.FloatTensor): N x Feature
        '''
        return self.layer(x)

# cnn layer 파라미터의 개수를 구해보자. 파라미터의 개수를 알아야 지금 모델이 내가 감당할 만한 모델인지 알 수 있다.
# kernel의 목적은 보고 있는 timestep 혹은 픽셀 정보를 1개의 뉴런으로 바꾸어주는 곱연산이다. 5 to 1
# 그리고 채널이란 독립적으로 존재하는 timestep 혹은 픽셀의 정보다. 이는 종합적으로 사용이 될 개별 정보로서 채널은 절대로 연속된 정보가 아니다.
# 따라서 kernel size x channel size
class ABC_cnn_emb_Model(nn.Module):
  def __init__(self, trans_emb=None, vocab_size=None, net_param=None, emb_size=256, hidden_size=128, emb_ratio=1):
    super().__init__()
    self.emb_size = emb_size
    self.emb_ratio = emb_ratio
    self.hidden_size = hidden_size
    if vocab_size is not None and net_param is not None and trans_emb is None:
      self.vocab_size_dict = vocab_size
      self.net_param = net_param
      self._make_embedding_layer()
    elif trans_emb is not None:
      self.emb = trans_emb
    
    self.emb_total_list = [x.embedding_dim for x in self.emb.layers]
    self.emb_total_size = sum(self.emb_total_list)
    
    self.conv_layer = nn.Sequential(
      nn.Conv1d(in_channels=self.emb_total_size, out_channels=self.hidden_size, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm1d(self.hidden_size),
      nn.ReLU(),
      nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm1d(self.hidden_size),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.MaxPool1d(2),
      nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm1d(self.hidden_size),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.MaxPool1d(2),
      nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm1d(self.hidden_size),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.MaxPool1d(2),
      # nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=0),
      # nn.BatchNorm1d(self.hidden_size),
      # nn.ReLU(),
      # nn.Dropout(0.5),
      # nn.MaxPool1d(2),
      nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm1d(self.hidden_size),
      nn.ReLU(),
      nn.AdaptiveMaxPool1d(1),
    )
    self.linear_layer = nn.Sequential(
      nn.Linear(self.hidden_size, emb_size)
      # nn.Linear(256, 512),
      # nn.ReLU(),
      # nn.Linear(512, emb_size),
    )
    
  def _make_embedding_layer(self):
    self.emb = MultiEmbeddingreduced(self.vocab_size_dict, self.net_param.emb, self.emb_ratio)
    
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      return emb
    else:
      pass
    
  def forward(self, input_seq, measure_numbers):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      unpacked_emb, _ = pad_packed_sequence(emb, batch_first=True)
      unpacked_emb = unpacked_emb.transpose(1,2)
      after_conv = self.conv_layer(unpacked_emb)
      before_linear = after_conv.view(after_conv.size(0), -1)
      batch_emb = self.linear_layer(before_linear)
      
      return batch_emb
    else:
      raise NotImplementedError