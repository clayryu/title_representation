from torch.nn.utils.rnn import PackedSequence
import torch.nn as nn
import torch

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