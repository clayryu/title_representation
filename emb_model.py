from torch.nn.utils.rnn import PackedSequence
import torch.nn as nn

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
  
class ABC_measn_emb_Model(nn.Module):
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
  
class TTLembModel(nn.Module): 
    def __init__(self, in_embedding_size=384, hidden_size=256, out_embedding_size=128):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_embedding_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, out_embedding_size),
                                   nn.ReLU(),
                                   nn.Linear(out_embedding_size, out_embedding_size)
                                    )
        
    def forward(self, x):
        '''
        x (torch.FloatTensor): N x Feature
        '''
        return self.layer(x)