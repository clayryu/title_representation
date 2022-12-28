import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import wandb
from data_utils import decode_melody
from wandb import Html

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from torch import dot
from torch.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def get_batch_contrastive_loss(emb1, emb2, margin=0.4):
  # batch가 1일 경우 계산이 되질 않는다.
  num_batch = len(emb1)
  if num_batch == 1:
    return torch.tensor(0)
  dot_product_value = torch.matmul(emb1, emb2.T)
  emb1_norm = norm(emb1, dim=-1) + 1e-6
  emb2_norm = norm(emb2, dim=-1) + 1e-6

  cos_sim_value = dot_product_value / emb1_norm.unsqueeze(1) / emb2_norm.unsqueeze(0)
  positive_sim = cos_sim_value.diag().unsqueeze(1) # N x 1 
  non_diag_index = [x for x in range(num_batch) for y in range(num_batch) if x!=y], [y for x in range(len(cos_sim_value)) for y in range(len(cos_sim_value)) if x!=y]
  negative_sim = cos_sim_value[non_diag_index].reshape(num_batch, num_batch-1)

  loss  = torch.clamp(margin - (positive_sim - negative_sim), min=0)
  return loss.mean()


class EmbTrainer:
  def __init__(self, abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, device, abc_model_name='abc_model', ttl_model_name='ttl_model'):
    
    self.abc_model = abc_model
    self.ttl_model = ttl_model
    self.abc_optimizer = abc_optimizer
    self.ttl_optimizer = ttl_optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    self.abc_model.to(device)
    self.ttl_model.to(device)
    
    self.grad_clip = 1.0
    self.best_valid_loss = 100
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    
    self.abc_model_name = abc_model_name
    self.ttl_model_name = ttl_model_name
    
    '''
    if isinstance(self.train_loader.dataset, torch.utils.data.dataset.Subset):
      vocab = self.train_loader.dataset.dataset.vocab
    else:
      vocab = self.train_loader.dataset.vocab
    self.vocab = vocab
    '''

  def save_abc_model(self, path):
    torch.save({'model':self.abc_model.state_dict(), 'optim':self.abc_optimizer.state_dict()}, path)
  
  def save_ttl_model(self, path):
    torch.save({'model':self.ttl_model.state_dict(), 'optim':self.ttl_optimizer.state_dict()}, path)
    
  def train_by_num_epoch(self, num_epochs):
    for epoch in tqdm(range(num_epochs)):
      self.abc_model.train()
      self.ttl_model.train()
      for batch in self.train_loader:
        loss_value = self._train_by_single_batch(batch)
        wandb.log({"training_loss": loss_value})
        self.training_loss.append(loss_value)
      self.abc_model.eval()
      self.ttl_model.eval()
      validation_loss, validation_acc = self.validate()
      wandb.log({"validation_loss": validation_loss,
                "validation_acc": validation_acc})
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      # if validation_acc > self.best_valid_accuracy:
      if validation_loss < self.best_valid_loss:
        print(f"Saving the model with best validation loss: Epoch {epoch+1}, Loss: {validation_loss:.4f} ")
        self.save_abc_model(f'{self.abc_model_name}_best.pt')
        self.save_ttl_model(f'{self.ttl_model_name}_best.pt')
      else:
        self.save_abc_model(f'{self.abc_model_name}_last.pt')
        self.save_ttl_model(f'{self.ttl_model_name}_last.pt')                   
      self.best_valid_loss = min(validation_loss, self.best_valid_loss)
      
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (Translator/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method
    '''
    
    melody, title = batch
        
    emb1 = self.abc_model(melody.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))
    
    loss = 0
    #pos_R = cos_sim(emb1[0], emb2[0]) # positive pair의 cosine 유사도 값
        
    loss = get_batch_contrastive_loss(emb1, emb2, margin=0.5)
                          
    if loss != 0:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.abc_model.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.ttl_model.parameters(), self.grad_clip)
                          
        self.abc_optimizer.step()
        self.abc_optimizer.zero_grad()

        self.ttl_optimizer.step()
        self.ttl_optimizer.zero_grad()

        #loss_record.append(loss.item())

    return loss.item()

    
  def validate(self, external_loader=None):
    '''
    This method calculates accuracy and loss for given data loader.
    It can be used for validation step, or to get test set result
    
    input:
      data_loader: If there is no data_loader given, use self.valid_loader as default.
      
    output: 
      validation_loss (float): Mean Binary Cross Entropy value for every sample in validation set
      validation_accuracy (float): Mean Accuracy value for every sample in validation set
    '''
    
    ### Don't change this part
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.abc_model.eval()
    self.ttl_model.eval()
                          
    '''
    Write your code from here, using loader, self.model, self.loss_fn.
    '''
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    total_sentence = 0
    correct_emb = 0
    
    abc_emb_all = torch.zeros(len(self.valid_loader.dataset), 128) # valid dataset size 1089 x embedding size 128
    ttl_emb_all = torch.zeros(len(self.valid_loader.dataset), 128)

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title = batch
        
        emb1 = self.abc_model(melody.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(title)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2

        # if idx == len(self.valid_loader)-1: # batch num 18
        #     last_space = len(self.valid_loader.dataset) % len(melody[3]) # total dataset 1089 % batch size 64
        #     abc_emb_all[len(self.valid_loader.dataset)-2:] = emb1 # 마지막 배치 size와 남은 리스트 크기가 일치 하지 않는다
        #     ttl_emb_all[len(self.valid_loader.dataset)-2:] = emb2
        # else:
        #     abc_emb_all[64*idx:64*(idx+1)] = emb1
        #     ttl_emb_all[64*idx:64*(idx+1)] = emb2
        
        # if idx == len(self.valid_loader)-1:
        #     cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy())
        #     sorted_cos_idx = np.argsort(cos_sim, axis=-1)
        #     for idx in range(len(self.valid_loader.dataset)):
        #         if idx in sorted_cos_idx[idx][-21:-1]: # pick best 20 scores
        #             correct_emb += 1
        
        if len(melody[3]) == 1:
            continue
        
        loss = get_batch_contrastive_loss(emb1, emb2, margin=0.5)
        
        num_tokens = melody.data.shape[0]
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
        
        '''
        cos_emb1 = emb1.detach().cpu().numpy()
        cos_emb2 = emb2.detach().cpu().numpy()
        # sklearn 사용을 위해서 numpy로 바꿔주고 다시 tensor로 바꿔 대각 요소들을 빼내어 1대1 비교값의 리스트를 만든다.
        cos_emb = torch.tensor(cosine_similarity(cos_emb1, cos_emb2)).diag() 
        acc = torch.sum(cos_emb > 0.5)

        validation_acc += acc.item()
        total_sentence += len(melody[3]) # packed sequence의 3번째 리스트는 배치된 문장의 순서이다.
        '''

      cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy())
      sorted_cos_idx = np.argsort(cos_sim, axis=-1)
      for idx in range(len(self.valid_loader.dataset)):
          if idx in sorted_cos_idx[idx][-21:-1]: # pick best 20 scores
              correct_emb += 1

        
    return validation_loss / num_total_tokens, correct_emb / len(self.valid_loader.dataset)
