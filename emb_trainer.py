import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import wandb
from wandb import Html
import numpy as np
from torch import dot
from torch.linalg import norm
from torch.nn import CosineEmbeddingLoss
import time

from sklearn.metrics.pairwise import cosine_similarity

from data_utils import decode_melody

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def get_batch_contrastive_loss(emb1, emb2, margin=0.30):
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
  def __init__(self, abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, args):
    
    self.abc_model = abc_model
    self.ttl_model = ttl_model
    self.abc_optimizer = abc_optimizer
    self.ttl_optimizer = ttl_optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    device = args.device
    self.abc_model.to(device)
    self.ttl_model.to(device)
    
    self.grad_clip = 1.0
    self.best_valid_loss = 100
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    
    self.abc_model_name = args.abc_model_name
    self.ttl_model_name = args.ttl_model_name
    
    self.make_log = not args.no_log
    
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
        loss_value, loss_dict = self._train_by_single_batch(batch)
        loss_dict = self._rename_dict(loss_dict, 'train')
        if self.make_log:
          wandb.log(loss_dict)
        self.training_loss.append(loss_value)
      self.abc_model.eval()
      self.ttl_model.eval()
      train_loss, train_acc = self.validate(external_loader=self.train_loader)
      validation_loss, validation_acc = self.validate()
      if self.make_log:
        wandb.log({
                  "validation_loss": validation_loss,
                  "validation_acc": validation_acc,
                  "train_loss": train_loss,
                  "train_acc": train_acc
                  })
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      # if validation_acc > self.best_valid_accuracy:
      if validation_loss < self.best_valid_loss:
        print(f"Saving the model with best validation loss: Epoch {epoch+1}, Loss: {validation_loss:.4f} ")
        self.save_abc_model(f'{self.abc_model_name}_best.pt')
        self.save_ttl_model(f'{self.ttl_model_name}_best.pt')
      elif num_epochs - epoch < 2:
        print(f"Saving the model with last epoch: Epoch {epoch+1}, Loss: {validation_loss:.4f} ")
        self.save_abc_model(f'{self.abc_model_name}_last.pt')
        self.save_ttl_model(f'{self.ttl_model_name}_last.pt')
      #else:
      #  self.save_abc_model(f'{self.abc_model_name}_last.pt')
      #  self.save_ttl_model(f'{self.ttl_model_name}_last.pt')                   
      self.best_valid_loss = min(validation_loss, self.best_valid_loss)
      
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (Translator/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method
    '''
    start_time = time.time()
    loss, loss_dict = self.get_loss_pred_from_single_batch(batch)
                          
    if loss != 0:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.abc_model.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.ttl_model.parameters(), self.grad_clip)
                          
        self.abc_optimizer.step()
        self.abc_optimizer.zero_grad()

        self.ttl_optimizer.step()
        self.ttl_optimizer.zero_grad()

        #loss_record.append(loss.item())
    loss_dict['time'] = time.time() - start_time
    #loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
    
    return loss.item(), loss_dict
  
  def get_loss_pred_from_single_batch(self, batch):
    melody, title = batch
    emb1 = self.abc_model(melody.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))
    
    if self.loss_fn == CosineEmbeddingLoss():
      loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
    else:
      loss = get_batch_contrastive_loss(emb1, emb2)
    loss_dict = {'total': loss.item()}
    return loss, loss_dict
    
  def validate(self, external_loader=None, topk=20):
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
    Write your code from here, using loader, self.model, self.loss_fn_fn.
    '''
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    total_sentence = 0
    correct_emb = 0
    
    abc_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size) # valid dataset size(10% of all) x embedding size
    ttl_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size)

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title = batch
        
        emb1 = self.abc_model(melody.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(title)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        
        if len(melody[3]) == 1:
            continue
        
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        else:
          loss = get_batch_contrastive_loss(emb1, emb2)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
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
      for idx in range(len(loader.dataset)):
          if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
              correct_emb += 1

        
    return validation_loss / num_total_tokens, correct_emb / len(self.valid_loader.dataset)
  
  '''
  def get_valid_loss_and_acc_from_batch(self, batch, abc_emb_all, ttl_emb_all, idx, loader):
    melody, title = batch
        
    emb1 = self.abc_model(melody.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))

    start_idx = idx * loader.batch_size
    end_idx = start_idx + len(title)

    abc_emb_all[start_idx:end_idx] = emb1
    ttl_emb_all[start_idx:end_idx] = emb2
    
    if len(melody[3]) == 1:
      continue
        
    loss = get_batch_contrastive_loss(emb1, emb2)
    
    num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
    validation_loss += loss.item() * num_tokens
    #print(validation_loss)
    num_total_tokens += num_tokens

    return validation_loss, num_total_tokens, validation_acc, loss_dict
  '''
  
  def _rename_dict(self, adict, prefix='train'):
    keys = list(adict.keys())
    for key in keys:
      adict[f'{prefix}.{key}'] = adict.pop(key)
    return dict(adict)
  
class EmbTrainerMeasure(EmbTrainer):
  def __init__(self, abc_model, ttl_model, 
               abc_optimizer, ttl_optimizer, 
               loss_fn, train_loader, valid_loader, 
               args):
    super().__init__(abc_model, ttl_model, 
               abc_optimizer, ttl_optimizer, 
               loss_fn, train_loader, valid_loader, 
               args)
    
  def get_loss_pred_from_single_batch(self, batch):
    melody, title, measure_numbers = batch
    emb1 = self.abc_model(melody.to(self.device), measure_numbers.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))
    
    if self.loss_fn == CosineEmbeddingLoss():
      loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
    else:
      loss = get_batch_contrastive_loss(emb1, emb2)
    loss_dict = {'total': loss.item()}
    return loss, loss_dict
  
  def validate(self, external_loader=None, topk=20):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.abc_model.eval()
    self.ttl_model.eval()
                          
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    total_sentence = 0
    correct_emb = 0
    
    abc_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size) # valid dataset size(10% of all) x embedding size
    ttl_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size)

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title, measure_numbers = batch
        
        emb1 = self.abc_model(melody.to(self.device), measure_numbers.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(title)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        
        if len(melody[3]) == 1: # got 1 batch
            continue
        
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        else:
          loss = get_batch_contrastive_loss(emb1, emb2)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
      '''
      # for comparing MRR loss between randomly initialized model and trained model
      abc_emb_all = torch.rand(len(self.valid_loader.dataset), 128) # valid dataset size(10% of all) x embedding size 128
      ttl_emb_all = torch.rand(len(self.valid_loader.dataset), 128)
      '''
      cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy()) # a x b mat b x a = a x a
      sorted_cos_idx = np.argsort(cos_sim, axis=-1)
      for idx in range(len(loader.dataset)):
        if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
          correct_emb += 1

        
    return validation_loss / num_total_tokens, correct_emb / len(self.valid_loader.dataset)
  
class EmbTrainerMeasureMRR(EmbTrainerMeasure):
  def __init__(self, abc_model, ttl_model, 
               abc_optimizer, ttl_optimizer, 
               loss_fn, train_loader, valid_loader, 
               args):
    super().__init__(abc_model, ttl_model, 
               abc_optimizer, ttl_optimizer, 
               loss_fn, train_loader, valid_loader, 
               args)
  
  def validate(self, external_loader=None, topk=20):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.abc_model.eval()
    self.ttl_model.eval()
                          
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    total_sentence = 0
    #correct_emb = 0
    sum_mrr = 0
    
    abc_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size) # valid dataset size(10% of all) x embedding size
    ttl_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size)

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title, measure_numbers = batch
        
        emb1 = self.abc_model(melody.to(self.device), measure_numbers.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(title)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        
        if len(melody[3]) == 1: # got 1 batch
            continue
        
        
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        else:
          loss = get_batch_contrastive_loss(emb1, emb2)
        #loss = get_batch_contrastive_loss(emb1, emb2)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
      '''
      # for comparing MRR loss between randomly initialized model and trained model
      abc_emb_all = torch.rand(len(self.valid_loader.dataset), 128) # valid dataset size(10% of all) x embedding size 128
      ttl_emb_all = torch.rand(len(self.valid_loader.dataset), 128)
      '''
      # calculate MRR
      mrrdict = {i-1:1/i for i in range(1, topk+1)} # {0:1.0, 1:0.5, ...}
      cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy()) 
      # tokens x emb(128) mat emb(128) x tokens = tokens x tokens of cosine similarity
      sorted_cos_idx = np.argsort(cos_sim, axis=-1)
      for idx in range(len(loader.dataset)):
        if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
          position = np.argwhere(sorted_cos_idx[idx][-topk:][::-1] == idx).item() # changing into ascending order
          quality_score = mrrdict[position]
          sum_mrr += quality_score

    return validation_loss / num_total_tokens, sum_mrr / len(loader.dataset)