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
from emb_loss import get_batch_contrastive_loss, get_batch_euclidean_loss, clip_crossentropy_loss

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

class EmbTrainer:
  def __init__(self, abc_model, ttl_model, loss_fn, train_loader, valid_loader, args):
    
    self.abc_model = abc_model
    self.ttl_model = ttl_model
    # self.abc_optimizer = abc_optimizer
    # self.ttl_optimizer = ttl_optimizer
    self.tau = torch.nn.Parameter(torch.tensor(1.0))
    self.abc_optimizer = torch.optim.Adam(list(abc_model.parameters()) + [self.tau], lr=args.lr)
    self.ttl_optimizer = torch.optim.Adam(list(ttl_model.parameters()) + [self.tau], lr=args.lr)
    self.margin = args.margin
    if args.lr_scheduler_type == 'Plateau':
      self.scheduler_abc = torch.optim.lr_scheduler.ReduceLROnPlateau(self.abc_optimizer, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
      self.scheduler_ttl = torch.optim.lr_scheduler.ReduceLROnPlateau(self.ttl_optimizer, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)
    elif args.lr_scheduler_type == 'Step':
      self.scheduler_abc = torch.optim.lr_scheduler.StepLR(self.abc_optimizer, step_size=args.scheduler_patience, gamma=0.5)
      self.scheduler_ttl = torch.optim.lr_scheduler.StepLR(self.ttl_optimizer, step_size=args.scheduler_patience, gamma=0.5)
    
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    device = args.device
    self.abc_model.to(device)
    self.ttl_model.to(device)
    self.tau.to(device)
    
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
      print(self.abc_optimizer.param_groups[0]['lr'])
      self.abc_model.train()
      self.ttl_model.train()
      for batch in self.train_loader:
        loss_value, loss_dict = self._train_by_single_batch(batch)
        loss_dict = self._rename_dict(loss_dict, 'train')
        # if self.make_log:
        #   wandb.log(loss_dict)
        self.training_loss.append(loss_value)
      self.abc_model.eval()
      self.ttl_model.eval()
      train_loss, train_acc = self.validate(external_loader=self.train_loader)
      validation_loss, validation_acc = self.validate()
      self.scheduler_abc.step(validation_loss)
      self.scheduler_ttl.step(validation_loss)
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
      # if validation_loss < self.best_valid_loss:
      #   print(f"Saving the model with best validation loss: Epoch {epoch+1}, Loss: {validation_loss:.4f} ")
      #   self.save_abc_model(f'{self.abc_model_name}_best.pt')
      #   self.save_ttl_model(f'{self.ttl_model_name}_best.pt')
      # elif num_epochs - epoch < 2:
      #   print(f"Saving the model with last epoch: Epoch {epoch+1}, Loss: {validation_loss:.4f} ")
      #   self.save_abc_model(f'{self.abc_model_name}_last.pt')
      #   self.save_ttl_model(f'{self.ttl_model_name}_last.pt')
      #else:
      #  self.save_abc_model(f'{self.abc_model_name}_last.pt')
      #  self.save_ttl_model(f'{self.ttl_model_name}_last.pt') 
      if epoch % 100 == 0:
        self.save_abc_model(f'{self.abc_model_name}_{epoch}.pt')
        self.save_ttl_model(f'{self.ttl_model_name}_{epoch}.pt')                
      self.best_valid_loss = min(validation_loss, self.best_valid_loss)
      
  def _train_by_single_batch(self, batch):

    start_time = time.time()
    loss, loss_dict = self.get_loss_pred_from_single_batch(batch)
                          
    if loss != 0:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(list(self.abc_model.parameters()) + [self.tau], self.grad_clip)
      torch.nn.utils.clip_grad_norm_(list(self.ttl_model.parameters()) + [self.tau], self.grad_clip)
                        
      self.abc_optimizer.step()
      self.abc_optimizer.zero_grad()

      self.ttl_optimizer.step()
      self.ttl_optimizer.zero_grad()
      
      if not isinstance(self.scheduler_abc, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.scheduler_abc.step()
        self.scheduler_ttl.step()

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
    elif self.loss_fn == get_batch_contrastive_loss:
      loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
    elif self.loss_fn == get_batch_euclidean_loss:
      loss = get_batch_euclidean_loss(emb1, emb2)
    elif self.loss_fn == clip_crossentropy_loss:
      loss = clip_crossentropy_loss(emb1, emb2, self.tau)
      
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
        elif self.loss_fn == get_batch_contrastive_loss:
          loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
        elif self.loss_fn == get_batch_euclidean_loss:
          loss = get_batch_euclidean_loss(emb1, emb2)
        elif self.loss_fn == clip_crossentropy_loss:
          loss = clip_crossentropy_loss(emb1, emb2, self.tau)
        
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
      if self.loss_fn == get_batch_contrastive_loss or self.loss_fn == CosineEmbeddingLoss:
        cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy())
        sorted_cos_idx = np.argsort(cos_sim, axis=-1)
        for idx in range(len(loader.dataset)):
            if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
                correct_emb += 1
                
      elif self.loss_fn == get_batch_euclidean_loss:
        euc_sim = torch.norm(abc_emb_all[:, None] - emb2, p=2, dim=-1)
        sorted_euc_idx = np.argsort(euc_sim, axis=-1)
        for idx in range(len(loader.dataset)):
            if idx in sorted_euc_idx[idx][-topk:]: # pick best 20 scores
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
  def __init__(self, abc_model, ttl_model, loss_fn, train_loader, valid_loader, args):
    super().__init__(abc_model, ttl_model, loss_fn, train_loader, valid_loader, args)
    
  def get_loss_pred_from_single_batch(self, batch):
    melody, title, measure_numbers = batch
    emb1 = self.abc_model(melody.to(self.device), measure_numbers.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))
    
    if self.loss_fn == CosineEmbeddingLoss():
      loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
    elif self.loss_fn == get_batch_contrastive_loss:
      loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
    elif self.loss_fn == get_batch_euclidean_loss:
      loss = get_batch_euclidean_loss(emb1, emb2)
    elif self.loss_fn == clip_crossentropy_loss:
      loss = clip_crossentropy_loss(emb1, emb2, self.tau)
      
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
        elif self.loss_fn == get_batch_contrastive_loss:
          loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
        elif self.loss_fn == get_batch_euclidean_loss:
          loss = get_batch_euclidean_loss(emb1, emb2)
        elif self.loss_fn == clip_crossentropy_loss:
          loss = clip_crossentropy_loss(emb1, emb2, self.tau)
        
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
  def __init__(self, abc_model, ttl_model, loss_fn, train_loader, valid_loader, args):
    super().__init__(abc_model, ttl_model, loss_fn, train_loader, valid_loader, args)
  
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
        end_idx = start_idx + len(emb1)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        
        if len(melody[3]) == 1: # got 1 batch
            continue
        
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        elif self.loss_fn == get_batch_contrastive_loss:
          loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
        elif self.loss_fn == get_batch_euclidean_loss:
          loss = get_batch_euclidean_loss(emb1, emb2)
        elif self.loss_fn == clip_crossentropy_loss:
          loss = clip_crossentropy_loss(emb1, emb2, self.tau)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
      '''
      # for comparing MRR loss between randomly initialized model and trained model
      abc_emb_all = torch.rand(len(self.valid_loader.dataset), 128) # valid dataset size(10% of all) x embedding size 128
      ttl_emb_all = torch.rand(len(self.valid_loader.dataset), 128)
      '''
      
      if self.loss_fn == get_batch_contrastive_loss or self.loss_fn == CosineEmbeddingLoss or self.loss_fn == clip_crossentropy_loss:
        # calculate MRR
        mrrdict = {i-1:1/i for i in range(1, topk+1)} # {0:1.0, 1:0.5, ...}
        cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy()) 
        # tokens x emb(128) mat emb(128) x tokens = tokens x tokens of cosine similarity
        sorted_cos_idx = np.argsort(cos_sim, axis=-1) # the most similar one goes to the end
        for idx in range(len(loader.dataset)):
          if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
            position = np.argwhere(sorted_cos_idx[idx][-topk:][::-1] == idx).item() # changing into ascending order
            quality_score = mrrdict[position]
            sum_mrr += quality_score
                
      elif self.loss_fn == get_batch_euclidean_loss:
        mrrdict = {i-1:1/i for i in range(1, topk+1)} # {0:1.0, 1:0.5, ...}
        euc_sim = torch.norm(abc_emb_all[:, None] - ttl_emb_all, p=2, dim=-1)
        sorted_euc_idx = torch.argsort(euc_sim, dim=-1)
        for idx in range(len(loader.dataset)):
          if idx in sorted_euc_idx[idx][-topk:]: # pick best 20 scores
            position = torch.argwhere(sorted_euc_idx[idx][-topk:].flip(-1) == idx).item() # changing into ascending order
            quality_score = mrrdict[position]
            sum_mrr += quality_score

    return validation_loss / num_total_tokens, sum_mrr / len(loader.dataset)