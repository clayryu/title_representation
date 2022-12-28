import torch
from torch.linalg import norm

def get_batch_contrastive_loss(emb1, emb2, margin=0.4):
  num_batch = len(emb1)
  dot_product_value = torch.matmul(emb1, emb2.T)
  emb1_norm = norm(emb1, dim=-1)
  emb2_norm = norm(emb2, dim=-1)

  cos_sim_value = dot_product_value / emb1_norm.unsqueeze(1) / emb2_norm.unsqueeze(0)
  positive_sim = cos_sim_value.diag().unsqueeze(1) # N x 1 
  non_diag_index = [x for x in range(num_batch) for y in range(num_batch) if x!=y], [y for x in range(len(cos_sim_value)) for y in range(len(cos_sim_value)) if x!=y]
  negative_sim = cos_sim_value[non_diag_index].reshape(num_batch, num_batch-1)

  loss  = torch.clamp(margin - (positive_sim - negative_sim), min=0)
  return loss.mean()