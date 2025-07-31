import torch
import numpy as np
from config.config import PSEUDO_NEG_K

@torch.no_grad()
def compute_centroids(encoder, D1_tensor, Neg_tensor=None, pseudo_neg_tensor=None):
    encoder.eval()

    # v1: mean of known positives
    z_D1_all = encoder(D1_tensor)
    v1 = z_D1_all.mean(dim=0)

    # v0: mean of (Neg ∪ pseudo-negatives)
    if Neg_tensor is not None and pseudo_neg_tensor is not None:
        z_neg_all = encoder(Neg_tensor)
        z_pneg_all = encoder(pseudo_neg_tensor)
        v0 = torch.cat([z_neg_all, z_pneg_all], dim=0).mean(dim=0)
    else:
        v0 = None

    return v1, v0


@torch.no_grad()
def extract_embeddings(encoder, data_tensor):
    encoder.eval()
    return encoder(data_tensor)


@torch.no_grad()
def assign_pseudo_labels(encoder, data_tensor, v1, v0):
    encoder.eval()
    z = encoder(data_tensor)
    dist_to_v1 = torch.norm(z - v1, dim=1)
    dist_to_v0 = torch.norm(z - v0, dim=1)
    labels = (dist_to_v1 < dist_to_v0).long().cpu().numpy()
    return labels, z.cpu().numpy()


@torch.no_grad()
def mine_pseudo_negatives(encoder, D1_tensor, Unlabeled_tensor, top_k=PSEUDO_NEG_K):
    encoder.eval()
    
    z_D1 = encoder(D1_tensor)
    v_D = z_D1.mean(dim=0)
    
    z_Unl = encoder(Unlabeled_tensor)
    dist_to_vD = torch.norm(z_Unl - v_D, dim=1)

    farthest_idx = torch.topk(dist_to_vD, k=top_k).indices.cpu().numpy()
    pseudo_neg_tensor = Unlabeled_tensor[farthest_idx]

    return pseudo_neg_tensor
