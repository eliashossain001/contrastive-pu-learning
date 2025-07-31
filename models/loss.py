import torch
import torch.nn as nn
import torch.nn.functional as F

class KLContrastivePULoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, z_i, z_j, label_indicator, uncertainty_weight):
        # z_i, z_j: (batch_size, output_dim)
        # label_indicator: (batch_size,) values in {0, 1}
        # uncertainty_weight: (batch_size,) ≥ 0
        cosine_similarity = nn.CosineSimilarity(dim=-1)

        pos_sim = torch.exp(cosine_similarity(z_i, z_j) / self.temperature)
        neg_sim = (
            torch.exp(cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature)
            .sum(dim=-1) + 1e-9
        )
        contrastive_term = -torch.log(pos_sim / neg_sim)

        kl_div = self.kl_loss(
            torch.log_softmax(z_i, dim=-1),
            torch.softmax(z_j, dim=-1)
        )

        weighted_loss = uncertainty_weight * contrastive_term
        return torch.mean(weighted_loss) + kl_div


def triplet_loss_anchor_positive_negative(z_anchor, z_positive, z_negative, margin=1.0):
    """
    Triplet loss to enforce:
        ||anchor - positive||^2 + margin <= ||anchor - negative||^2
    """
    dist_pos = torch.norm(z_anchor - z_positive, dim=1)
    dist_neg = torch.norm(z_anchor - z_negative, dim=1)
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()
