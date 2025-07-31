import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from models.loss import KLContrastivePULoss
from utils.dataset import SessionDataset
from config.config import BATCH_SIZE, NUM_EPOCHS_STAGE1, LEARNING_RATE_STAGE1
from utils.save_utils import save_plot

def stage1_train(encoder, D1_tensor, Unl_tensor):
    contrastive_loss = KLContrastivePULoss(temperature=0.5)
    optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE_STAGE1)
    
    D1_loader = DataLoader(SessionDataset(D1_tensor), batch_size=BATCH_SIZE, shuffle=True)
    Unl_loader = DataLoader(SessionDataset(Unl_tensor), batch_size=BATCH_SIZE, shuffle=True)

    loss_history = []

    for epoch in range(NUM_EPOCHS_STAGE1):
        encoder.train()
        total_loss = 0.0

        # Compute tau
        raw_tau = np.std(D1_tensor.numpy()) / np.log(1 + epoch + 1)
        tau = max(raw_tau, 2.0)
        print(f"[Stage 1 | Epoch {epoch+1}/{NUM_EPOCHS_STAGE1}] raw_tau = {raw_tau:.4f}, tau = {tau:.4f}")

        for batch_pos, batch_unl in zip(D1_loader, Unl_loader):
            optimizer.zero_grad()
            min_bs = min(batch_pos.size(0), batch_unl.size(0))
            x_pos = batch_pos[:min_bs]
            x_unl = batch_unl[:min_bs]

            z_pos = encoder(x_pos)
            z_unl = encoder(x_unl)

            v_D = z_pos.mean(dim=0)
            v_1 = v_D
            v_0 = (v_D - tau * v_1) / tau

            dist_to_v1 = torch.norm(z_unl - v_1, dim=1)
            dist_to_v0 = torch.norm(z_unl - v_0, dim=1)
            label_indicator = (dist_to_v1 < dist_to_v0).float()

            uncertainty_weight = torch.norm(z_unl - v_D, dim=1)

            loss = contrastive_loss(z_pos, z_unl, label_indicator, uncertainty_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss)
        print(f"  → Stage 1 Loss = {total_loss:.4f}")

    return loss_history



def plot_stage1_loss(loss_history, output_dir):
    fig = plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 1: ConPU Loss")
    plt.tight_layout()
    save_plot(fig, "stage1_loss.png", output_dir)
