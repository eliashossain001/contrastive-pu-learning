import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from utils.dataset import SessionDataset
from models.loss import triplet_loss_anchor_positive_negative
from config.config import BATCH_SIZE, NUM_EPOCHS_STAGE2, LEARNING_RATE_STAGE2
from utils.save_utils import save_plot

def stage2_train(encoder, D1_tensor, pseudo_neg_tensor):
    optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE_STAGE2)

    D1_loader = DataLoader(SessionDataset(D1_tensor), batch_size=BATCH_SIZE, shuffle=True)
    pseudo_neg_loader = DataLoader(SessionDataset(pseudo_neg_tensor), batch_size=BATCH_SIZE, shuffle=True)

    triplet_loss_history = []

    for epoch in range(NUM_EPOCHS_STAGE2):
        encoder.train()
        total_triplet_loss = 0.0

        for batch_pos, batch_pneg in zip(D1_loader, pseudo_neg_loader):
            optimizer.zero_grad()
            min_bs = min(batch_pos.size(0), batch_pneg.size(0))

            x_pos = batch_pos[:min_bs]
            x_pneg = batch_pneg[:min_bs]

            z_pos = encoder(x_pos)
            z_pos2 = encoder(x_pos)  # second forward as anchor
            z_pneg = encoder(x_pneg)

            t_loss = triplet_loss_anchor_positive_negative(z_pos, z_pos2, z_pneg, margin=1.0)
            t_loss.backward()
            optimizer.step()
            total_triplet_loss += t_loss.item()

        triplet_loss_history.append(total_triplet_loss)
        print(f"[Stage 2 | Epoch {epoch+1}/{NUM_EPOCHS_STAGE2}] Triplet Loss = {total_triplet_loss:.4f}")

    return triplet_loss_history


def plot_stage2_loss(loss_history, output_dir):
    fig = plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.title("Stage 2: Pseudo-Negative Push Loss")
    plt.tight_layout()
    save_plot(fig, "stage2_loss.png", output_dir)
