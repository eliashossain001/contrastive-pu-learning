import torch
from models.encoder import LSTMEncoder
from train.stage1_train import stage1_train, plot_stage1_loss
from train.stage2_train import stage2_train, plot_stage2_loss
from train.inference import (
    compute_centroids,
    mine_pseudo_negatives,
    assign_pseudo_labels,
    extract_embeddings
)
from data.load_data import load_and_split_data, normalize_features
from utils.dataset import SessionDataset
from utils.save_utils import init_output_dir
from config.config import *

from experiments.experiment_1_baseline import run_frozen_encoder_classification
from experiments.experiment_2_cv_roc import run_cross_validation_evaluation
from tsne.tsne_plot import run_tsne, plot_tsne

def main():
    # Initialize output directory
    output_dir = init_output_dir()

    # Step 1: Load and preprocess data
    D1_df, Neg_df, Unl_df, val_df = load_and_split_data()
    D1, Neg, Unl, Val, feature_cols, scaler = normalize_features(D1_df, Neg_df, Unl_df, val_df)

    D1_tensor = torch.tensor(D1, dtype=torch.float32)
    Neg_tensor = torch.tensor(Neg, dtype=torch.float32)
    Unl_tensor = torch.tensor(Unl, dtype=torch.float32)
    Val_tensor = torch.tensor(Val, dtype=torch.float32)

    # Step 2: Build encoder model
    encoder = LSTMEncoder(
        input_dim=D1.shape[1],
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_heads=NUM_HEADS
    )

    # Step 3: Stage 1 training (ConPU)
    print("\n Stage 1 Training (Positives vs Unlabeled)...")
    stage1_loss = stage1_train(encoder, D1_tensor, Unl_tensor)
    plot_stage1_loss(stage1_loss, output_dir)

    # Step 4: Mine pseudo-negatives
    print("\n Mining pseudo-negatives from unlabeled set...")
    pseudo_neg_tensor = mine_pseudo_negatives(encoder, D1_tensor, Unl_tensor)

    # Step 5: Stage 2 training (triplet loss)
    print("\n Stage 2 Training (Triplet push)...")
    stage2_loss = stage2_train(encoder, D1_tensor, pseudo_neg_tensor)
    plot_stage2_loss(stage2_loss, output_dir)

    # Step 6: Final inference
    v1, v0 = compute_centroids(encoder, D1_tensor, Neg_tensor, pseudo_neg_tensor)

    # Step 7: Pseudo-label assignment on unlabeled + t-SNE
    labels_unl, embeddings_unl = assign_pseudo_labels(encoder, Unl_tensor, v1, v0)
    embeddings_unl_2d = run_tsne(embeddings_unl)
    plot_tsne(embeddings_unl_2d, labels_unl, title="t-SNE: Unlabeled Clustering", output_dir=output_dir, filename="tsne_unlabeled.png")

    # Step 8: Validation t-SNE
    labels_val, embeddings_val = assign_pseudo_labels(encoder, Val_tensor, v1, v0)
    embeddings_val_2d = run_tsne(embeddings_val)
    plot_tsne(embeddings_val_2d, labels_val, title="t-SNE: Validation Set Clustering", output_dir=output_dir, filename="tsne_validation.png")

    # Step 9: Experiment 1 — classification on frozen encoder
    run_frozen_encoder_classification(encoder, D1_tensor, Neg_tensor, output_dir)

    # Step 10: Experiment 2 — classification with CV and ROC
    run_cross_validation_evaluation(encoder, D1_tensor, Neg_tensor, output_dir)


if __name__ == "__main__":
    main()
