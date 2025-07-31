# ConPU-Learning: Contrastive Positive-Unlabeled Learning with LSTM + Attention

This repository implements a **Contrastive PU Learning** pipeline with:
- Stage 1: Training on positives vs. unlabeled using a KL-contrastive loss
- Stage 2: Pushing pseudo-negatives using triplet loss
- Final evaluation using frozen encoder embeddings on standard classifiers

---

## Architecture
- LSTM encoder with Multi-Head Self-Attention
- KL-based contrastive loss for ConPU learning
- Triplet loss for refining pseudo-negatives
- t-SNE visualization
- Classifier benchmarking (LogReg, RF, SVM, etc.)

---

## Project Structure

```
conpu_pu_learning/
├── config/                 # Hyperparameters and constants
├── data/                  # Data loader + normalization
├── models/                # Encoder + Losses
├── train/                 # Stage 1/2 training + inference
├── utils/                 # Dataset loader and metrics
├── tsne/                  # Visualization for t-SNE
├── experiments/           # Downstream classifier evaluations
├── main.py                # Full training and evaluation pipeline
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <this_repo>
cd conpu_pu_learning
pip install -r requirements.txt
```

Make sure your dataset file is:
```
Augmented_Positive_Unlabeled_Data.csv
```
in the root directory.

---

## Run the Full Pipeline

```bash
python main.py
```

---

## 📊 Output

- Stage 1 and Stage 2 loss plots
- Mined pseudo-negatives
- Final pseudo-labels
- t-SNE plots
- Classifier performance reports (Accuracy, Precision, Recall, F1, ROC-AUC)

---

## Citation

If you use this code, consider citing this repository or the relevant research idea behind ConPU learning.