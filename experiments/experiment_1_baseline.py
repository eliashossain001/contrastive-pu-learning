import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os 

@torch.no_grad()
def run_frozen_encoder_classification(encoder, D1_tensor, Neg_tensor, output_dir, scaler=None, D1_raw=None, Neg_raw=None):
    encoder.eval()

    # Get embeddings
    z_pos = encoder(D1_tensor).cpu().numpy()
    z_neg = encoder(Neg_tensor).cpu().numpy()

    # Labels
    y_pos = np.ones(len(z_pos), dtype=int)
    y_neg = np.zeros(len(z_neg), dtype=int)

    # Combine
    X_all = np.vstack([z_pos, z_neg])
    y_all = np.concatenate([y_pos, y_neg])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Naive Bayes", GaussianNB()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]

    results = []
    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append((name, acc, prec, rec, f1))

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    print("\n📊 Classifier performance on frozen encoder embeddings:")
    print(results_df.to_string(index=False, float_format='{:.5f}'.format))
    results_df.to_csv(os.path.join(output_dir, "classifier_report.csv"), index=False)
    return results_df
