import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.save_utils import save_plot

@torch.no_grad()
def run_cross_validation_evaluation(encoder, D1_tensor, Neg_tensor, output_dir, num_folds=5):
    encoder.eval()

    # Embed D1 and Neg
    z_pos = encoder(D1_tensor).cpu().numpy()
    z_neg = encoder(Neg_tensor).cpu().numpy()

    X = np.vstack([z_pos, z_neg])
    y = np.concatenate([np.ones(len(z_pos)), np.zeros(len(z_neg))])

    classifiers = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),  
        ("NaiveBayes", GaussianNB()),
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]


    summary = []

    for name, clf in classifiers:
        print(f"\n🔍 Evaluating: {name}")
        aucs = []

        for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42).split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            # AUC
            auc = roc_auc_score(y_test, y_proba)
            aucs.append(auc)

            # Classification report
            report = classification_report(y_test, y_pred, digits=4)
            report_file = os.path.join(output_dir, f"report_{name}_fold{fold}.txt")
            with open(report_file, "w") as f:
                f.write(report)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig_cm = plt.figure()
            disp.plot(cmap='Blues')
            plt.title(f"{name} - Fold {fold} Confusion Matrix")
            save_plot(fig_cm, f"confusion_{name}_fold{fold}.png", output_dir)

            # ROC Curve
            fig_roc = plt.figure()
            RocCurveDisplay.from_predictions(y_test, y_proba)
            plt.title(f"{name} - Fold {fold} ROC Curve")
            save_plot(fig_roc, f"roc_{name}_fold{fold}.png", output_dir)

        summary.append((name, np.mean(aucs)))

    # Save overall AUC summary
    summary_df = pd.DataFrame(summary, columns=["Model", "Mean_ROC_AUC"])
    summary_df.to_csv(os.path.join(output_dir, "cv_auc_summary.csv"), index=False)
    print("\n Cross-validation AUC summary:")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))
