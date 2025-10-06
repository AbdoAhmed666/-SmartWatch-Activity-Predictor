# src/evaluation.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, label_encoder=None, save_prefix="models/"):
    """
    Print classification report and confusion matrix.
    If label_encoder provided, convert numeric labels back to strings for nicer report.
    """
    y_pred = model.predict(X_test)
    # If model is GridSearchCV object, get best_estimator_
    if hasattr(model, "best_estimator_"):
        estimator = model.best_estimator_
    else:
        estimator = model

    if label_encoder is not None:
        y_test_disp = label_encoder.inverse_transform(y_test.astype(int))
        y_pred_disp = label_encoder.inverse_transform(y_pred.astype(int))
        labels = label_encoder.classes_
    else:
        y_test_disp = y_test
        y_pred_disp = y_pred
        labels = np.unique(np.concatenate([y_test_disp, y_pred_disp]))

    print("✅ Accuracy:", np.mean(y_test_disp == y_pred_disp))
    print("\n📊 Classification Report:\n", classification_report(y_test_disp, y_pred_disp, labels=labels))
    cm = confusion_matrix(y_test_disp, y_pred_disp, labels=labels)
    print("\n🔄 Confusion Matrix:\n", cm)

    os.makedirs(save_prefix, exist_ok=True)
    # Save classification report to csv
    clf_report = classification_report(y_test_disp, y_pred_disp, labels=labels, output_dict=True)
    report_df = pd.DataFrame(clf_report).transpose()
    report_df.to_csv(os.path.join(save_prefix, "classification_report.csv"))

    # Plot confusion matrix heatmap and save
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_prefix, "confusion_matrix.png"))
    plt.close()

    return report_df, cm
