"""
=========================================================
evaluate.py
Evaluate Best Global Model
=========================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src import config
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from src.dataset import (
    load_dataset,
    split_dataset,
    create_tf_dataset,
)

def evaluate_global_model():

    print("=" * 60)
    print("Loading Best Global Model")
    print("=" * 60)

    # ---------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------

    df, encoder = load_dataset()

    train_df, valid_df, test_df = split_dataset(df)

    test_dataset = create_tf_dataset(
        test_df,
        shuffle=False,
    )

    # ---------------------------------------------------
    # Load Model
    # ---------------------------------------------------

    model_path = config.MODELS_PATH / "best_global_model.keras"

    model = tf.keras.models.load_model(model_path)

    print()

    print("Model Loaded Successfully")

    print(model_path)

    # ---------------------------------------------------
    # Evaluate
    # ---------------------------------------------------

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )
    # ---------------------------------------------------
# Predictions
# ---------------------------------------------------

    print("\nGenerating Predictions...")

    y_true = test_df["label"].values

    predictions = model.predict(
    test_dataset,
    verbose=1,
  )

    y_pred = np.argmax(predictions, axis=1)

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

    cm = confusion_matrix(
    y_true,
    y_pred,
  )

    labels = encoder.classes_

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False,
    )

    plt.title("Confusion Matrix")

    os.makedirs(
    config.RESULTS_PATH,
    exist_ok=True,
    )

    save_path = config.RESULTS_PATH / "confusion_matrix.png"

    plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    )

    plt.show()
    # ---------------------------------------------------
    # Classification Report
    # ---------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_,
    digits=4,
    )

    print(report)
    # ---------------------------------------------------
# Precision-Recall Curve (One-vs-Rest)
# ---------------------------------------------------

 # ---------------------------------------------------
# Precision-Recall Curve (One-vs-Rest)
# ---------------------------------------------------

    print("\nGenerating Precision-Recall Curve...")

    y_true_bin = label_binarize(
        y_true,
        classes=range(len(encoder.classes_))
    )

    plt.figure(figsize=(8,8))

    ap_scores = {}

    for i, class_name in enumerate(encoder.classes_):

        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i],
            predictions[:, i]
        )

        ap = average_precision_score(
            y_true_bin[:, i],
            predictions[:, i]
        )

        ap_scores[class_name] = ap

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{class_name} (AP = {ap:.3f})"
        )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)

    plt.title(
        "Precision-Recall Curve (One-vs-Rest)",
        fontsize=15,
        fontweight="bold"
    )

    plt.grid(True)
    plt.legend(loc="lower left")

    pr_path = config.RESULTS_PATH / "precision_recall_curve.png"

    plt.savefig(
        pr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nPrecision-Recall Curve Saved")
    print(pr_path)
    # ---------------------------------------------------
# Average Precision Scores
# ---------------------------------------------------

    print("\n" + "=" * 60)
    print("AVERAGE PRECISION (AP)")
    print("=" * 60)

    for class_name, ap in ap_scores.items():
        print(f"{class_name:10s}: {ap:.4f}")

    # Compute Macro AP
    macro_ap = np.mean(list(ap_scores.values()))

    # Compute Weighted AP
    weighted_ap = average_precision_score(
        y_true_bin,
        predictions,
        average="weighted"
    )

    print(f"\nMacro Average Precision    : {macro_ap:.4f}")
    print(f"Weighted Average Precision : {weighted_ap:.4f}")

    # Save AP scores
    ap_df = pd.DataFrame({
        "Class": list(ap_scores.keys()) + [
            "Macro Average",
            "Weighted Average"
        ],
        "Average Precision": list(ap_scores.values()) + [
            macro_ap,
            weighted_ap
        ]
    })

    ap_path = config.RESULTS_PATH / "average_precision.csv"

    ap_df.to_csv(ap_path, index=False)

    print("\nAverage Precision Saved")
    print(ap_path)
# ---------------------------------------------------
# ---------------------------------------------------
# ROC Curve (One-vs-Rest)
# ---------------------------------------------------

    print("\nGenerating ROC Curve...")

# One-hot encode true labels
    y_true_bin = label_binarize(
    y_true,
    classes=range(len(encoder.classes_))
    )

    plt.figure(figsize=(8,8))

    for i, class_name in enumerate(encoder.classes_):

        fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        predictions[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_name} (AUC = {roc_auc:.3f})"
    )

# Random classifier reference
    plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=2,
    color="gray",
)

    plt.xlabel("False Positive Rate", fontsize=12)

    plt.ylabel("True Positive Rate", fontsize=12)

    plt.title(
    "ROC Curve (One-vs-Rest)",
    fontsize=15,
    fontweight="bold"
)

    plt.legend(loc="lower right")

    plt.grid(True)

    roc_path = config.RESULTS_PATH / "roc_curve.png"

    plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

    plt.show()

    print("\nROC Curve Saved")

    print(roc_path)
# ---------------------------------------------------
# Save Classification Report
# ---------------------------------------------------

    report_dict = classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_,
    output_dict=True,
)

    report_df = pd.DataFrame(report_dict).transpose()

    report_path = config.RESULTS_PATH / "classification_report.csv"

    report_df.to_csv(report_path)

    print("\nClassification Report Saved")

    print(report_path)
    # ---------------------------------------------------
    # Save Overall Metrics
    # ---------------------------------------------------

    metrics_df = pd.DataFrame({
    "Metric": [
        "Test Accuracy",
        "Test Loss"
    ],
    "Value": [
        accuracy,
        loss
    ]
    })

    metrics_path = config.RESULTS_PATH / "metrics.csv"

    metrics_df.to_csv(
    metrics_path,
    index=False,
)

    print("\nMetrics Saved")

    print(metrics_path)
    print("\nConfusion Matrix Saved")

    print(save_path)

    print()

    print("=" * 60)

    print("TEST RESULTS")

    print("=" * 60)

    print(f"Test Loss     : {loss:.4f}")

    print(f"Test Accuracy : {accuracy*100:.2f}%")

    return model, test_dataset, encoder


if __name__ == "__main__":

      evaluate_global_model()