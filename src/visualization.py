# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_distribution(df: pd.DataFrame, column: str, save_path=None):
    plt.figure(figsize=(8,4))
    # if categorical: countplot, else hist
    if df[column].dtype == object or df[column].dtype.name == 'category':
        sns.countplot(y=column, data=df, order=df[column].value_counts().index)
    else:
        sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_correlation(df: pd.DataFrame, save_path=None):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if numeric_df.shape[1] == 0:
        print("⚠️ No numeric columns found for correlation plot.")
        return
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features Only)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path="models/feature_importance.png"):
    # If GridSearchCV, extract best_estimator_
    if hasattr(model, "best_estimator_"):
        estimator = model.best_estimator_
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        print("⚠️ Model has no feature_importances_ attribute.")
        return

    fi = estimator.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
    fi_df = fi_df.sort_values("importance", ascending=False).head(30)

    plt.figure(figsize=(8,6))
    sns.barplot(x="importance", y="feature", data=fi_df)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
