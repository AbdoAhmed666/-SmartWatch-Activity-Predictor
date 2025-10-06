# src/model_training.py
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

def train_random_forest_grid(X_train, y_train, X_test=None, y_test=None,
                             param_grid=None, cv=5, scoring='f1_macro',
                             save_path="models/best_rf.pkl", n_jobs=-1, verbose=2):
    """Run GridSearchCV on RandomForest, return best model and cv results."""
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    print("🔎 Running GridSearchCV on RandomForest...")
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("✅ Grid search done. Best params:", grid.best_params_)
    if X_test is not None and y_test is not None:
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy of best model: {acc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(grid, save_path)  # save GridSearchCV object (contains best_estimator_)
    return grid

def save_model_object(obj, path="models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model_object(path="models/model.pkl"):
    return joblib.load(path)

def quick_compare_models(X, y, cv=5):
    """
    Quick cross-val compare on a few models. Returns dict of mean cv scores.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC()
    }
    results = {}
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
        results[name] = np.mean(scores)
        print(f"{name} mean f1_macro (cv={cv}): {results[name]:.4f}")
    return results
