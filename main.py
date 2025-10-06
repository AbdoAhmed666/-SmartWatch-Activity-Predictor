# train_improved.py
import os
import joblib
from src import data_preprocessing, feature_engineering, model_training, evaluation, visualization

DATA_PATH = "data/aw_fb_data.csv"   # change if you want to train on other CSV
MODEL_SAVE_PATH = "models/best_rf_grid.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"
ENCODERS_SAVE_PATH = "models/encoders.pkl"

def main():
    print("📂 Loading data...")
    df = data_preprocessing.load_data(DATA_PATH)

    print("🧹 Cleaning data...")
    df = data_preprocessing.clean_data(df)

    # detect target column automatically
    if "activity" in df.columns:
        target_col = "activity"
    elif "activity_trimmed" in df.columns:
        target_col = "activity_trimmed"
    else:
        raise ValueError("No activity target column found.")

    print("⚙️ Adding features...")
    df = feature_engineering.add_features(df, rolling_window=3)

    print("🔡 Encoding categorical features...")
    # we will encode target as well and keep its encoder (for inverse transform reporting)
    df_encoded, encoders = data_preprocessing.encode_categorical(df, target_col, encode_target=True)

    print("📊 Visualizing (correlation) ...")
    visualization.plot_correlation(df, save_path="models/correlation.png")

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(df_encoded, target_col, test_size=0.2, random_state=42)

    print("📏 Scaling features...")
    X_train_scaled, X_test_scaled, scaler = data_preprocessing.scale_features(X_train, X_test)

    print("🔎 Hyperparameter tuning & training (RandomForest)...")
    # small grid for speed; expand later
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid = model_training.train_random_forest_grid(X_train_scaled, y_train, X_test_scaled, y_test,
                                                  param_grid=param_grid, cv=4, scoring='f1_macro',
                                                  save_path=MODEL_SAVE_PATH, n_jobs=-1, verbose=2)

    print("💾 Saving scaler & encoders...")
    data_preprocessing.save_scaler(scaler, SCALER_SAVE_PATH)
    data_preprocessing.save_encoders(encoders, ENCODERS_SAVE_PATH)

    print("📈 Evaluating best model on test set (and saving artifacts)...")
    # grid is GridSearchCV object - evaluation accepts label encoder to print human labels
    target_le = encoders.get(target_col, None)
    report_df, cm = evaluation.evaluate_model(grid, X_test_scaled, y_test, label_encoder=target_le, save_prefix="models/")

    print("📊 Saving feature importance plot...")
    # get feature names from X_train columns
    feature_names = list(X_train.columns)
    visualization.plot_feature_importance(grid, feature_names, save_path="models/feature_importance.png")

    print("✅ Training pipeline finished. Artifacts saved to models/")

if __name__ == "__main__":
    main()
