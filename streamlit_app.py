# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src import feature_engineering, data_preprocessing, visualization
from pyngrok import ngrok

# Connect ngrok
NGROK_AUTH = "31MmotBq9kQNwjbyRo1pIv9FoH9_6y7FXW49bjG4VPSdQHzG1"
ngrok.set_auth_token(NGROK_AUTH)
public_url = ngrok.connect(8501)  # Streamlit default port
print("🌍 Streamlit app running at:", public_url)

# --------- Helpers ---------
@st.cache_resource
def load_artifacts(model_path="models/best_rf_grid.pkl",
                   scaler_path="models/scaler.pkl",
                   encoders_path="models/encoders.pkl",
                   sample_data_path="data/aw_fb_data.csv"):
    """Load model (GridSearchCV or estimator), scaler, encoders and example schema."""
    artifacts = {}
    # model
    if os.path.exists(model_path):
        model_obj = joblib.load(model_path)
        artifacts["model_obj"] = model_obj
        artifacts["estimator"] = getattr(model_obj, "best_estimator_", model_obj)
    else:
        artifacts["model_obj"] = None
        artifacts["estimator"] = None

    # scaler
    artifacts["scaler"] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # encoders
    artifacts["encoders"] = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}

    # sample schema
    if os.path.exists(sample_data_path):
        df_sample = pd.read_csv(sample_data_path)
        # remove dropped cols that were used earlier
        for c in ["Unnamed: 0", "X1"]:
            if c in df_sample.columns:
                df_sample = df_sample.drop(columns=[c])
        artifacts["sample_df"] = df_sample
        artifacts["feature_columns"] = [c for c in df_sample.columns if c not in ("activity","activity_trimmed")]
    else:
        artifacts["sample_df"] = None
        artifacts["feature_columns"] = None

    return artifacts

def preprocess_for_prediction(df_input: pd.DataFrame, encoders: dict, scaler):
    """Apply feature engineering, encoders and scaler to dataframe and return numpy array ready for model."""
    # 1) add engineered features same as training
    df_fe = feature_engineering.add_features(df_input.copy(), rolling_window=3)

    # 2) apply encoders (for categorical columns)
    df_enc = data_preprocessing.apply_encoders(df_fe, encoders) if encoders else df_fe.copy()

    # 3) ensure numeric only (fill missing cols with 0 if any)
    # Convert columns to the order used in training if possible. We'll keep columns present in df_enc.
    cols = df_enc.columns.tolist()
    # convert all to numeric where possible
    for c in cols:
        # try convert object columns that remained (maybe encoded to -1)
        try:
            df_enc[c] = pd.to_numeric(df_enc[c])
        except Exception:
            # leave as is (if cannot convert)
            df_enc[c] = df_enc[c]

    # Replace any NaN/inf
    df_enc = df_enc.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4) scale numeric array (scaler expects numeric numpy array)
    if scaler is not None:
        X_arr = np.array(df_enc, dtype=float)
        X_scaled = scaler.transform(X_arr)
    else:
        X_scaled = np.array(df_enc, dtype=float)

    return X_scaled, df_enc

def predict_and_attach(df_input: pd.DataFrame, estimator, encoders, scaler, target_le=None):
    """Return dataframe with predictions and probabilities (if available)."""
    X_scaled, df_enc = preprocess_for_prediction(df_input, encoders, scaler)
    # choose estimator (if grid object provided, use best_estimator_)
    model = estimator
    if model is None:
        raise RuntimeError("No trained model loaded.")
    # predict
    preds = model.predict(X_scaled)
    # probabilities if supported
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
    # decode preds if label encoder for target exists
    if target_le is not None:
        try:
            preds_disp = target_le.inverse_transform(preds.astype(int))
        except Exception:
            preds_disp = preds
    else:
        preds_disp = preds

    res_df = df_input.copy().reset_index(drop=True)
    res_df["predicted_label"] = preds_disp
    if probs is not None:
        # add top probability and optionally column-wise per class
        top_proba = probs.max(axis=1)
        res_df["pred_confidence"] = top_proba
    return res_df, probs

# --------- UI ---------
st.set_page_config(layout="wide", page_title="SmartWatch Activity Predictor", initial_sidebar_state="expanded")
st.title("🕒 Smart-Watch Activity Predictor — Streamlit UI")
st.markdown("Upload your CSV or type a single sample to get activity predictions using the trained model.")

# Load artifacts
with st.spinner("Loading model & artifacts..."):
    artifacts = load_artifacts()
model_obj = artifacts["model_obj"]
estimator = artifacts["estimator"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
sample_df = artifacts["sample_df"]
feature_columns_default = artifacts["feature_columns"]
target_encoder = encoders.get("activity") or encoders.get("activity_trimmed")

# Sidebar options
st.sidebar.header("Settings")
st.sidebar.write("Model & artifacts paths are loaded from `models/` by default.")
if model_obj is None:
    st.sidebar.error("No trained model found at models/best_rf_grid.pkl. Run training first.")
else:
    st.sidebar.success("Trained model loaded.")

# Show basic model info
st.sidebar.subheader("Model info")
if model_obj is not None:
    st.sidebar.write(type(model_obj))
    if hasattr(model_obj, "best_params_"):
        st.sidebar.write("Best params:", model_obj.best_params_)

# Main columns
col1, col2 = st.columns([2,1])

with col1:
    st.header("1) Predict from file (CSV)")
    uploaded_file = st.file_uploader("Upload CSV (must match schema)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Uploaded preview:")
            st.dataframe(df_upload.head())
            # Preprocess columns: drop unnamed columns if present
            for c in ["Unnamed: 0","X1"]:
                if c in df_upload.columns:
                    df_upload = df_upload.drop(columns=[c])
            # If user uploaded only features without target, ok. If contains target, drop it for prediction.
            if "activity" in df_upload.columns:
                df_input = df_upload.drop(columns=["activity"])
            elif "activity_trimmed" in df_upload.columns:
                df_input = df_upload.drop(columns=["activity_trimmed"])
            else:
                df_input = df_upload

            if st.button("Run predictions on uploaded CSV"):
                if estimator is None:
                    st.error("Model not loaded.")
                else:
                    res_df, probs = predict_and_attach(df_input, estimator, encoders, scaler, target_encoder)
                    st.success("✅ Predictions done")
                    st.dataframe(res_df.head(50))
                    # Downloadable CSV
                    csv = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
                    # If prob present, show top 20 with low confidence
                    if probs is not None:
                        low_conf = res_df.sort_values("pred_confidence").head(20)
                        st.markdown("### Lowest confidence predictions (top 20)")
                        st.dataframe(low_conf[["predicted_label","pred_confidence"]].head(20))
        except Exception as e:
            st.error(f"Failed to read/process CSV: {e}")

    st.markdown("---")
    st.header("2) Predict single sample (manual input)")
    st.write("Fill the fields below (we auto-populate from a sample row).")

    # Build a manual input form using sample_df columns if exists
    sample_row = None
    if sample_df is not None:
        sample_row = sample_df.iloc[0]
    else:
        sample_row = pd.Series({})

    # Collect inputs dynamically based on feature columns if available
    form = st.form("single_input_form")
    inputs = {}
    if feature_columns_default:
        for col in feature_columns_default:
            default_val = sample_row.get(col, 0)
            if pd.api.types.is_numeric_dtype(type(default_val)) or isinstance(default_val, (int, float, np.number)):
                inputs[col] = form.number_input(label=col, value=float(default_val))
            else:
                inputs[col] = form.text_input(label=col, value=str(default_val))
    else:
        # fallback: minimal common features
        inputs["hear_rate"] = form.number_input("hear_rate", value=70.0)
        inputs["steps"] = form.number_input("steps", value=10.0)
        inputs["distance"] = form.number_input("distance", value=0.01)

    submit_single = form.form_submit_button("Predict single sample")
    if submit_single:
        # build df
        df_single = pd.DataFrame([inputs])
        try:
            res_single, probs_single = predict_and_attach(df_single, estimator, encoders, scaler, target_encoder)
            st.success(f"Predicted: {res_single['predicted_label'].iloc[0]}")
            if 'pred_confidence' in res_single.columns:
                st.write("Confidence:", float(res_single['pred_confidence'].iloc[0]))
            st.write(res_single.T)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col2:
    st.header("Model Artifacts & Diagnostics")
    if os.path.exists("models/classification_report.csv"):
        st.markdown("### Classification report")
        try:
            cr = pd.read_csv("models/classification_report.csv", index_col=0)
            st.dataframe(cr)
        except Exception:
            st.write("classification_report.csv exists but couldn't be read.")
    if os.path.exists("models/confusion_matrix.png"):
        st.markdown("### Confusion matrix")
        st.image("models/confusion_matrix.png", use_column_width=True)
    if os.path.exists("models/feature_importance.png"):
        st.markdown("### Feature importance (top features)")
        st.image("models/feature_importance.png", use_column_width=True)
    if os.path.exists("models/correlation.png"):
        st.markdown("### Correlation (numeric features)")
        st.image("models/correlation.png", use_column_width=True)

    st.markdown("---")
    st.header("Quick actions")
    if st.button("Reload artifacts"):
        st.experimental_rerun()

    st.markdown("Download saved model & artifacts (for deployment):")
    if os.path.exists("models/best_rf_grid.pkl"):
        with open("models/best_rf_grid.pkl","rb") as f:
            st.download_button("Download GridSearch model", f, "best_rf_grid.pkl")
    if os.path.exists("models/scaler.pkl"):
        with open("models/scaler.pkl","rb") as f:
            st.download_button("Download scaler.pkl", f, "scaler.pkl")
    if os.path.exists("models/encoders.pkl"):
        with open("models/encoders.pkl","rb") as f:
            st.download_button("Download encoders.pkl", f, "encoders.pkl")

st.markdown("---")
st.caption("Built by you — ML pipeline trained on wearable heart-rate & activity data. Make sure uploaded CSVs match the feature schema used during training (see sample row preview above).")
