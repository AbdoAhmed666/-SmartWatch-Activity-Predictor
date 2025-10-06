# 🕒 SmartWatch Activity Predictor

A complete end-to-end **Machine Learning project** that predicts human activity from smartwatch data using a trained **Random Forest model** and an interactive **Streamlit web UI**.  
This project includes **data preprocessing, feature engineering, model training, evaluation, and deployment** — all in one repository.

---

## 🚀 Project Overview

This system takes raw smartwatch sensor data (heart rate, steps, distance, etc.) and predicts the **user’s activity** (e.g., walking, running, sitting).  
It supports both **batch predictions from CSV files** and **single-sample predictions** entered manually through the Streamlit interface.

You can run the web UI locally or expose it online using **ngrok**.

---

## 🧠 Features

- 📊 **Data Preprocessing & Feature Engineering**
  - Clean missing data
  - Generate rolling-window statistics
  - Encode categorical variables
  - Normalize numerical features


- 🤖 **Machine Learning Model**
  - Best model: **Random Forest with GridSearchCV**
  - Saved model artifacts for easy reloading (`pkl` files)
  - Supports re-scaling and encoding automatically

- 🧩 **Streamlit Web App**
  - Upload CSV or input manually
  - Displays predictions with confidence scores
  - Shows classification report, confusion matrix, and feature importance

- 🌍 **Ngrok Integration**
  - Instantly share your Streamlit app with a public URL

---

## 🧾 Project Structure

```
smart_watch/
│
├── data/
│   ├── aw_fb_data.csv
│   ├── data_for_weka_aw.csv
│   └── data_for_weka_fb.csv
│
├── models/
│   ├── best_rf_grid.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── correlation.png
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── eda.py
│
├── notebooks/
│   └── (Jupyter notebooks for EDA and experiments)
│
├── streamlit_app.py      # Streamlit interface
├── main.py               # Entry point or pipeline orchestrator
├── requirements.txt      # Dependencies list
├── .gitignore
└── README.md
```

---

## 🧰 Installation & Setup

### 1️⃣ Create and Activate Environment
```bash
conda create -n smart_watch python=3.10 -y
conda activate smart_watch
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 4️⃣ Public URL via Ngrok (optional)
The app automatically connects to **ngrok** on port `8501` and prints the public URL in your terminal:
```
🌍 Streamlit app running at: https://<your-ngrok-url>.ngrok-free.app
```

---

## 📈 Model Artifacts

| Artifact | Description |
|-----------|--------------|
| `best_rf_grid.pkl` | Trained Random Forest model with tuned hyperparameters |
| `scaler.pkl` | StandardScaler used during training |
| `encoders.pkl` | Encoders for categorical features |
| `classification_report.csv` | Precision, recall, F1 metrics |
| `confusion_matrix.png` | Visual confusion matrix |
| `feature_importance.png` | Top contributing features |
| `correlation.png` | Correlation heatmap of numeric features |


---

## 🎨 Streamlit UI Preview

The UI allows you to:
- Upload your smartwatch data file
- See model predictions and confidence
- View key diagnostic plots and metrics

---

## 📊 Performance Metrics

| Metric | Score |
|---------|--------|
| **Accuracy** | 0.94 |
| **Precision (macro avg)** | 0.93 |
| **Recall (macro avg)** | 0.92 |
| **F1-Score (macro avg)** | 0.92 |

> *Metrics based on the validation set using the best Random Forest model.*

---

## 🧑‍💻 Example Usage

**Upload CSV file:**
1. Click “Browse files” in the Streamlit interface.
2. Select a CSV file with the same feature schema as the training data.
3. Click “Run predictions”.

**Single sample input:**
1. Fill in the input fields (auto-filled from a sample).
2. Click “Predict single sample”.
3. Get predicted activity and confidence instantly.

---

## 🧾 Requirements

Main dependencies:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `streamlit`
- `pyngrok`

Install all via:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Deployment Notes

- Make sure `models/` folder exists and contains trained artifacts.
- CSVs uploaded must match the feature schema from training.
- If you retrain your model, update:
  - `models/best_rf_grid.pkl`
  - `models/scaler.pkl`
  - `models/encoders.pkl`

---

## 🧑‍🔬 Author

**Abdelrhman Ahmed**  
📧 *[abdoibrahim122000@gmail.com]*  
ML Engineer | Data Scientist | IoT & Sensor Data Enthusiast  

---

## 🏁 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it with attribution.

---

> “AI becomes powerful when connected to real-world sensors — this smartwatch project brings ML to your wrist.”
