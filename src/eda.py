import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to CSV files
aw_fb_data_path = "../data/aw_fb_data.csv"
weka_aw_path = "../data/data_for_weka_aw.csv"
weka_fb_path = "../data/data_for_weka_fb.csv"


# Load datasets
aw_fb_data = pd.read_csv(aw_fb_data_path)
weka_aw = pd.read_csv(weka_aw_path)
weka_fb = pd.read_csv(weka_fb_path)

# =============================
# Basic Info
# =============================
print("📌 Dataset Shapes:")
print("aw_fb_data:", aw_fb_data.shape)
print("data_for_weka_aw:", weka_aw.shape)
print("data_for_weka_fb:", weka_fb.shape)

print("\n📌 Columns in aw_fb_data:")
print(aw_fb_data.columns.tolist())

print("\n📌 Columns in data_for_weka_aw:")
print(weka_aw.columns.tolist())

print("\n📌 Columns in data_for_weka_fb:")
print(weka_fb.columns.tolist())

# =============================
# Missing Values
# =============================
print("\n📌 Missing Values per Dataset:")
print("aw_fb_data:\n", aw_fb_data.isnull().sum())
print("\ndata_for_weka_aw:\n", weka_aw.isnull().sum())
print("\ndata_for_weka_fb:\n", weka_fb.isnull().sum())

# Head samples
# =============================
print("\n📌 Head of aw_fb_data:")
print(aw_fb_data.head())

print("\n📌 Head of data_for_weka_aw:")
print(weka_aw.head())

print("\n📌 Head of data_for_weka_fb:")
print(weka_fb.head())

# Correlation Heatmap
# =============================
plt.figure(figsize=(10,6))
sns.heatmap(aw_fb_data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - aw_fb_data")
plt.show()