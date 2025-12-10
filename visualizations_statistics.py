import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    precision_recall_curve
)
import joblib

print("Loading merged dataset...")
df = pd.read_csv("merged_accident_weather.csv", low_memory=False, parse_dates=["Start_Time"])

print("Recomputing engineered features (must match model.py)")

df["is_severe"] = (df["Severity_acc"] >= 3).astype(int)

df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
df["Month"] = df["Start_Time"].dt.month

df["Is_Rush_Hour"] = df["Hour"].isin([7,8,9,16,17,18]).astype(int)
df["Is_Night"] = ((df["Hour"] <= 6) | (df["Hour"] >= 21)).astype(int)
df["Is_Weekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
df["Is_Winter"] = df["Month"].isin([12,1,2]).astype(int)
df["Is_Summer"] = df["Month"].isin([6,7,8]).astype(int)

df["Low_Visibility"] = (df["Visibility(mi)"] < 2).astype(int)
df["Freezing_Temp"] = (df["Temperature(F)"] <= 32).astype(int)
df["High_Wind"] = (df["Wind_Speed(mph)"] > 20).astype(int)

df["Rain_At_Night"] = ((df["Weather_Event"] == "RAIN") & (df["Is_Night"] == 1)).astype(int)
df["Snow_At_Night"] = ((df["Weather_Event"] == "SNOW") & (df["Is_Night"] == 1)).astype(int)
df["Rain_Rush_Hour"] = ((df["Weather_Event"] == "RAIN") & (df["Is_Rush_Hour"] == 1)).astype(int)
df["Fog_LowVisibility"] = ((df["Weather_Event"] == "FOG") & (df["Low_Visibility"] == 1)).astype(int)
df["Rain_Night"] = ((df["Weather_Event"] == "RAIN") & (df["Is_Night"] == 1)).astype(int)
df["Snow_Rush"] = ((df["Weather_Event"] == "SNOW") & (df["Is_Rush_Hour"] == 1)).astype(int)

print("Loading trained model...")
model = joblib.load("model_catboost_final.pkl")

feature_cols = model.feature_names_

categorical_cols = [
    "Weather_Event",
    "Weather_Condition_Clean",
    "State_acc",
    "County_acc",
    "City_acc",
    "Street"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

# Extract model input matrix
X = df[feature_cols]
y = df["is_severe"]

print("Generating predictions...")
probs = model.predict_proba(X)[:, 1]
preds = model.predict(X)

# roc curve
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

# confusion matrix
cm = confusion_matrix(y, preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y, probs)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.savefig("precision_recall.png")
plt.close()

# Feature importance
print("Generating feature importance plot...")

importances = model.get_feature_importance(prettified=True)

plt.figure(figsize=(10, 14))
sns.barplot(
    data=importances.sort_values("Importances", ascending=False).head(20),
    y="Feature Id",
    x="Importances",
    palette="viridis"
)
plt.title("CatBoost Feature Importance (Top 20)", fontsize=16)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("All visualizations saved:")
print("   - roc_curve.png")
print("   - confusion_matrix.png")
print("   - precision_recall.png")
print("   - feature_importance.png")
