import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from catboost import CatBoostClassifier
import joblib

# True  -> faster, uses subset of balanced data for iteration (~200,000 samples)
# False -> full balanced data for best accuracy (final run ~2,000,000+ samples)
FAST_DEV_MODE = True

print("Loading merged dataset...")

df = pd.read_csv(
    "merged_accident_weather.csv",
    parse_dates=["Start_Time"],
    low_memory=False,
)

# Define severe accident: Severity_acc >= 3
if "Severity_acc" not in df.columns:
    raise ValueError("Expected 'Severity_acc' column in merged_accident_weather.csv")

df["is_severe"] = (df["Severity_acc"] >= 3).astype(int)

print(f"Dataset size: {len(df)}")
print("Severe accidents:", df["is_severe"].sum())
print("Non-severe:", len(df) - df["is_severe"].sum())

# HANDLE COLUMN NAME VARIANTS
# Precipitation column may be suffixed after merge
if "Precipitation(in)_acc" in df.columns:
    precip_col = "Precipitation(in)_acc"
elif "Precipitation(in)" in df.columns:
    precip_col = "Precipitation(in)"
else:
    raise ValueError("No precipitation column found (expected 'Precipitation(in)_acc' or 'Precipitation(in)').")

# Geographic columns from accidents side
state_col = "State_acc" if "State_acc" in df.columns else "State"
county_col = "County_acc" if "County_acc" in df.columns else "County"
city_col = "City_acc" if "City_acc" in df.columns else "City"

# RECLEANING (TO ENSURE CONSISTENCY)
df["Weather_Event"] = df["Weather_Event"].fillna("NONE")
df["Weather_Condition_Clean"] = df["Weather_Condition_Clean"].fillna("OTHER")

# Ensure key numeric columns are numeric
num_cols_base = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    precip_col,
]

for col in num_cols_base:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill base numeric NaNs with medians
df[num_cols_base] = df[num_cols_base].fillna(df[num_cols_base].median())

# Duration_minutes was created in clean.py, but recompute safely if missing
if "Duration_minutes" not in df.columns:
    df["Duration_minutes"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
df["Duration_minutes"] = df["Duration_minutes"].fillna(df["Duration_minutes"].median())

# ADD / RECOMPUTE TIME-BASED FEATURES
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
df["Month"] = df["Start_Time"].dt.month

df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["Is_Night"] = df["Hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
df["Is_Rush_Hour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
df["Is_Winter"] = df["Month"].isin([12, 1, 2]).astype(int)
df["Is_Summer"] = df["Month"].isin([6, 7, 8]).astype(int)

# HAZARD FLAGS (WEATHER / VISIBILITY / WIND)
df["Low_Visibility"] = (df["Visibility(mi)"] < 3).astype(int)
df["Freezing_Temp"] = (df["Temperature(F)"] < 32).astype(int)
df["High_Wind"] = (df["Wind_Speed(mph)"] > 25).astype(int)

# Interactions using Weather_Event + time features
df["Snow_At_Night"] = (
    (df["Weather_Event"] == "SNOW") & (df["Is_Night"] == 1)
).astype(int)

df["Rain_Rush_Hour"] = (
    (df["Weather_Event"] == "RAIN") & (df["Is_Rush_Hour"] == 1)
).astype(int)

df["Fog_LowVisibility"] = (
    (df["Weather_Event"] == "FOG") & (df["Low_Visibility"] == 1)
).astype(int)

df["Rain_Night"] = (
    (df["Weather_Event"] == "RAIN") & (df["Is_Night"] == 1)
).astype(int)

df["Snow_Rush"] = (
    (df["Weather_Event"] == "SNOW") & (df["Is_Rush_Hour"] == 1)
).astype(int)

# ROAD INFRASTRUCTURE BINARY FEATURES
road_flag_cols = [
    "Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway",
    "Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal","Turning_Loop"
]

for col in road_flag_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().map(
            {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)
    else:
        df[col] = 0

# GEOGRAPHIC FEATURES
if "Start_Lat" in df.columns and "Start_Lng" in df.columns:
    df["Start_Lat"] = pd.to_numeric(df["Start_Lat"], errors="coerce")
    df["Start_Lng"] = pd.to_numeric(df["Start_Lng"], errors="coerce")
else:
    raise ValueError("Expected 'Start_Lat' and 'Start_Lng' in merged_accident_weather.csv")

df["Start_Lat"] = df["Start_Lat"].fillna(df["Start_Lat"].median())
df["Start_Lng"] = df["Start_Lng"].fillna(df["Start_Lng"].median())

for col in [state_col, county_col, city_col]:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in merged_accident_weather.csv")

df[state_col] = df[state_col].fillna("UNK").astype(str)
df[county_col] = df[county_col].fillna("UNK").astype(str)
df[city_col] = df[city_col].fillna("UNK").astype(str)

if "Street" in df.columns:
    df["Street"] = df["Street"].fillna("UNK").astype(str)
else:
    df["Street"] = "UNK"

critical_cols = [
    "Temperature(F)","Humidity(%)","Visibility(mi)",
    "Wind_Speed(mph)", precip_col
]

df = df.dropna(subset=critical_cols + ["Start_Time"])

severe = df[df["is_severe"] == 1]
non_severe = df[df["is_severe"] == 0]

target_non_severe = min(len(non_severe), len(severe) * 2)
non_severe_sample = non_severe.sample(n=target_non_severe, random_state=42)

df_balanced = pd.concat([severe, non_severe_sample])
df_balanced = df_balanced.sample(frac=1.0, random_state=42)

if FAST_DEV_MODE:
    df_balanced = df_balanced.sample(frac=0.10, random_state=42)
    print("FAST_DEV_MODE ON: using 10% of balanced data for fast iteration.")

print(f"Balanced dataset size: {len(df_balanced)}")
print("Balanced severe count:", df_balanced["is_severe"].sum())
print("Balanced non-severe count:", len(df_balanced) - df_balanced["is_severe"].sum())

# FEATURE SETUP
feature_cols = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    precip_col,

    "Hour",
    "DayOfWeek",
    "Month",
    "Is_Rush_Hour",
    "Is_Night",
    "Is_Weekend",
    "Is_Winter",
    "Is_Summer",

    "Low_Visibility",
    "Freezing_Temp",
    "High_Wind",

    "Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway",
    "Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal","Turning_Loop",

    "Start_Lat",
    "Start_Lng",
    state_col,
    county_col,
    city_col,
    "Street",

    "Weather_Event",
    "Weather_Condition_Clean",
]

X = df_balanced[feature_cols]
y = df_balanced["is_severe"]

cat_features = [
    "Weather_Event",
    "Weather_Condition_Clean",
    state_col,
    county_col,
    city_col,
    "Street",
]

print("Categorical features:", cat_features)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
weight_ratio = neg / pos
class_weights = [1.0, weight_ratio]

print(f"Class weight ratio (non-severe:severe) = {weight_ratio:.2f}")

model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.035,
    depth=8,
    l2_leaf_reg=3.0,
    random_strength=1.5,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=class_weights,
    random_seed=42,
    early_stopping_rounds=80,
    verbose=200,
)

print("Training model...")

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
)

print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", acc)
print("ROC-AUC:", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTop Feature Importances:")
fi = model.get_feature_importance(prettified=True)
print(fi.head(25))

fi.to_csv("feature_importance.csv", index=False)
print("Feature importances saved to feature_importance.csv")

joblib.dump(model, "model_catboost_final.pkl")
print("\nModel saved to model_catboost_final.pkl")
