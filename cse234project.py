

# SECTION 1: IMPORT LIBRARIES


from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance



# SECTION 2: LOAD DATASET


file_path = Path(r"C:\Users\Lenovo\Downloads\AirQualityUCI.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {file_path}\n"
        "Check the file location and filename."
    )

df = pd.read_csv(r"C:\Users\Lenovo\Downloads\AirQualityUCI.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

df.columns = df.columns.astype(str).str.strip()

df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")

df = df.loc[
    :, ~df.columns.str.contains(r"^Unnamed", case=False)
]

print("Dataset loaded successfully.")
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

required_columns = ["Date", "Time", "CO(GT)"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )



# SECTION 3: DATA CLEANING


# Create the datetime column before converting sensor data.
df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str).str.strip()
    + " "
    + df["Time"].astype(str).str.strip(),
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)

# Remove the original date and time columns.
df = df.drop(columns=["Date", "Time"])

# IMPORTANT: Do not convert Datetime to numeric.
numeric_columns = [
    col for col in df.columns if col != "Datetime"
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# The dataset uses -200 to indicate missing measurements.
df[numeric_columns] = df[numeric_columns].replace(
    -200, np.nan
)

# Remove invalid dates and sort chronologically.
df = df.dropna(subset=["Datetime"])
df = df.sort_values("Datetime")
df = df.drop_duplicates(subset="Datetime")
df = df.set_index("Datetime")

# Remove columns with no valid readings.
df = df.dropna(axis=1, how="all")

print("\nCleaned dataset shape:", df.shape)
print("\nValid readings:")
print(df[["CO(GT)"]].notna().sum())

if df.empty or df["CO(GT)"].notna().sum() < 10:
    raise ValueError(
        "Not enough valid carbon monoxide readings. "
        "Check the CSV format and missing-value handling."
    )

# Check which expected sensor features are available.
features = [
    "CO(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "Hour",
    "DayOfWeek",
    "Month"
]

missing_features = [
    col for col in features
    if col not in df.columns and col not in [
        "Hour", "DayOfWeek", "Month"
    ]
]

if missing_features:
    print("\nUnavailable sensor features:", missing_features)

features = [
    col for col in features
    if col in df.columns or col in [
        "Hour", "DayOfWeek", "Month"
    ]
]

print("\nFeatures to be used:", features)



# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)

# Missing values chart
df.isnull().mean().mul(100).sort_values().plot(
    kind="bar",
    figsize=(12, 5)
)

plt.title("Missing Values Percentage")
plt.ylabel("Missing Values (%)")
plt.tight_layout()
plt.show()

# Carbon monoxide distribution
sns.histplot(
    data=df,
    x="CO(GT)",
    kde=True
)

plt.title("Distribution of Carbon Monoxide")
plt.xlabel("CO(GT)")
plt.tight_layout()
plt.show()

# Carbon monoxide over time
df["CO(GT)"].plot(figsize=(12, 5))

plt.title("Carbon Monoxide Levels Over Time")
plt.ylabel("CO(GT)")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()



# SECTION 5: STATISTICAL ANALYSIS


# Pearson correlation between temperature and CO
if "T" in df.columns:
    valid = df[["T", "CO(GT)"]].dropna()

    if (
        len(valid) > 2
        and valid["T"].nunique() > 1
        and valid["CO(GT)"].nunique() > 1
    ):
        correlation, p_value = pearsonr(
            valid["T"],
            valid["CO(GT)"]
        )

        print("\nTemperature-CO Pearson correlation:",
              correlation)
        print("P-value:", p_value)
    else:
        print("\nInsufficient variation for Pearson correlation.")
else:
    print("\nTemperature column not available.")

# Compare morning and evening CO levels
df["Hour"] = df.index.hour

morning = df.loc[
    df["Hour"].between(6, 10), "CO(GT)"
].dropna()

evening = df.loc[
    df["Hour"].between(17, 21), "CO(GT)"
].dropna()

if len(morning) > 1 and len(evening) > 1:
    statistic, p_value = ttest_ind(
        morning,
        evening,
        equal_var=False
    )

    print("\nMorning average CO:", morning.mean())
    print("Evening average CO:", evening.mean())
    print("Morning vs evening t-test p-value:", p_value)
else:
    print("\nNot enough readings for the morning/evening t-test.")



# SECTION 6: FEATURE ENGINEERING


# Predict the next hour's CO level.
df["Target"] = df["CO(GT)"].shift(-1)

# Retain only observations followed by a reading
# exactly one hour later.
timestamps = pd.Series(df.index, index=df.index)
next_time = timestamps.shift(-1)

valid_next_hour = (
    (next_time - timestamps)
    == pd.Timedelta(hours=1)
)

df = df.loc[valid_next_hour].copy()

# Extract time-based features.
df["Hour"] = df.index.hour
df["DayOfWeek"] = df.index.dayofweek
df["Month"] = df.index.month

# Remove rows with an unknown target.
df = df.dropna(subset=["Target"])

# Use only columns that exist in the dataset.
features = [col for col in features if col in df.columns]

X = df[features]
y = df["Target"]

if len(df) < 10:
    raise ValueError(
        "Too few valid hourly observations for model training. "
        "Check the timestamps and missing values."
    )

# Chronological 80:20 split (no random shuffling).
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", len(features))



# SECTION 7: MODEL SELECTION AND TRAINING


models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        ))
    ])
}

results = []
trained_models = {}

for name, model in models.items():
    print("\nTraining:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

print("\nMODEL EVALUATION RESULTS")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_performance_results.csv",
    index=False
)


# =====================================================
# SECTION 8: MODEL COMPARISON


results_df.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Model Performance Comparison")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nBest model based on RMSE:", best_name)


#
# SECTION 9: ACTUAL VS PREDICTED VALUES


predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print("\nActual vs Predicted:")
print(comparison.head(10))

comparison.to_csv(
    "actual_vs_predicted.csv",
    index=False
)

plot_count = min(100, len(y_test))

plt.figure(figsize=(10, 5))

plt.plot(
    range(plot_count),
    y_test.iloc[:plot_count].to_numpy(),
    label="Actual"
)

plt.plot(
    range(plot_count),
    predictions[:plot_count],
    label="Predicted"
)

plt.title("Actual vs Predicted CO Levels")
plt.xlabel("Test Observation")
plt.ylabel("CO(GT)")
plt.legend()
plt.tight_layout()
plt.show()


# =====================================================
# SECTION 10: FEATURE IMPORTANCE
# =====================================================

importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=3,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# =====================================================
# SECTION 11: FINAL SUMMARY
# =====================================================

best_result = results_df.iloc[0]

print("\nPROJECT COMPLETED SUCCESSFULLY")
print("Target: Next-hour carbon monoxide level")
print("Best model:", best_name)
print("Best RMSE:", best_result["RMSE"])
print("Best R2 Score:", best_result["R2 Score"])
print("Results saved to CSV files.")
