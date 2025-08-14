import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

TARGET = "C2-7 Cobb"
FEATURES_SINGLE = ["C2-6 Cobb"]
FEATURES_MULTI = [
    "C2-6 Cobb", "SVA", "CCI", "CCL",
    "C2 Slope", "C3 Slope", "C4 Slope", "C5 Slope",
    "C6 Slope", "Lower C6 Slope", "C7 Slope"
]
BEST_PARAMS_SINGLE = {
    "Linear":        {},
    "Ridge":         {"alpha": 10},
    "Lasso":         {"alpha": 0.01},
    "RandomForest":  {"max_depth": 5, "n_estimators": 100},
    "SVR":           {"C": 10, "epsilon": 0.1},
}
BEST_PARAMS_MULTI = {
    "Linear":        {},
    "Ridge":         {"alpha": 1},
    "Lasso":         {"alpha": 0.01},
    "RandomForest":  {"max_depth": None, "n_estimators": 100},
    "SVR":           {"C": 10, "epsilon": 0.1},
}
EXTERNAL_FILE = "Validation.xlsx"
OUT_DIR = Path("external_eval_outputs")
OUT_DIR.mkdir(exist_ok=True)

df_ext = pd.read_excel(EXTERNAL_FILE)

need_cols = set([TARGET] + FEATURES_SINGLE + FEATURES_MULTI)
missing = [c for c in need_cols if c not in df_ext.columns]
if missing:
    raise ValueError(f"The external table is missing required columns：{missing}")

y_true = df_ext[TARGET].values
X_s = df_ext[FEATURES_SINGLE].copy()
X_m = df_ext[FEATURES_MULTI].copy()

scaler_multi = StandardScaler().fit(X_m)
X_m_scaled = scaler_multi.transform(X_m)

def make_model(name: str, params: dict):
    if name == "Linear":
        return LinearRegression()
    elif name == "Ridge":
        return Ridge(**params)
    elif name == "Lasso":
        return Lasso(**params)
    elif name == "RandomForest":
        return RandomForestRegressor(random_state=42, **params)
    elif name == "SVR":
        return SVR(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

metrics_rows = []
pred_df = pd.DataFrame(index=df_ext.index)

for name, params in BEST_PARAMS_SINGLE.items():
    model = make_model(name, params)
    model.fit(X_s, y_true)
    y_pred = model.predict(X_s)
    pred_df[f"y_pred_{name}_single"] = y_pred

    metrics_rows.append({
        "Branch": "Single",
        "Model": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    })

for name, params in BEST_PARAMS_MULTI.items():
    model = make_model(name, params)
    model.fit(X_m_scaled, y_true)
    y_pred = model.predict(X_m_scaled)
    pred_df[f"y_pred_{name}_multi"] = y_pred

    metrics_rows.append({
        "Branch": "Multi",
        "Model": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_path = OUT_DIR / "external_validation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

out_pred_path = OUT_DIR / f"{Path(EXTERNAL_FILE).stem}_with_predictions.xlsx"
out_df = pd.concat([df_ext, pred_df], axis=1)
out_df.to_excel(out_pred_path, index=False)

print("External validation completed")
print(f"- Metrics summary: {metrics_path}")
print(f"- Prediction results: {out_pred_path}")