import json
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


internal_file = "file1.xlsx"
external_files = [
    "file2.xlsx"
]

target = 'C2-7 Cobb'
features_single = ['C2-6 Cobb']
features_multi = [
    'C2-6 Cobb', 'SVA', 'CCI', 'CCL',
    'C2 Slope', 'C3 Slope', 'C4 Slope', 'C5 Slope',
    'C6 Slope', 'Lower C6 Slope', 'C7 Slope'
]

df = pd.read_excel(internal_file)
X_single = df[features_single]
X_multi = df[features_multi]
y = df[target]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_m_scaled = scaler.fit_transform(X_train_m)
X_test_m_scaled = scaler.transform(X_test_m)
X_multi_scaled = scaler.transform(X_multi)

def build_models():
    return {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'SVR': SVR()
    }

param_grids = {
    'Ridge':        {'alpha': [0.01, 0.1, 1, 10]},
    'Lasso':        {'alpha': [0.01, 0.1, 1, 10]},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'SVR':          {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1]}
}

best_params_single = {}
best_params_multi = {}

for name, model in build_models().items():
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_s, y_train_s)
        best_params_single[name] = grid.best_params_
    else:
        best_params_single[name] = {}

for name, model in build_models().items():
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_m_scaled, y_train_m)
        best_params_multi[name] = grid.best_params_
    else:
        best_params_multi[name] = {}

print("\n=== Best Params (Single) ===")
for k, v in best_params_single.items():
    print(f"{k}: {v}")

print("\n=== Best Params (Multi) ===")
for k, v in best_params_multi.items():
    print(f"{k}: {v}")

out_dir = Path("model_outputs")
out_dir.mkdir(exist_ok=True)

with open(out_dir/"best_params_single.json", "w", encoding="utf-8") as f:
    json.dump(best_params_single, f, ensure_ascii=False, indent=2)
with open(out_dir/"best_params_multi.json", "w", encoding="utf-8") as f:
    json.dump(best_params_multi, f, ensure_ascii=False, indent=2)

pd.DataFrame.from_dict(best_params_single, orient="index").to_csv(out_dir/"best_params_single.csv")
pd.DataFrame.from_dict(best_params_multi,  orient="index").to_csv(out_dir/"best_params_multi.csv")

final_models_single = {}
final_models_multi = {}

scaler_final = StandardScaler().fit(X_multi)
X_multi_scaled_full = scaler_final.transform(X_multi)

for name, base_model in build_models().items():
    params_s = best_params_single.get(name, {})
    model_s = build_models()[name].set_params(**params_s)
    model_s.fit(X_single, y)
    final_models_single[name] = model_s

    params_m = best_params_multi.get(name, {})
    model_m = build_models()[name].set_params(**params_m)
    model_m.fit(X_multi_scaled_full, y)
    final_models_multi[name] = model_m

test_pred_single = {
    name: mdl.predict(X_test_s) for name, mdl in final_models_single.items()
}
X_test_m_scaled_final = scaler_final.transform(X_test_m)
test_pred_multi = {
    name: mdl.predict(X_test_m_scaled_final) for name, mdl in final_models_multi.items()
}

cv_pred_single = {
    name: cross_val_predict(mdl, X_single, y, cv=5) for name, mdl in final_models_single.items()
}
cv_pred_multi = {
    name: cross_val_predict(mdl, X_multi_scaled_full, y, cv=5) for name, mdl in final_models_multi.items()
}

external_pred_files = []
for f in external_files:
    ext = pd.read_excel(f)
    out = ext.copy()

    for name, mdl in final_models_single.items():
        out[f'y_pred_{name}_single'] = mdl.predict(ext[features_single])

    X_ext_m_scaled = scaler_final.transform(ext[features_multi])
    for name, mdl in final_models_multi.items():
        out[f'y_pred_{name}_multi'] = mdl.predict(X_ext_m_scaled)

    out_path = out_dir / f"{Path(f).stem}_with_predictions.xlsx"
    out.to_excel(out_path, index=False)
    external_pred_files.append(str(out_path))

test_pred_df = pd.concat(
    [
        pd.DataFrame({'Branch': 'Single', 'Model': name, 'y_pred': vals})
        for name, vals in test_pred_single.items()
    ] + [
        pd.DataFrame({'Branch': 'Multi',  'Model': name, 'y_pred': vals})
        for name, vals in test_pred_multi.items()
    ],
    ignore_index=True
)
test_pred_df.to_excel(out_dir/"internal_test_predictions_only.xlsx", index=False)

cv_pred_df = pd.concat(
    [
        pd.DataFrame({'Branch': 'Single', 'Model': name, 'y_pred': vals})
        for name, vals in cv_pred_single.items()
    ] + [
        pd.DataFrame({'Branch': 'Multi',  'Model': name, 'y_pred': vals})
        for name, vals in cv_pred_multi.items()
    ],
    ignore_index=True
)
cv_pred_df.to_excel(out_dir/"cv_predictions_only.xlsx", index=False)

print("\nParameters exported：")
print(f"- {out_dir/'best_params_single.json'}")
print(f"- {out_dir/'best_params_multi.json'}")
print(f"- {out_dir/'best_params_single.csv'}")
print(f"- {out_dir/'best_params_multi.csv'}")

print("\nPrediction results saved：")
print(f"- {out_dir/'internal_test_predictions_only.xlsx'}")
print(f"- {out_dir/'cv_predictions_only.xlsx'}")
for p in external_pred_files:
    print(f"- {p}")