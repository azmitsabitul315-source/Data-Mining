import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. Pembuatan Data Sintetis Non-Linier
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# 2. Pemisahan Data Latih & Uji Standar (train_test_split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fitting Model Baseline & Evaluasi Error
lr_baseline = LinearRegression()
lr_baseline.fit(X_train, y_train)
y_pred_base = lr_baseline.predict(X_test)

print("=== MODUL 1: REGRESI LINIER BASELINE ===")
print(f"MSE Baseline : {mean_squared_error(y_test, y_pred_base):.4f}")
print(f"R2 Score     : {r2_score(y_test, y_pred_base):.4f}\n")



print("=== MODUL 2.1: FEATURE SCALING (StandardScaler vs MinMaxScaler) ===")
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler()
}

errors = {}
scaled_predictions = {}

for scaler_label, scaler in scalers.items():
    # Fit transform pada data training, transform pada data test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training Linear Regression pada data berskala
    lr_scaled = LinearRegression()
    lr_scaled.fit(X_train_scaled, y_train)
    
    # Prediksi & hitung MSE
    y_pred_scale = lr_scaled.predict(X_test_scaled)
    key = f"{scaler_label}_scaling"
    errors[key] = mean_squared_error(y_test, y_pred_scale)
    scaled_predictions[scaler_label] = y_pred_scale

for key, error_val in errors.items():
    print(f"{key} MSE: {error_val:.4f}")
print()

# Plot Actual vs Predicted 
sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_palette('deep')

plt.figure(figsize=(7, 5))
plt.scatter(y_test, scaled_predictions['standard'], alpha=0.7, color='blue', label='StandardScaler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Perfect Fit')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.title('Actual vs Predicted Values (Modul 2)')
plt.legend()
plt.show()

print("=== MODUL 2.2: PEMISAHAN DATA TINGKAT LANJUT (ShuffleSplit) ===")
ss = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
for fold, (tr_idx, te_idx) in enumerate(ss.split(X)):
    print(f"Iterasi ShuffleSplit {fold+1}: Data Latih = {len(tr_idx)}, Data Uji = {len(te_idx)}")
print()

print("=== MODUL 2.3: REGRESI POLINOMIAL & KOMPLEKSITAS MODEL ===")
degrees = [1, 3, 10]
plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_tr_poly = poly.fit_transform(X_train)
    X_te_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_tr_poly, y_train)

    X_grid = np.linspace(0, 5, 200).reshape(-1, 1)
    y_grid = model.predict(poly.transform(X_grid))

    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Data Latih')
    plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Data Uji')
    plt.plot(X_grid, y_grid, color='red', linewidth=2, label=f'Derajat {degree}')
    plt.title(f'Polinomial Derajat {degree}')
    plt.xlabel('Fitur X')
    plt.ylabel('Target y')
    plt.legend()

plt.tight_layout()
plt.show()



print("=== MODUL 3: EVALUASI K-FOLD CROSS-VALIDATION ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
poly_best = PolynomialFeatures(degree=3, include_bias=False)
X_poly_all = poly_best.fit_transform(X)

scores = []
lr_kfold = LinearRegression()

for fold, (train_index, test_index) in enumerate(kf.split(X_poly_all)):
    X_tr, X_te = X_poly_all[train_index], X_poly_all[test_index]
    y_tr, y_te = y[train_index], y[test_index]

    lr_kfold.fit(X_tr, y_tr)
    pred = lr_kfold.predict(X_te)
    score = r2_score(y_te, pred)
    scores.append(score)
    print(f"Fold {fold+1} R2 Score: {score:.4f}")

print(f"Rata-rata R2 Cross-Validation: {np.mean(scores):.4f}")