import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========= FILE INPUT / OUTPUT =========
TRAIN_FILE = Path("splits_phone/phone_train.csv")
VAL_FILE = Path("splits_phone/phone_val.csv")
TEST_FILE = Path("splits_phone/phone_test.csv")

ARTIFACT_DIR = Path("artifacts/phone")
ARTIFACT_DIR.mkdir(exist_ok=True)

print("📥 Loading train/val/test datasets...")
train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

# ========= CHỌN FEATURES =========
FEATURES = ["length", "has_country_code", "country_code",
            "digit_entropy", "repeat_ratio", "prefix"]

X_train = train[FEATURES]
y_train = train["label"]

X_val = val[FEATURES]
y_val = val["label"]

X_test = test[FEATURES]
y_test = test["label"]

# ========= KẾT HỢP TRAIN + VAL CHO CROSS-VAL =========
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = pd.concat([y_train, y_val], axis=0)

# ========= THỬ 2 MÔ HÌNH =========
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier()
}

param_grids = {
    "Logistic Regression": {
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['liblinear']
    },
    "Random Forest": {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [5, 10, None]
    }
}

best_model = None
best_score = -1
best_name = None

print("\n🔍 TRAINING MODELS...\n")

for model_name in models:
    print(f"➡ Training: {model_name}")

    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', models[model_name])
    ])

    grid = GridSearchCV(pipe, param_grids[model_name],
                        cv=5, scoring='f1', n_jobs=-1)

    grid.fit(X_train_full, y_train_full)

    print(f"   ✓ Best params: {grid.best_params_}")
    print(f"   ✓ Best CV F1-score: {grid.best_score_:.4f}\n")

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = model_name

print("\n🔥 Mô hình tốt nhất dựa trên VAL F1:", best_name)

# ========= TESTING =========
print("\n🔎 TESTING BEST MODEL...\n")
y_pred = best_model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Lưu bảng kết quả
result_file = ARTIFACT_DIR / "phone_test_results.csv"
report_df.to_csv(result_file)
print("📄 Test metrics saved to:", result_file)

# ========= CONFUSION MATRIX =========
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham","Spam"],
            yticklabels=["Ham","Spam"])
plt.title(f"Confusion Matrix – {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_file = ARTIFACT_DIR / "phone_confusion_matrix.png"
plt.savefig(cm_file, dpi=300)
print("🖼 Confusion matrix saved to:", cm_file)

# ========= SAVE MODEL =========
model_file = ARTIFACT_DIR / "phone_best_model.joblib"
joblib.dump(best_model, model_file)
print("💾 Model saved to:", model_file)

print("\n🎉 TRAINING DONE!")
print("✨ Best model:", best_name)
