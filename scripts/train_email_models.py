import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt


# ==== 1) Đường dẫn ====

SPLIT_DIR = Path("splits")
OUT_DIR = Path("artifacts/email")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(SPLIT_DIR / "dataset_train.csv")
val = pd.read_csv(SPLIT_DIR / "dataset_val.csv")
test = pd.read_csv(SPLIT_DIR / "dataset_test.csv")


# ==== 2) Ghép subject + body thành text ====

for df in (train, val, test):
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

X_train = train["text"]
y_train = train["label"]

X_val = val["text"]
y_val = val["label"]

X_test = test["text"]
y_test = test["label"]


# ==== 3) TF-IDF ====

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.97,
    max_features=200_000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, OUT_DIR / "tfidf_vectorizer.joblib")


# ==== 4) Hàm tính metrics ====

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ==== 5) Logistic Regression ====

lr = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000)
lr_params = {"C": [0.1, 1, 10]}

lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring="f1", n_jobs=-1)
lr_grid.fit(X_train_tfidf, y_train)

best_lr = lr_grid.best_estimator_
y_val_lr = best_lr.predict(X_val_tfidf)
lr_val_metrics = get_metrics(y_val, y_val_lr)


# ==== 6) Random Forest ====

rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1)
rf_params = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20],
}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train_tfidf, y_train)

best_rf = rf_grid.best_estimator_
y_val_rf = best_rf.predict(X_val_tfidf)
rf_val_metrics = get_metrics(y_val, y_val_rf)


# ==== 7) Chọn mô hình tốt nhất ====

if lr_val_metrics["f1"] >= rf_val_metrics["f1"]:
    best_model = best_lr
    best_name = "Logistic Regression"
else:
    best_model = best_rf
    best_name = "Random Forest"

print(f"\n🔥 Mô hình tốt nhất dựa trên F1 (VAL): {best_name}")

# ==== 8) Đánh giá trên TEST ====

y_test_pred = best_model.predict(X_test_tfidf)
test_metrics = get_metrics(y_test, y_test_pred)


# ==== 9) Lưu kết quả ====

pd.DataFrame([test_metrics]).to_csv(
    OUT_DIR / "email_test_results.csv", index=False
)

joblib.dump(best_model, OUT_DIR / "email_best_model.joblib")


# ==== 10) Vẽ confusion matrix ====

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(OUT_DIR / "email_confusion_matrix.png", dpi=200)
plt.close()


print("\n🎉 TRAINING DONE!")
print("📌 Best model:", best_name)
print("📌 Test metrics saved to:", OUT_DIR / "email_test_results.csv")
print("📌 Confusion matrix saved to:", OUT_DIR / "email_confusion_matrix.png")
print("📌 Model saved to:", OUT_DIR / "email_best_model.joblib")
