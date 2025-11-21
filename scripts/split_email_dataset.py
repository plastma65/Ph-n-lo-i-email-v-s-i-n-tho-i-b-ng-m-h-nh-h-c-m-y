# scripts/split_email_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

IN_FILE = Path("data/dataset_email_final_cleaned.csv")
OUT_DIR = Path("splits")
OUT_DIR.mkdir(exist_ok=True)

print("📥 Đang đọc dataset cuối ...")
df = pd.read_csv(IN_FILE, encoding="utf-8")

print("🔍 Tổng số dòng:", len(df))
print(df["label"].value_counts())

# ================================
# 👉 70% TRAIN / 15% VAL / 15% TEST
# ================================

train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=42
)

print("\n📊 Tỷ lệ phân bố nhãn:")
print("Train:", train["label"].value_counts(normalize=True).to_dict())
print("Val  :", val["label"].value_counts(normalize=True).to_dict())
print("Test :", test["label"].value_counts(normalize=True).to_dict())

# ================================
# 👉 LƯU FILE
# ================================
train.to_csv(OUT_DIR / "dataset_train.csv", index=False, encoding="utf-8")
val.to_csv(OUT_DIR / "dataset_val.csv", index=False, encoding="utf-8")
test.to_csv(OUT_DIR / "dataset_test.csv", index=False, encoding="utf-8")

print("\n✅ DONE! Đã chia train/val/test theo tỷ lệ 70/15/15")
print("📌 File train:", OUT_DIR / "dataset_train.csv")
print("📌 File val  :", OUT_DIR / "dataset_val.csv")
print("📌 File test :", OUT_DIR / "dataset_test.csv")
print("🔚 Kết thúc script.")