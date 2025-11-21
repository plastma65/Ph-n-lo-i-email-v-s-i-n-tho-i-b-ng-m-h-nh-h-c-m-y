import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

IN_FILE = Path("data/phone_features.csv")
OUT_DIR = Path("splits_phone")
OUT_DIR.mkdir(exist_ok=True)

print("📥 Loading phone feature dataset...")
df = pd.read_csv(IN_FILE)

print("📊 Tổng số mẫu:", len(df))
print(df["label"].value_counts())

# 70% train, 15% val, 15% test (stratified)
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

print("\n📊 Phân bố nhãn:")
print("Train:", train["label"].value_counts())
print("Val:", val["label"].value_counts())
print("Test:", test["label"].value_counts())

train.to_csv(OUT_DIR / "phone_train.csv", index=False)
val.to_csv(OUT_DIR / "phone_val.csv", index=False)
test.to_csv(OUT_DIR / "phone_test.csv", index=False)

print("\n✅ DONE: Đã chia train/val/test cho phone")
print("📌 Train:", OUT_DIR / "phone_train.csv")
print("📌 Val:", OUT_DIR / "phone_val.csv")
print("📌 Test:", OUT_DIR / "phone_test.csv")
