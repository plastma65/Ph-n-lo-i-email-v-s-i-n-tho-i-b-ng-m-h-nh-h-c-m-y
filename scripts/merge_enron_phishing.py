import pandas as pd
from pathlib import Path

DATA_CLEAN = Path("data_clean")
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

enron_path = DATA_CLEAN / "enron_clean.csv"
phishing_path = DATA_CLEAN / "phishing_clean.csv"

if not enron_path.exists():
    raise FileNotFoundError("Không tìm thấy enron_clean.csv")

if not phishing_path.exists():
    raise FileNotFoundError("Không tìm thấy phishing_clean.csv")

print("📥 Đang đọc enron_clean.csv ...")
enron = pd.read_csv(enron_path, encoding="utf-8")

print("📥 Đang đọc phishing_clean.csv ...")
phishing = pd.read_csv(phishing_path, encoding="utf-8")

print("🔗 Đang merge 2 dataset ...")
df = pd.concat([enron, phishing], ignore_index=True)

# đảm bảo đúng schema
df["email_from"] = df["email_from"].astype(str)
df["domain"] = df["domain"].astype(str)
df["subject"] = df["subject"].astype(str)
df["body"] = df["body"].astype(str)
df["label"] = df["label"].astype(int)

# loại email rỗng
df = df[df["body"].str.strip() != ""]

# loại trùng (HAM trùng PHISHING)
df = df.drop_duplicates(subset=["subject", "body"])

# xáo trộn dữ liệu
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_file = OUT_DIR / "dataset_email_cleaned.csv"
df.to_csv(out_file, index=False, encoding="utf-8")

print("✅ MERGE HOÀN TẤT!")
print("📌 Số dòng cuối cùng:", len(df))
print("📌 Dataset được lưu tại:", out_file)
