import pandas as pd
from pathlib import Path
import re

IN_FILE = Path("data/dataset_email_cleaned.csv")
OUT_FILE = Path("data/dataset_email_final_cleaned.csv")

print("📥 Đang đọc dataset...")

df = pd.read_csv(IN_FILE, encoding="utf-8")

print("🔧 Làm sạch nội dung email...")

# bỏ email body rỗng
df = df[df["body"].str.strip() != ""]

# loại html tag
df["body"] = df["body"].str.replace(r"<[^>]+>", " ", regex=True)

# loại ký tự lạ
df["body"] = df["body"].str.replace(r"[^a-zA-Z0-9\s.,!?@:/\-]", " ", regex=True)

# thu gọn nhiều khoảng trắng
df["body"] = df["body"].str.replace(r"\s+", " ", regex=True).str.strip()

# bỏ email quá ngắn (< 20 ký tự)
df = df[df["body"].str.len() > 20]

# bỏ email quá dài (> 50,000 ký tự)
df = df[df["body"].str.len() < 50000]

print("🧹 Loại trùng lặp...")
df = df.drop_duplicates(subset=["subject", "body"])

print("📦 Đang lưu file cleaned...")
df.to_csv(OUT_FILE, index=False, encoding="utf-8")

print("✅ CLEAN FINAL DONE!")
print("📌 Số dòng còn lại:", len(df))
print("📌 File lưu tại:", OUT_FILE)
