import pandas as pd
import numpy as np
from pathlib import Path
import math
import re

IN_FILE = Path("data/phone_dataset_cleaned.csv")
OUT_FILE = Path("data/phone_features.csv")

print("📥 Loading dataset...")
df = pd.read_csv(IN_FILE)

# --- FIX: ÉP TOÀN BỘ CỘT PHONE VỀ STRING ---
df["phone"] = df["phone"].astype(str)

# ======== HÀM TÍNH ĐẶC TRƯNG ========

def has_country_code(phone):
    phone = str(phone)
    return 1 if phone.startswith("+") else 0

def extract_country_code(phone):
    phone = str(phone)
    if phone.startswith("+"):
        match = re.match(r"\+(\d{1,3})", phone)
        if match:
            return int(match.group(1))
    return 0

def digit_entropy(phone):
    digits = [d for d in str(phone) if d.isdigit()]
    if len(digits) == 0:
        return 0
    counts = {}
    for d in digits:
        counts[d] = counts.get(d, 0) + 1
    total = len(digits)
    entropy = 0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy

def repeat_ratio(phone):
    digits = [d for d in str(phone) if d.isdigit()]
    if len(digits) < 2:
        return 0
    repeats = sum(digits[i] == digits[i+1] for i in range(len(digits)-1))
    return repeats / (len(digits)-1)

def prefix(phone):
    digits = re.sub(r"\D","", str(phone))
    if len(digits) >= 3:
        return int(digits[:3])
    return int(digits) if digits else 0

# ======== TẠO ĐẶC TRƯNG ========

print("🔧 Engineering features...")

df["length"] = df["phone"].apply(lambda x: len(re.sub(r'[^0-9]', '', str(x))))

df["has_country_code"] = df["phone"].apply(has_country_code)

df["country_code"] = df["phone"].apply(extract_country_code)

df["digit_entropy"] = df["phone"].apply(digit_entropy)

df["repeat_ratio"] = df["phone"].apply(repeat_ratio)

df["prefix"] = df["phone"].apply(prefix)

df = df.dropna()

print("💾 Saving feature dataset...")
df.to_csv(OUT_FILE, index=False, encoding="utf-8")

print("✅ DONE – Tính đặc trưng số điện thoại")
print("📌 Lưu tại:", OUT_FILE)
print("📌 Số dòng:", len(df))
