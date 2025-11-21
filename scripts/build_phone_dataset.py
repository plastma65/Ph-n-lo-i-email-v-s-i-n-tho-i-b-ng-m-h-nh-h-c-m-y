import pandas as pd
import numpy as np
from pathlib import Path
import re
import random

RAW_DIR = Path("data_raw/phone")
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

# ========= 1) HÀM TẠO SỐ ĐIỆN THOẠI SYNTHETIC =========

GLOBAL_CC = ["+1", "+33", "+44", "+49", "+61", "+81", "+82", "+84", "+65", "+852"]
VN_PREFIX = ["03", "07", "08", "09"]

def gen_global_phone():
    cc = random.choice(GLOBAL_CC)
    length = random.randint(8, 10)
    num = "".join(str(random.randint(0, 9)) for _ in range(length))
    return cc + num

def gen_vn_phone():
    prefix = random.choice(VN_PREFIX)
    body = "".join(str(random.randint(0, 9)) for _ in range(8))
    return prefix + body

def clean_phone(x: str):
    x = str(x)
    x = re.sub(r"[^0-9+]", "", x)
    if 7 <= len(x) <= 15:
        return x
    return None

# ========= 2) ĐỌC 3 FILE SPAM =========

print("📥 Đang đọc truecaller_spam.csv ...")
tc = pd.read_csv(RAW_DIR / "truecaller_spam.csv")

print("📥 Đang đọc robocall_spam.csv ...")
rb = pd.read_csv(RAW_DIR / "robocall_spam.csv")

print("📥 Đang đọc extra_spam_phones.csv ...")
ex = pd.read_csv(RAW_DIR / "extra_spam_phones.csv")

# Đảm bảo có cột phone, category, label
for df in (tc, rb, ex):
    if "phone" not in df.columns:
        raise ValueError("Thiếu cột 'phone' trong một file spam")
    df["category"] = df.get("category", "spam")
    df["label"] = 1

spam = pd.concat([tc[["phone", "category", "label"]],
                  rb[["phone", "category", "label"]],
                  ex[["phone", "category", "label"]]],
                 ignore_index=True)

spam["phone"] = spam["phone"].apply(clean_phone)
spam = spam.dropna(subset=["phone"])
spam = spam.drop_duplicates(subset=["phone"])

print("🔢 Số lượng phone spam sau khi làm sạch:", len(spam))

# ========= 3) TẠO SỐ HỢP LỆ (HAM) =========

n_spam = len(spam)
target_ham = n_spam   # tạo cùng số lượng để dữ liệu cân bằng

ham_numbers = set()
spam_numbers = set(spam["phone"].tolist())

print("🛠 Đang tạo số hợp lệ (ham) synthetic ...")

while len(ham_numbers) < target_ham:
    if random.random() < 0.5:
        p = gen_global_phone()
    else:
        p = gen_vn_phone()
    p = clean_phone(p)
    if not p:
        continue
    if p in spam_numbers:
        continue
    ham_numbers.add(p)

ham = pd.DataFrame({"phone": list(ham_numbers)})
ham["category"] = "legit"
ham["label"] = 0

print("🔢 Số lượng phone ham tạo ra:", len(ham))

# ========= 4) GỘP & LÀM SẠCH CUỐI =========

df = pd.concat([spam, ham], ignore_index=True)
df = df.dropna(subset=["phone"])
df = df.drop_duplicates(subset=["phone", "label"])

# shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_file = OUT_DIR / "phone_dataset_cleaned.csv"
df.to_csv(out_file, index=False, encoding="utf-8")

print("\n✅ DONE – ĐÃ XÂY DỰNG DATASET PHONE")
print("📌 Tổng số mẫu:", len(df))
print(df["label"].value_counts())
print("📌 Lưu tại:", out_file)
