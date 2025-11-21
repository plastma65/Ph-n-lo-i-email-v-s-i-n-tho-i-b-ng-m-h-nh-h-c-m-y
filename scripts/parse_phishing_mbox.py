import mailbox
import pandas as pd
from pathlib import Path
import re
import os

# === 1) Tạo thư mục output nếu chưa có ===
OUT_DIR = Path("data_clean")
OUT_DIR.mkdir(exist_ok=True)

# === 2) Đường dẫn chứa file phishing ===
PHISH_DIR = Path("data_raw/phishing")

files = [
    PHISH_DIR / "phishing-2022.txt",
    PHISH_DIR / "phishing-2023.txt",
    PHISH_DIR / "phishing-2024.txt"
]

rows = []

def safe_str(x):
    """Chuyển mọi kiểu dữ liệu thành string an toàn"""
    try:
        return str(x)
    except:
        return ""

def extract_body(msg):
    """Trích body từ email mbox"""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        return part.get_payload(decode=True).decode("utf-8", "ignore")
                    except:
                        return ""
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    return payload.decode("utf-8", "ignore")
                except:
                    return ""
    except:
        return ""
    return ""

# === 3) Parse từng file ===
for file_path in files:
    if not file_path.exists():
        print(f"⚠ FILE KHÔNG TỒN TẠI: {file_path}")
        continue

    print(f"➡ Đang parse file: {file_path.name}")

    try:
        mbox = mailbox.mbox(file_path)
    except Exception as e:
        print("‼ LỖI MỞ FILE:", e)
        continue

    for msg in mbox:
        try:
            email_from = safe_str(msg.get("From", ""))
            raw_subject = msg.get("Subject", "")
            subject = safe_str(raw_subject)
            body = safe_str(extract_body(msg))

            # trích domain
            domain = ""
            match = re.search(r"@([A-Za-z0-9.\-]+)", email_from)
            if match:
                domain = match.group(1).lower()

            rows.append([email_from, domain, subject, body, 1])  # label = 1
        except Exception as e:
            # Nếu có lỗi, bỏ qua email lỗi
            continue

# === 4) Tạo DataFrame & ép kiểu string ===
df = pd.DataFrame(rows, columns=["email_from", "domain", "subject", "body", "label"])

# Ép mọi cột thành chuỗi để tránh lỗi "unhashable"
df["subject"] = df["subject"].astype(str)
df["body"] = df["body"].astype(str)

# Loại trùng lặp
df = df.drop_duplicates(subset=["subject", "body"], keep="first")

# === 5) Lưu file kết quả ===
out_file = OUT_DIR / "phishing_clean.csv"
df.to_csv(out_file, index=False, encoding="utf-8")

print("\n✅ PARSE HOÀN TẤT!")
print("📌 Tổng số email phishing:", len(df))
print("📌 File lưu tại:", out_file)
