# scripts/parse_enron.py

import email
from pathlib import Path
import pandas as pd
import re

# Thư mục maildir của ENRON (tính từ thư mục project/)
ENRON_ROOT = Path("../enron_mail_20150507/maildir")

OUT_DIR = Path("data_clean")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "enron_clean.csv"

rows = []

def extract_body(msg):
    """Lấy body text từ email (ưu tiên text/plain)."""
    try:
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition", ""))

                # bỏ attachment, chỉ lấy text/plain
                if ctype == "text/plain" and "attachment" not in cdispo:
                    try:
                        parts.append(
                            part.get_payload(decode=True).decode("utf-8", "ignore")
                        )
                    except Exception:
                        pass
            return "\n".join(parts)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode("utf-8", "ignore")
    except Exception:
        pass
    return ""

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""

if not ENRON_ROOT.exists():
    raise FileNotFoundError(f"Không tìm thấy thư mục ENRON: {ENRON_ROOT}")

print(f"Đang duyệt thư mục ENRON: {ENRON_ROOT}")

count = 0
for path in ENRON_ROOT.rglob("*"):
    if not path.is_file():
        continue
    try:
        # Đọc file dạng text (enron là text thuần)
        with open(path, "r", errors="ignore") as f:
            raw = f.read()
        msg = email.message_from_string(raw)

        email_from = safe_str(msg.get("From", ""))
        subject = safe_str(msg.get("Subject", ""))
        body = safe_str(extract_body(msg))

        # bỏ email rỗng hoàn toàn
        if not subject.strip() and not body.strip():
            continue

        # domain từ from (nếu có)
        domain = ""
        m = re.search(r"@([A-Za-z0-9.\-]+)", email_from)
        if m:
            domain = m.group(1).lower()

        rows.append([email_from, domain, subject, body, 0])  # label = 0 (HAM)
        count += 1
        if count % 10000 == 0:
            print(f"  Đã parse {count} email...")

    except Exception:
        # nếu mail lỗi thì bỏ qua
        continue

print(f"Tổng số email ENRON parse được: {count}")
print("Đang ghi CSV...")

df = pd.DataFrame(
    rows,
    columns=["email_from", "domain", "subject", "body", "label"]
)

# ép kiểu & loại trùng (subject+body)
df["subject"] = df["subject"].astype(str)
df["body"] = df["body"].astype(str)
df = df.drop_duplicates(subset=["subject", "body"], keep="first")

df.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"✅ Đã lưu ENRON sạch tại: {OUT_FILE}")
print("Số dòng sau khi loại trùng:", len(df))
