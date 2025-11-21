# scripts/normalize_email_schema.py
import pandas as pd, csv
from pathlib import Path

# Chấp nhận 2 vị trí phổ biến của file
candidates = [Path('data/dataset_email_cleaned.csv'),
              Path('dataset_email_cleaned.csv')]
in_path = next((p for p in candidates if p.exists()), None)
if in_path is None:
    raise FileNotFoundError("Không tìm thấy dataset_email_cleaned.csv (ở ./ hoặc ./data/).")

out_path = Path('data/dataset_email_cleaned_norm.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

# Đọc robust cho nội dung dài/có dấu phẩy/ngoặc kép
df = pd.read_csv(in_path, encoding='utf-8', engine='python',
                 quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')

# 1) content -> body
if 'body' not in df.columns and 'content' in df.columns:
    df = df.rename(columns={'content': 'body'})

# 2) bảo đảm subject/email_from/domain tồn tại (có thể để rỗng)
for col in ['subject', 'email_from', 'domain']:
    if col not in df.columns:
        df[col] = ''

# 3) chuẩn nhãn
if 'label' not in df.columns:
    # fallback nếu thiếu (không phải trường hợp của bạn)
    if 'type' in df.columns:
        m = {'phishing':1, 'spam':1, 'scam':1, 'ham':0, 'legit':0, 'legitimate':0}
        df['label'] = df['type'].astype(str).str.lower().map(m).fillna(0).astype('int8')
    else:
        raise ValueError("Thiếu cột 'label' và không có cột thay thế ('type').")
else:
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype('int8')

# 4) chỉ giữ đúng thứ tự 5 cột chuẩn
df = df[['email_from','domain','subject','body','label']]

# 5) vệ sinh rỗng/NA
df['subject'] = df['subject'].fillna('')
df['body']    = df['body'].fillna('')

df.to_csv(out_path, index=False, encoding='utf-8')
print("✅ Saved:", out_path, "| Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Label distribution:", df['label'].value_counts().to_dict())
