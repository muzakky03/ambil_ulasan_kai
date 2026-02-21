from google_play_scraper import reviews, Sort
import pandas as pd
import os

# ================== KONFIGURASI ==================
APP_PACKAGE = 'com.kai.kaiaccess'
JUMLAH_ULASAN = 3000

# ================== AMBIL ULASAN ==================
result, _ = reviews(
    APP_PACKAGE,
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=JUMLAH_ULASAN
)

df = pd.DataFrame(result)

# ================== VALIDASI ==================
if df.empty:
    print("Data ulasan kosong, proses dihentikan")
    exit()

# Pastikan kolom ada
kolom_wajib = ['content', 'score', 'at']
for k in kolom_wajib:
    if k not in df.columns:
        print(f"Kolom {k} tidak ditemukan")
        print("Kolom tersedia:", df.columns)
        exit()

# ================== AMBIL KOLOM SESUAI DASHBOARD ==================
df = df[['content', 'score', 'at']]

# ================== KONVERSI SENTIMEN ==================
# Karena app.py butuh: sentiment_predicted
def label_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score <= 2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_predicted'] = df['score'].apply(label_sentiment)

# Hapus neutral (karena dashboard kamu hanya hitung pos & neg)
df = df[df['sentiment_predicted'] != 'neutral']

# ================== SIMPAN CSV ==================
os.makedirs('data', exist_ok=True)

file_path = 'data/ulasan_kai.csv'
df.to_csv(file_path, index=False)

print("CSV berhasil dibuat:", file_path)
print("Jumlah data:", len(df))
