from google_play_scraper import reviews, Sort
import pandas as pd
import os

# ID aplikasi KAI Access
app_id = 'com.kai.kaiaccess'

# Ambil ulasan
result, _ = reviews(
    app_id,
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=200
)

# Konversi ke DataFrame
df = pd.DataFrame(result)

# Ambil kolom penting
df = df[['content', 'score', 'at']]

# Contoh label sentimen sederhana
df['sentiment_predicted'] = df['score'].apply(
    lambda x: 'positive' if x >= 4 else 'negative'
)

# Simpan ke folder data
os.makedirs('data', exist_ok=True)
df.to_csv('data/ulasan_kai.csv', index=False)

print("Ulasan berhasil diperbarui!")