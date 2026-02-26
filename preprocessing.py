import pandas as pd
import re
import joblib
import os

DATA_PATH = "data/ulasan_kai.csv"

# Cek file ada
if not os.path.exists(DATA_PATH):
    print("File ulasan tidak ditemukan")
    exit()

# Load Data
df = pd.read_csv(DATA_PATH)

# Ambil kolom penting 
df = df[['content', 'score', 'at']]

# Cleaning Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['content'] = df['content'].apply(clean_text)

# Load Model
model = joblib.load("model_svm.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Transform
X = vectorizer.transform(df['content'])

# Prediksi
df['sentiment_predicted'] = model.predict(X)

# Simpan kembali (overwrite)
df.to_csv(DATA_PATH, index=False)

print("Preprocessing & Prediksi selesai")
print("Jumlah data:", len(df))