import pandas as pd
import re
import joblib
import os

from sklearn.metrics import accuracy_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stopword = StopWordRemoverFactory().create_stop_word_remover()

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
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = stopword.remove(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['content'] = df['content'].apply(clean_text)

# label_asli dari score 
def label_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score <= 2:
        return 'negative'
    else:
        return 'neutral'

df['label_asli'] = df['score'].apply(label_sentiment)

# Load Model
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# Transform
X = vectorizer.transform(df['content'])

# Prediksi
df['sentiment_predicted'] = model.predict(X)

# Hitung Akurasi (label_asli & hasil model)
accuracy = accuracy_score(df['label_asli'], df['sentiment_predicted'])
print(f"Akurasi Model: {accuracy}")

# Simpan kembali (overwrite)
df.to_csv(DATA_PATH, index=False)

print("Preprocessing & Prediksi selesai")
print("Jumlah data:", len(df))