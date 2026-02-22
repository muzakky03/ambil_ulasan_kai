from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'dashboard-upload-only'

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
PER_PAGE = 10


# ================= HELPER =================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_latest_csv():
    if not os.path.exists(UPLOAD_FOLDER):
        return None
    files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    return max(files, key=os.path.getmtime) if files else None


def load_data():
    csv_path = session.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        csv_path = get_latest_csv()

    if not csv_path:
        return None

    df = pd.read_csv(csv_path)

    required_cols = {'content', 'sentiment_predicted', 'at'}
    if not required_cols.issubset(df.columns):
        return None

    # Clean
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    df = df.dropna(subset=['at'])
    df['sentiment_predicted'] = df['sentiment_predicted'].astype(str).str.strip().str.lower()

    # Tambahan kolom
    df['month'] = df['at'].dt.strftime('%b')
    df['month_num'] = df['at'].dt.month
    df['year'] = df['at'].dt.year

    return df


# ================= ROUTE DASHBOARD =================
@app.route('/')
def dashboard():
    bulan = request.args.get('bulan', 'all')
    tahun = request.args.get('tahun', 'all')
    sentiment = request.args.get('sentiment', 'all')
    page = request.args.get('page', 1, type=int)

    df = load_data()

    # Default kosong
    total = pos = neg = 0
    ulasan = []
    labels_grafik, tren_pos, tren_neg = [], [], []
    bulan_list, tahun_list = [], []
    total_pages = 1

    if df is not None:

        tahun_list = sorted(df['year'].unique())
        df_filtered = df.copy()

        # Filter Tahun
        if tahun != 'all':
            df_filtered = df_filtered[df_filtered['year'] == int(tahun)]

        # List bulan
        bulan_data = df_filtered[['month', 'month_num']].drop_duplicates().sort_values('month_num')
        bulan_list = bulan_data['month'].tolist()

        # Filter bulan
        if bulan != 'all':
            df_filtered = df_filtered[df_filtered['month'] == bulan]

        # Filter sentimen
        if sentiment != 'all':
            df_filtered = df_filtered[df_filtered['sentiment_predicted'] == sentiment]

        # Count
        total = len(df_filtered)
        pos = len(df_filtered[df_filtered['sentiment_predicted'] == 'positive'])
        neg = len(df_filtered[df_filtered['sentiment_predicted'] == 'negative'])

        # Grafik (per tahun saja)
        df_chart = df if tahun == 'all' else df[df['year'] == int(tahun)]
        chart = df_chart[['month', 'month_num']].drop_duplicates().sort_values('month_num')
        labels_grafik = chart['month'].tolist()

        for b in labels_grafik:
            d = df_chart[df_chart['month'] == b]
            tren_pos.append(len(d[d['sentiment_predicted'] == 'positive']))
            tren_neg.append(len(d[d['sentiment_predicted'] == 'negative']))

        # Pagination
        total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
        page = max(1, min(page, total_pages))
        start = (page - 1) * PER_PAGE
        data_page = df_filtered.iloc[start:start + PER_PAGE]

        ulasan = [{'teks': r['content'], 'label': r['sentiment_predicted']} for _, r in data_page.iterrows()]

    return render_template(
        'index.html',
        total=total, pos=pos, neg=neg,
        ulasan=ulasan,
        labels_grafik=labels_grafik,
        tren_pos=tren_pos, tren_neg=tren_neg,
        current_page=page, total_pages=total_pages,
        bulan_aktif=bulan, bulan_list=bulan_list,
        sentiment_filter=sentiment,
        tahun_list=tahun_list, tahun_aktif=tahun
    )


# ================= UPLOAD =================
@app.route('/upload', methods=['POST'])
def upload_csv():
    file = request.files.get('file')
    if not file or file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('dashboard'))

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    session['csv_path'] = path
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)