from flask import Flask, render_template, request, session
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'dashboard-upload-only'

UPLOAD_FOLDER = 'data'
PER_PAGE = 10

# HELPER 
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

    # Cleaning tanggal
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    df = df.dropna(subset=['at'])

    df['sentiment_predicted'] = df['sentiment_predicted'].astype(str).str.strip().str.lower()

    # Kolom tambahan waktu
    df['year'] = df['at'].dt.year
    df['month_num'] = df['at'].dt.month
    df['month'] = df['at'].dt.strftime('%b')
    return df

# ROUTE DASHBOARD 
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

        # LIST TAHUN 
        tahun_list = sorted(df['year'].unique())

        df_filtered = df.copy()

        # FILTER TAHUN 
        if tahun != 'all':
            df_filtered = df_filtered[df_filtered['year'] == int(tahun)]

        # LIST BULAN (SESUIAI TAHUN TERPILIH) 
        bulan_data = (
        df_filtered[['year', 'month_num', 'month']]
        .drop_duplicates()
        .sort_values(['year', 'month_num'])
        )
        bulan_list = bulan_data['month'].drop_duplicates().tolist()

        # FILTER BULAN 
        if bulan != 'all':
            df_filtered = df_filtered[df_filtered['month'] == bulan]

        # FILTER SENTIMEN 
        if sentiment != 'all':
            df_filtered = df_filtered[df_filtered['sentiment_predicted'] == sentiment]

        # COUNT 
        total = len(df_filtered)
        pos = len(df_filtered[df_filtered['sentiment_predicted'] == 'positive'])
        neg = len(df_filtered[df_filtered['sentiment_predicted'] == 'negative'])

        # GRAFIK TREN PERBULAN
        df_chart = df.copy()

        # Jika pilih tahun tertentu
        if tahun != 'all':
            df_chart = df_chart[df_chart['year'] == int(tahun)]

        # Group berdasarkan tahun & bulan
        grouped = df_chart.groupby(['year', 'month_num'])

        # Urutkan berdasarkan waktu
        sorted_keys = sorted(grouped.groups.keys())

        for yr, mn in sorted_keys:
            data_bulan = df_chart[
                (df_chart['year'] == yr) &
                (df_chart['month_num'] == mn)
            ]

            pos_count = len(data_bulan[data_bulan['sentiment_predicted'] == 'positive'])
            neg_count = len(data_bulan[data_bulan['sentiment_predicted'] == 'negative'])

            label = pd.to_datetime(f'{yr}-{mn}-01').strftime('%b %Y')

            labels_grafik.append(label)
            tren_pos.append(pos_count)
            tren_neg.append(neg_count)

        # PAGINATION 
        total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
        page = max(1, min(page, total_pages))

        start = (page - 1) * PER_PAGE
        data_page = df_filtered.iloc[start:start + PER_PAGE]

        ulasan = [
            {
                'teks': row['content'],
                'label': row['sentiment_predicted']
            }
            for _, row in data_page.iterrows()
        ]

    return render_template(
        'index.html',
        total=total,
        pos=pos,
        neg=neg,
        ulasan=ulasan,
        labels_grafik=labels_grafik,
        tren_pos=tren_pos,
        tren_neg=tren_neg,
        current_page=page,
        total_pages=total_pages,
        bulan_aktif=bulan,
        bulan_list=bulan_list,
        sentiment_filter=sentiment,
        tahun_list=tahun_list,
        tahun_aktif=tahun
    )

# RUN 
if __name__ == '__main__':
    app.run(debug=True)