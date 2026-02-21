from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'dashboard-upload-only'

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
PER_PAGE = 10


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_latest_csv():
    if not os.path.exists(UPLOAD_FOLDER):
        return None

    files = [
        os.path.join(UPLOAD_FOLDER, f)
        for f in os.listdir(UPLOAD_FOLDER)
        if f.endswith('.csv')
    ]
    return max(files, key=os.path.getmtime) if files else None


@app.route('/')
def dashboard():

    # ================= PARAMETER =================
    bulan_aktif = request.args.get('bulan') or 'all'
    sentiment_filter = request.args.get('sentiment') or 'all'
    tahun_aktif = request.args.get('tahun') or 'all'
    page = request.args.get('page', 1, type=int)

    # ================= DEFAULT =================
    total = pos = neg = 0
    ulasan = []
    labels_grafik, tren_pos, tren_neg = [], [], []
    bulan_list = []
    tahun_list = []
    total_pages = 1

    # ================= CSV AKTIF =================
    csv_path = session.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        csv_path = get_latest_csv()

    if csv_path and os.path.exists(csv_path):

        df = pd.read_csv(csv_path)

        required_cols = {'content', 'sentiment_predicted', 'at'}
        if required_cols.issubset(df.columns):

            # ============== CLEAN DATA ==============
            df['at'] = pd.to_datetime(df['at'], errors='coerce')
            df = df.dropna(subset=['at'])

            df['sentiment_predicted'] = (
                df['sentiment_predicted']
                .astype(str)
                .str.strip()
                .str.lower()
            )

            df['month'] = df['at'].dt.strftime('%b')
            df['month_num'] = df['at'].dt.month
            df['year'] = df['at'].dt.year

            # ============== LIST TAHUN ==============
            tahun_list = sorted(df['year'].unique())

            # ============== FILTER DATA ==============
            df_filtered = df.copy()

            # Filter Tahun
            if tahun_aktif != 'all':
                try:
                    df_filtered = df_filtered[
                        df_filtered['year'] == int(tahun_aktif)
                    ]
                except ValueError:
                    tahun_aktif = 'all'

            # ============== LIST BULAN (TERBARU DULU) ==============
            bulan_data = (
                 df_filtered[['year', 'month', 'month_num']]
                .drop_duplicates()
                .sort_values(['year', 'month_num'])
            )

            bulan_list = bulan_data['month'].tolist()

            # Filter Bulan
            if bulan_aktif != 'all':
                df_filtered = df_filtered[
                    df_filtered['month'] == bulan_aktif
                ]

            # Filter Sentiment
            if sentiment_filter != 'all':
                df_filtered = df_filtered[
                    df_filtered['sentiment_predicted'] == sentiment_filter
                ]

            # ============== COUNT ==============
            total = len(df_filtered)
            pos = len(df_filtered[df_filtered['sentiment_predicted'] == 'positive'])
            neg = len(df_filtered[df_filtered['sentiment_predicted'] == 'negative'])

            # ============== GRAFIK (BERDASARKAN FILTER TAHUN SAJA) ==============
            df_chart = df.copy()

            if tahun_aktif != 'all':
                df_chart = df_chart[df_chart['year'] == int(tahun_aktif)]

            chart_data = (
                df_chart[['year', 'month', 'month_num']]
                .drop_duplicates()
                .sort_values(['year', 'month_num'])
            )

            labels_grafik = chart_data['month'].tolist()

            for b in labels_grafik:
                d = df_chart[df_chart['month'] == b]

                tren_pos.append(
                    len(d[d['sentiment_predicted'] == 'positive'])
                )
                tren_neg.append(
                    len(d[d['sentiment_predicted'] == 'negative'])
                )

            # ============== PAGINATION ==============
            total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

            if page < 1:
                page = 1
            if page > total_pages:
                page = total_pages

            start = (page - 1) * PER_PAGE
            data_page = df_filtered.iloc[start:start + PER_PAGE]

            ulasan = [
                {'teks': r['content'], 'label': r['sentiment_predicted']}
                for _, r in data_page.iterrows()
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
        bulan_aktif=bulan_aktif,
        bulan_list=bulan_list,
        sentiment_filter=sentiment_filter,
        tahun_list=tahun_list,
        tahun_aktif=tahun_aktif
    )


@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return redirect(url_for('dashboard'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('dashboard'))

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    session['csv_path'] = path
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)