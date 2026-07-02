import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Informatics Research Novelty System", layout="wide", page_icon="💻")

st.title("💻 Dasbor Analisis Kebaruan (Novelty) & Kelayakan Riset Informatika")
st.write("Sistem otomatis penguji kelayakan proposal/judul berdasarkan kesenjangan literatur terindeks.")

# 2. DATA MOCK REFERENSI PUBLIKASI TERDAHULU
@st.cache_data
def dapatkan_database():
    return pd.DataFrame([
        {"judul": "Implementasi Algoritma Apriori untuk Analisis Pola Pembelian di Supermarket", "abstrak": "Penelitian ini membahas aturan asosiasi menggunakan algoritma Apriori untuk data penjualan. Hasil menunjukkan pola pembelian konsumen yang dominan.", "metode": "Apriori", "studi_kasus": "Retail / Supermarket", "teknologi": "Python", "sentimen": "Positif"},
        {"judul": "Penerapan Data Mining untuk Prediksi Penyakit Diabetes Menggunakan Naive Bayes", "abstrak": "Sistem prediksi kesehatan dengan Naive Bayes untuk klasifikasi penyakit diabetes berdasarkan riwayat medis pasien. Akurasi mencapai 85%.", "metode": "Naive Bayes", "studi_kasus": "Kesehatan / Diabetes", "teknologi": "R Language", "sentimen": "Positif"},
        {"judul": "Analisis Sentimen Pelanggan Toko Online Menggunakan Metode SVM", "abstrak": "Menganalisis ulasan negatif dan positif dari e-commerce marketplace menggunakan algoritma Support Vector Machine. Menemukan kendala pengiriman.", "metode": "SVM", "studi_kasus": "E-commerce / Marketplace", "teknologi": "Python", "sentimen": "Negatif"},
        {"judul": "Deteksi Objek Real-time Menggunakan Algoritma YOLOv8 pada Kamera Pengawas", "abstrak": "Penggunaan deep learning arsitektur YOLOv8 untuk memonitor kepadatan lalu lintas dan objek kendaraan secara instan pada tangkapan CCTV.", "metode": "YOLOv8", "studi_kasus": "Smart City / CCTV", "teknologi": "PyTorch", "sentimen": "Positif"},
        {"judul": "Sistem Pakar Diagnosa Kerusakan Komputer dengan Metode Forward Chaining", "abstrak": "Aplikasi berbasis web membantu teknisi junior mendiagnosis kerusakan hardware komputer menggunakan penalaran runut maju (forward chaining).", "metode": "Forward Chaining", "studi_kasus": "Hardware / TI Support", "teknologi": "PHP & MySQL", "sentimen": "Netral"}
    ])

db_publikasi = dapatkan_database()

# 3. FUNGSI EKSTRAKSI TEKS DARI DOKUMEN
def baca_docx(file):
    doc = Document(file)
    teks_penuh = [paragraf.text for paragraf in doc.paragraphs]
    return "\n".join(teks_penuh)

def ekstrak_parameter_otomatis(teks):
    """Fungsi ekstraksi heuristik sederhana berbasis kata kunci di bidang Informatika"""
    teks_lower = teks.lower()
    
    # Deteksi Metode
    metode = "Kombinasi / Belum Terdeteksi"
    for m in ["svm", "naive bayes", "apriori", "yolov8", "lstm", "cnn", "forward chaining", "random forest"]:
        if m in teks_lower:
            metode = m.upper()
            break
            
    # Deteksi Teknologi
    tech = "Python / Umum"
    for t in ["python", "r language", "php", "pytorch", "tensorflow", "laravel", "flutter"]:
        if t in teks_lower:
            tech = t.capitalize()
            break
            
    # Deteksi Studi Kasus
    studi = "Sistem Enterprise"
    for s in ["supermarket", "diabetes", "e-commerce", "cctv", "komputer", "sekolah", "aplikasi", "web"]:
        if s in teks_lower:
            studi = s.capitalize()
            break
            
    return metode, studi, tech

# 4. SIDEBAR INPUT (MANUAL DAN UPLOAD FILE)
st.sidebar.header("📥 Metode Input Dokumen")
opsi_input = st.sidebar.radio("Pilih Metode Input:", ("Manual Input (Ketik/Copas)", "Unggah File (CSV / DOCX)"))

judul_input = ""
abstrak_input = ""

if opsi_input == "Manual Input (Ketik/Copas)":
    judul_input = st.sidebar.text_input("Judul Riset yang Diajukan:", "Analisis Sentimen Opini Publik Menggunakan Deep Learning LSTM")
    abstrak_input = st.sidebar.text_area("Abstrak Riset:", "Penelitian ini bertujuan mengklasifikasikan data sentimen masyarakat terkait kebijakan baru menggunakan arsitektur deep learning Long Short-Term Memory (LSTM) berbasis Python TensorFlow.")
else:
    file_diunggah = st.sidebar.file_uploader("Pilih Berkas Berformat .csv atau .docx", type=["csv", "docx"])
    if file_diunggah is not None:
        if file_diunggah.name.endswith(".docx"):
            konten = baca_docx(file_diunggah)
            # Mengasumsikan baris pertama dokumen adalah judul, sisanya abstrak
            baris = [b.strip() for b in konten.split("\n") if b.strip()]
            judul_input = baris[0] if len(baris) > 0 else "Judul dari file Docx"
            abstrak_input = "\n".join(baris[1:]) if len(baris) > 1 else konten
        elif file_diunggah.name.endswith(".csv"):
            df_csv = pd.read_csv(file_diunggah)
            st.sidebar.success("File CSV Berhasil Dimuat!")
            # Ambil baris pertama dari kolom judul dan abstrak pada csv
            kolom = df_csv.columns.tolist()
            judul_input = str(df_csv.iloc[0, 0]) if len(kolom) > 0 else ""
            abstrak_input = str(df_csv.iloc[0, 1]) if len(kolom) > 1 else ""

# Tampilkan review teks yang diekstrak/diinput
st.subheader("📝 Teks Penelitian yang Sedang Dianalisis")
col_j, col_a = st.columns([1, 2])
with col_j:
    st.info(f"**Judul:**\n{judul_input if judul_input else '*Belum diisi*'}")
with col_a:
    st.info(f"**Abstrak / Konten:**\n{abstrak_input[:400] if abstrak_input else '*Belum diisi*'}...")

# Tombol Analisis Utama
mulai_analisis = st.button("🚀 Jalankan Analisis Kebaruan Komprehensif")

# 5. ENGINE ANALISIS & VISUALISASI DASHBOARD
if mulai_analisis and judul_input and abstrak_input:
    # Gabungkan judul + abstrak untuk pemrosesan semantik
    teks_target = f"{judul_input} {abstrak_input}"
    korpus_db = (db_publikasi['judul'] + " " + db_publikasi['abstrak']).tolist()
    
    # Hitung Semantic Similarity (TF-IDF + Cosine Similarity)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(korpus_db + [teks_target])
    skor_kemiripan = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    max_score = np.max(skor_kemiripan) * 100
    idx_paling_mirip = np.argmax(skor_kemiripan)
    literatur_terdekat = db_publikasi.iloc[idx_paling_mirip]
    
    # Ekstraksi Parameter Riset Baru
    metode_u, studi_u, tech_u = ekstrak_parameter_otomatis(teks_target)
    
    st.write("---")
    
    # TAB UTAMA DASHBOARD
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Persentase Kemiripan Teks", 
        "📊 Analisis Research Gap", 
        "☁️ Ekstraksi Keyword & Tren", 
        "⚖️ Sentimen & Dampak Keilmuan"
    ])
    
    # ------------------ TAB 1: SEMANTIC SIMILARITY ------------------
    with tab1:
        st.header("🎯 Skor Kemiripan Semantik (Semantic Similarity)")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric(label="Persentase Kemiripan Tertinggi", value=f"{max_score:.2f}%")
            if max_score < 45:
                st.success("Kesimpulan: KELAYAKAN TINGGI (Ide Riset Sangat Baru/Novel)")
            elif max_score < 75:
                st.warning("Kesimpulan: PERLU PENYESUAIAN (Modifikasi variabel pendukung)")
            else:
                st.error("Kesimpulan: REKOMENDASI DITOLAK (Kemiripan terlalu tinggi / Duplikasi)")
                
        with col_m2:
            st.write(f"**Riset Terdahulu Paling Identik:**")
            st.markdown(f"👉 *\"{literatur_terdekat['judul']}\"*")
            
        # Grafik Distribusi Kemiripan
        db_hasil = db_publikasi.copy()
        db_hasil['Kemiripan (%)'] = skor_kemiripan * 100
        fig_sim = px.bar(db_hasil, x='Kemiripan (%)', y='judul', orientation='h', 
                         title="Komparasi Kemiripan dengan Dokumen Database",
                         color='Kemiripan (%)', color_continuous_scale='Bluered')
        st.plotly_chart(fig_sim, use_container_width=True)

    # ------------------ TAB 2: RESEARCH GAP ------------------
    with tab2:
        st.header("📊 Matriks & Visualisasi Batas Celah (*Research Gap*)")
        
        matriks_gap = pd.DataFrame({
            "Variabel/Parameter": ["Metode / Algoritma Utama", "Objek / Lokasi / Studi Kasus", "Teknologi / Framework Pendukung"],
            "Riset Terdahulu (Paling Mirip)": [literatur_terdekat['metode'], literatur_terdekat['studi_kasus'], literatur_terdekat['teknologi']],
            "Riset Anda (Hasil Ekstraksi)": [metode_u, studi_u, tech_u]
        })
        st.table(matriks_gap)
        
        # Radar Chart / Visual Pola Gap Komparasi sederhana
        kategori = ['Metode', 'Studi Kasus', 'Teknologi']
        # Hitung skor logika biner (1 jika berbeda/baru, 0 jika sama dengan riset terdahulu)
        gap_skor_anda = [
            1 if metode_u.lower() != literatur_terdekat['metode'].lower() else 0,
            1 if studi_u.lower() != literatur_terdekat['studi_kasus'].lower() else 0,
            1 if tech_u.lower() != literatur_terdekat['teknologi'].lower() else 0,
        ]
        
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatterpolar(
              r=gap_skor_anda + [gap_skor_anda[0]],
              theta=kategori + [kategori[0]],
              fill='toself',
              name='Tingkat Kebaruan Komponen'
        ))
        fig_gap.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
                              title="Peta Kekuatan Kebaruan Komponen (Nilai 1 = Benar-Benar Baru)")
        st.plotly_chart(fig_gap, use_container_width=True)

    # ------------------ TAB 3: KEYWORD EXTRACTION & TREND ------------------
    with tab3:
