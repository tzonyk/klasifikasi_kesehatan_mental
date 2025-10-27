import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Klasifikasi Kesehatan Mental",
    page_icon="🧠"
)

model = joblib.load("model_kesehatan_mental.joblib")

st.title("🧠 Klasifikasi Kesehatan Mental Siswa")
st.markdown("Aplikasi machine learning untuk memprediksi *potensi burnout* pada siswa berdasarkan data akademik dan kebiasaan mereka.")

jurusan = st.selectbox("Jurusan", ["Teknik Otomotif", "Teknik Mesin", "Akuntansi", "Tata Boga", "Multimedia","Perkantoran"])
usia = st.slider("Usia", 13, 20, 16)
jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
pendapatan = st.selectbox("Pendapatan Keluarga", ["Rendah", "Menengah", "Tinggi"])
lokasi = st.selectbox("Lokasi Sekolah", ["Urban", "Suburban", "Rural"])
jam_hp = st.slider("Jumlah Jam HP Harian", 1, 12, 5)

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame(
        [[jurusan, usia, jenis_kelamin, pendapatan, lokasi, jam_hp]],
        columns=["Jurusan", "Usia", "Jenis Kelamin", "Pendapatan Keluarga", "Lokasi Sekolah", "Jumlah Jam HP Harian"]
    )
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Potensi burnout kamu diprediksi *{prediksi}* dengan tingkat keyakinan *{presentase*100:.2f}%* 🧩")
    st.balloons()

st.divider()
st.caption("Dibuat dengan ❤ oleh *Toni Kurniawan*")
