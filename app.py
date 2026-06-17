
import streamlit as st
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="📊",
    layout="centered"
)

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Load Model
model = joblib.load("model_sentimen.pkl")
tfidf = joblib.load("tfidf.pkl")

# Judul
st.markdown("""
<div class='title'>
Analisis Sentimen
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub'>
Program Makan Siang Gratis pada TikTok<br>
Metode TF-IDF + Logistic Regression
</div>
""", unsafe_allow_html=True)

# Input
st.markdown("<div class='card'>", unsafe_allow_html=True)

teks = st.text_area(
    "Masukkan komentar",
    height=180,
    placeholder="Contoh: Program ini sangat membantu masyarakat..."
)

prediksi = st.button("Analisis Sentimen")

st.markdown("</div>", unsafe_allow_html=True)

# Prediksi
if prediksi:

    if teks.strip() == "":
        st.warning("Masukkan komentar terlebih dahulu.")
    else:

        vector = tfidf.transform([teks])

        hasil = model.predict(vector)[0]
        prob = model.predict_proba(vector).max() * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if hasil == "Positif":
            warna = "#16a34a"

        elif hasil == "Negatif":
            warna = "#dc2626"

        else:
            warna = "#f59e0b"

        st.markdown(f"""
        <div class='result' style="border-left:8px solid {warna};">
            <h2>{hasil}</h2>
            <p><b>Tingkat Keyakinan:</b> {prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob/100)

# Footer
st.markdown("---")

st.caption("""
Sistem Analisis Sentimen Program Makan Siang Gratis pada TikTok

Metode:
- TF-IDF
- Logistic Regression

© 2026
""")
