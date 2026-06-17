
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
Analisis Sentimen Program Makan Siang Gratis pada TikTok
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub'>
Sistem ini menganalisis sentimen masyarakat terhadap Program Makan Bergizi Gratis berdasarkan komentar di TikTok menggunakan metode TF-IDF dan algoritma Logistic Regression.
<br>

</div>
""", unsafe_allow_html=True)

# Input

teks = st.text_area(
    "Masukkan komentar",
    height=150,
    placeholder="Contoh: Program ini sangat membantu masyarakat..."
)

prediksi = st.button("Analisis Sentimen")

st.markdown("</div>", unsafe_allow_html=True)

# Prediksi
# Prediksi
if prediksi:

    if teks.strip() == "":
        st.warning("Masukkan komentar terlebih dahulu.")

    else:

        vector = tfidf.transform([teks])

        hasil = model.predict(vector)[0]
        prob = model.predict_proba(vector).max() * 100

        st.markdown("## 📊 HASIL ANALISIS")

        if hasil == 2:
            st.success("🟢 POSITIF")
            st.write("Komentar menunjukkan dukungan terhadap program.")

        elif hasil == 1:
            st.error("🔴 NEGATIF")
            st.write("Komentar mengandung opini negatif terhadap program.")

        elif hasil == 0:
            st.warning("🟡 NETRAL")
            st.write("Komentar bersifat informatif atau tidak menunjukkan kecenderungan sentimen yang kuat.")

        st.markdown("### Confidence Score")

        st.progress(prob / 70)

        st.markdown(
            f"<p style='font-size:14px;'><b>{prob:.2f}%</b></p>",
            unsafe_allow_html=True
)

# Footer
st.markdown("---")

st.caption("""
Sistem Analisis Sentimen Program Makan Siang Gratis pada TikTok


© 2026
""")
