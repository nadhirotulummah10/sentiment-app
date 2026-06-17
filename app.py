
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

        st.markdown("### Tingkat Keyakinan")
        st.progress(prob / 100)
        st.write(f"**{prob:.2f}%**")

# Footer
st.markdown("---")

st.caption("""
Sistem Analisis Sentimen Program Makan Siang Gratis pada TikTok


© 2026
""")
