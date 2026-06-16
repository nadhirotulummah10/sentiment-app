import streamlit as st
import joblib

model = joblib.load("model_sentimen.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Analisis Sentimen")

text = st.text_area("Masukkan teks")

if st.button("Analisis"):
    if text.strip() == "":
        st.warning("Teks kosong")
    else:
        x = tfidf.transform([text])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]

        if pred == 0:
            label = "Negatif"
        elif pred == 1:
            label = "Netral"
        else:
            label = "Positif"

        st.success(label)

        st.write("Probabilitas:")
        st.write(f"Negatif: {proba[0]*100:.2f}%")
        st.write(f"Netral: {proba[1]*100:.2f}%")
        st.write(f"Positif: {proba[2]*100:.2f}%")