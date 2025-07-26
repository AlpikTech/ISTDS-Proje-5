import streamlit as st
import datetime
from pymongo import MongoClient
import json
from streamlit_lottie import st_lottie

with open("assets/feedback.json", "r") as dosya:
    animasyon = json.load(dosya)

# 🔐 MongoDB bağlantısı
# Buraya kendi bağlantı linkini koy
client = MongoClient(
    "mongodb+srv://AlpikTech:ijQkFS1b6drq8eHY@istdsproje5.eb23zqp.mongodb.net/")  # MongoDB URI'ını buraya yaz
db = client["feedback"]
collection = db["userFeedBack"]

st.set_page_config(page_title="Geri Bildirim", layout="centered")
with st.form("Geri Bildirim Sayfası"):
    st.write("Görüşlerin bizim çok için önemli! Lütfen aşağıdaki formu doldur.")

    # 🧾 Form alanları
    isim = st.text_input("Adınız:")
    st.markdown("Sitemizi kaç puanla değerlendirirsiniz?")
    puan = st.feedback("stars")
    mesaj = st.text_area("Geri Bildiriminiz")

    if st.form_submit_button("Gönder"):
        if mesaj.strip() == "":
            st.warning("Lütfen boş mesaj göndermeyin.")
        else:
            yeni_geri_bildirim = {
                "date": datetime.datetime.now(),
                "username": isim if isim else "Anonim",
                "point": puan+1,
                "feedback": mesaj
            }

            try:
                collection.insert_one(yeni_geri_bildirim)
                st.toast('Geri bildiriminiz başarıyla alındı.')
                st.balloons()
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
st_lottie(animasyon, speed=1, loop=True, height=300)

