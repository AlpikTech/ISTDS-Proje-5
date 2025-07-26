import streamlit as st

st.set_page_config(page_title="Sıkça Sorulan Sorular", page_icon="❓")

st.title("❓ Sıkça Sorulan Sorular (SSS)")


import json
from streamlit_lottie import st_lottie

with open("assets/Questions.json", "r") as dosya:
    animasyon = json.load(dosya)
st_lottie(animasyon, speed=1, loop=False, height=300)

with st.expander("🧠 Bu uygulama nasıl çalışıyor?"):
    st.write("""
    Uygulama, NLP, TF-IDF, Logistic Regression gibi tekniklerle çalışıyor. Daha ayrıntılı bilgi için **Naıl Çalışır?** sayfasına göz atabilirsiniz.
    """)

with st.expander("📊 Yüzde (%) kaç güvenilirlik ile çalışıyor?"):
    st.write("""
    Model, testlerden %96 lık skor almıştır. Daha ayrıntılı bilgi için ana sayfaya göz atabilirsiniz.
    """)

with st.expander("💬 Hangi dilleri destekliyor?"):
    st.write("""
    Kullanılan verisetlerinden dolayı yalnızca **İngilizce yorumlar** desteklenmektedir. İngilizce dışında yapılan yorumlar tutarsız sonuç verecektir.
    """)

with st.expander("📝 Model hangi verilerle eğitildi?"):
    st.write("""
    Model, çeşitli e-ticaret sitelerinden toplanan **gerçek kullanıcı yorumları** ve **üretilmiş sahte yorumlar** ile eğitildi. 
    Eğitim verisi, yorumun tonunu, uzunluğunu, yapısını ve kelime çeşitliliğini analiz eder.
    """)

with st.expander("🔒 Verilerim kaydediliyor mu?"):
    st.write("""
    Hayır. Girdiğiniz yorumlar "MongoDB ye Kaydet"e tıklanmadığı sürece **hiçbir şekilde kaydedilmez** ve tamamen gizli tutulur.
    Uygulama sadece anlık olarak tahmin işlemi yapar.
    """)

with st.expander("📦 API ya da dış sistemle entegrasyon mümkün mü?"):
    st.write("""
    Şu an için sadece web arayüzü kullanılabilir. Anca githubtan modelimizin pickle dosyasını kullanabilirsiniz.
    """)

with st.expander("📱 Mobil cihazlarda çalışıyor mu?"):
    st.write("""
    Evet! Uygulama mobil tarayıcılarda da düzgün çalışacak şekilde responsive olarak tasarlanmıştır.
    """)

with st.expander("💡 Uygulama neden bazen yanlış tahmin yapıyor?"):
    st.write("""
    Modeller, tahmin yaparken veriye dayalı karar verir. Çok kısa, tek kelimelik ya da belirsiz yorumlar model için zorlayıcı olabilir.
    """)
