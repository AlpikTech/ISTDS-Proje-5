import streamlit as st

def how_it_works():
    st.title("Nasıl Çalışır?")

    st.markdown("""
    Bu uygulama bir **Sahte Yorum Tespit Sistemi** olarak çalışıyor. Ana işleyişi şu şekilde:

    ## **Uygulamanın Temel İşleyişi**

    ### **1** Metin İşleme ve Temizleme
    - 🔤 Yorum küçük harflere dönüştürülür
    - 🧹 HTML etiketleri, URL'ler, sayılar, noktalama işaretleri temizlenir
    - 🛑 Stop words kaldırılır, köklerine indirgenir (lemmatization)

    ### 2 Özellik Çıkarımı
    Yorumdan şu istatistiksel özellikler çıkarılır:
    - 📈 **TF-IDF** kelime skorları
    - ✍️ **Metin uzunluğu**, **kelime sayısı**, **ortalama kelime uzunluğu**
    - ❗ Ünlem (!) ve ❓ Soru işareti sayıları
    - 🔠 Büyük harf kullanımı oranı
    - 📚 Flesch okuma kolaylığı skoru

    ### 3 Makine Öğrenmesi ile Tahmin
    - 🤖 **Logistic Regression** modeli kullanılır
    - 📊 Model, **Kaggle verisi** ile eğitilmiştir
    - ✅ Tahmin doğruluğu: **%96**

    ### 4 Sonuçların Gösterimi
    - 🧪 Yorum **Sahte mi, Gerçek mi?** sonucu
    - 🎯 **Tahmin yüzdesi** (animasyonlu gösterge ile)
    - 🧠 **Etkili kelimeler** analizi (hangi kelimeler kararı etkiledi?)

    ---

    ## 🛠 Teknik Altyapı
    - **Frontend**: Streamlit 🖥️
    - **NLP**: NLTK, TF-IDF 📚
    - **Model**: LogisticRegression 🤖
    - **Veritabanı**: MongoDB (isteğe bağlı kayıt) 🗃️
    - **Dil Desteği**: Sadece İngilizce 🇬🇧

    ---

    ## 📊 Örnek Kullanım Akışı
    1. 🧾 Kullanıcı yorum yazar
    2. ▶️ "Analiz Et" butonuna tıklar
    3. 📊 Model işleyip **%XX gerçeklik oranı** verir
    4. 🔍 Etkili kelimeleri listeler

    🛒 Uygulama özellikle **e-ticaret yorumlarının güvenilirliğini** ölçmek için tasarlanmıştır!
    """)

# Eğer sekme sistemin varsa şöyle çağırabilirsin:
# with tab2:
#     how_it_works()

# Tek sayfa demo için:
if __name__ == "__main__":
    how_it_works()
