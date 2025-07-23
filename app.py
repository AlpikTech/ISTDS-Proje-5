import streamlit as st
import pandas as pd
import pymongo
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import textstat
import os

from textstat.backend.metrics import flesch_reading_ease, flesch_kincaid_grade

st.markdown("""# Sahte Yorum Tespiti

Ne kullandım:
- NLP: NLTK, TF-IDF, Metin Analizi
- Model: `LogisticRegression`
- Data: `Kaggle` and `Web Scraping`
""")
st.warning("Sadece İngilizce Yorumlar Çalışır!")
is_user_supporting = False

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

nltk.data.path.append(nltk_data_path)
# NLTK veri setlerini indir (sadece ilk çalıştırmada gerekli)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


# Metin işleme fonksiyonları
def clean_text(text):
    if pd.isna(text):
        return ""

    # Küçük harfe çevir
    text = text.lower()

    # HTML etiketlerini temizle
    text = re.sub(r'<[^>]+>', '', text)

    # URL'leri temizle
    text = re.sub(r'http\S+', '', text)

    # Noktalama işaretlerini temizle
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Sayıları temizle
    text = re.sub(r'\d+', '', text)

    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize
    tokens = word_tokenize(text)

    # Stop words'leri kaldır ve lemmatize et
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return ' '.join(tokens)


def extract_text_features(text):
    # Metin özelliklerini çıkar
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)

    features = {
        'text_length': len(processed),
        'word_count': len(processed.split()) if processed else 0,
        'avg_word_length': np.mean([len(word) for word in processed.split()]) if processed else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_count': sum(1 for c in text if c.isupper()),
        'flesch_score': textstat.flesch_reading_ease(text) if text else 0,
        'flesch_kincaid': textstat.flesch_kincaid_grade(text) if text else 0
    }

    return processed, features


@st.cache_resource
def load_model_and_components():
    """Model ve gerekli bileşenleri yükle"""
    try:
        model = load('model.pkl')
        tfidf = load('tfidf.pkl')
        scaler = load('scaler.pkl')
        return model, tfidf, scaler
    except FileNotFoundError:
        st.error("Model dosyaları bulunamadı! Lütfen önce modeli eğitin ve kaydedin.")
        return None, None, None



def predict_fake_review(text, model, tfidf, scaler, top_n=5):
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)

    numerical_features = ['text_length', 'word_count', 'avg_word_length',
                          'exclamation_count', 'question_count', 'capital_count',
                          'flesch_score', 'flesch_kincaid']
    features = {
        'text_length': len(processed),
        'word_count': len(processed.split()),
        'avg_word_length': np.mean([len(word) for word in processed.split()]) if processed else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_count': sum(1 for c in text if c.isupper()),
        'flesch_score': flesch_reading_ease(text, lang='en') if text else 0,
        'flesch_kincaid': flesch_kincaid_grade(text, 'en') if text else 0
    }

    tfidf_vector = tfidf.transform([processed])
    numerical_vector = scaler.transform([[features[col] for col in numerical_features]])
    X_new = sp.hstack([tfidf_vector, numerical_vector])

    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0]

    # TF-IDF feature isimleri
    feature_names = tfidf.get_feature_names_out()
    # Yorumdaki kelimeler
    tokens = processed.split()

    # Model katsayıları (TF-IDF kısmı)
    coefs = model.coef_[0][:len(feature_names)]

    # Tokenların katsayıları
    token_coefs = {}
    for token in tokens:
        if token in feature_names:
            idx = np.where(feature_names == token)[0][0]
            token_coefs[token] = coefs[idx]

    # En pozitif ve en negatif etkili kelimeler
    top_positive = sorted([(k, v) for k, v in token_coefs.items() if v > 0], key=lambda x: x[1], reverse=True)[:top_n]
    top_negative = sorted([(k, v) for k, v in token_coefs.items() if v < 0], key=lambda x: x[1])[:top_n]

    return {
        'prediction': 'Sahte' if prediction == 1 else 'Gerçek',
        'fake_probability': probability[1],
        'real_probability': probability[0],
        'top_positive': top_positive,
        'top_negative': top_negative
    }



# Streamlit arayüzü
review = st.text_area("Yorumunuzu girin:", key="comment", height=100)
if st.checkbox("MongoDB'ye kaydet (Herkes görebilir)"):
    is_user_supporting = True

if st.button("Analiz Et"):
    if review.strip():
        # Model ve bileşenleri yükle
        model, tfidf, scaler = load_model_and_components()

        if model and tfidf and scaler:
            # Tahmin yap
            with st.spinner('Yorum analiz ediliyor...'):
                result = predict_fake_review(review, model, tfidf, scaler)

            if result:
                # Sonuçları göster
                st.subheader("Analiz Sonucu:")

                col1, col2 = st.columns(2)

                with col1:
                    if result['fake_probability'] > 0.70:
                        st.metric("Tahmin", "Sahte")
                    else:
                        st.metric("Tahmin", "Gerçek")


                with col2:
                    confidence = max(result['fake_probability'], result['real_probability'])
                # Detaylı probabilities
                st.subheader("Detaylı Sonuçlar:")
                st.write(f"🔴 Sahte olma olasılığı: **{result['fake_probability']:.2%}**")
                st.write(f"🟢 Gerçek olma olasılığı: **{result['real_probability']:.2%}**")

                st.subheader("Karar Verirken Etkili Kelimeler 🔍")
                if result['top_positive']:
                    with st.expander("Sahte Yorum İhtimalini Artıran Kelimeler:"):
                        for word, coef in result['top_positive']:
                            st.markdown(f"- `{word.capitalize()}` (etki: {coef:.3f})")
                if result['top_negative']:
                    with st.expander("Gerçek Yorum İhtimalini Artıran Kelimeler:"):
                        for word, coef in result['top_negative']:
                            st.markdown(f"- `{word.capitalize()}` (etki: {coef:.3f})")

                # Görsel gösterim
                import matplotlib.pyplot as plt
                import mplcyberpunk

                plt.style.use("cyberpunk")
                plt.rcParams['font.family'] = 'Segoe UI'
                fig, ax = plt.subplots(figsize=(8, 4))
                categories = ['Gerçek', 'Sahte']
                probabilities = [result['real_probability'], result['fake_probability']]

                bars = ax.bar(categories, probabilities, alpha=0.7)
                ax.set_ylabel('Olasılık')
                ax.set_title('Yorum Analiz Sonucu')
                ax.set_ylim(0, 1)

                # Bar üzerinde değerleri göster
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{prob:.2%}', ha='center', va='bottom')

                st.pyplot(fig)
                if is_user_supporting == True:
                    import pandas as pd
                    from pymongo import MongoClient

                    # 1️⃣ DataFrame oluştur (örnek)
                    data = {
                        'text': [review],
                        'real_probability': [result['real_probability']],
                        'fake_probability': [result['fake_probability']]
                    }
                    df = pd.DataFrame(data)

                    # 2️⃣ MongoDB bağlantısı kur
                    client = MongoClient("mongodb+srv://AlpikTech:ijQkFS1b6drq8eHY@istdsproje5.eb23zqp.mongodb.net/")  # MongoDB URI'ını buraya yaz
                    db = client['reviews']  # Veritabanı adı
                    collection = db['datas']  # Koleksiyon adı

                    # 3️⃣ DataFrame'i dictionary listesine çevir
                    records = df.to_dict(orient='records')

                    # 4️⃣ MongoDB'ye yükle
                    mongo_result = collection.insert_many(records)


    else:
        st.warning("Lütfen bir yorum girin.")


# Bilgi kutusu
with st.expander("Nasıl çalışır?"):
    st.write("""
    Bu uygulama, yorumların sahte mi gerçek mi olduğunu belirlemek için:

    1. **Metin temizleme**: Yorumlar temizlenir ve standardize edilir
    2. **Özellik çıkarımı**: TF-IDF, metin uzunluğu, noktalama işaretleri gibi özellikler çıkarılır
    3. **Makine öğrenmesi**: Eğitilmiş model ile tahmin yapılır
    4. **Sonuç**: Sahte/gerçek tahmini ve güven oranı gösterilir
    """)