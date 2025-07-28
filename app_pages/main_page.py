import streamlit as st
import streamlit.components.v1 as components

st.markdown("""# Sahte Yorum Tespiti""")

# Page Definitions for the Navigation Demo App


# Adding pages to the sidebar navigation using st.navigation


st.markdown("""## Sahte Yorum Tahmin Sitemizi Hemen Deneyin
Yüksek Doğruluk Oranıyla Çalışan Bir GameChanger.""")
st.page_link("app_pages/app.py", label="Hemen Dene", icon="❕")


with st.expander("Doğruluk oranı"):
    st.markdown("""f1 Score: %96""")
    st.image("assets/confusion_matrix.jpg")

st.markdown("""
#### Açık Kaynak Harika Bir Proje
# Tamamen Python İle Yazıldı
""")
import json
from streamlit_lottie import st_lottie

with open("assets/programming.json", "r") as dosya:
    animasyon = json.load(dosya)
st_lottie(animasyon, speed=1, loop=True, height=300)




st.markdown("""---
# Copyright (C) 2025 Mehmet Alparslan Tuncel
## GNU GPLv3 license — see: [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html)
""")