import streamlit as st
from datetime import datetime
import json
from streamlit_lottie import st_lottie
import datetime

year = datetime.datetime.now().year
age = year - 2013

with open("assets/Coding_animation.json", "r") as dosya:
    animasyon = json.load(dosya)



st.markdown(f"""# Ben Kimim?
Merhaba! Ben Alparslan. {age} yaşındayım. Şu an ISTDS kursunu alıyorum. Hatta bu proje ISTDS bitirme projem. Geri bildirim kısmına eksik bulduğunuz şeyleri yazabilirsiniz.
# İlgim Nasıl Başladı?
Teknolojiye ve yazılıma olan tutkum küçük yaşlarda başladı. İlk başta sıkça oynadığım oyun Minecraft'a mod yazmakla başladı. Sonra Inokids adındaki kursla kendimi geliştirdim. hatta 2. leveldan başlayım 15. levela kadar geldim. Şu an ise ISTDS kursunu tamamlamak üzereyim.
## Fun Fact:
Minecrafta Mod yazmaya 8 yaşımda başladım. Inokids kursuna aldığımda 9 yaşımdaydım. 11 yaşımda ise ISTDS ye başladım.""")

left, right = st.columns(2)
with left:
    st.markdown("""# Şu An Ne Yapıyorum?
Python, Arduino ve oyun geliştirme alanlarında projeler yapıyor, yeni şeyler öğrenmeye devam ediyorum. Kod yazmayı bir hobi olarak görüyor, her gün kendimi geliştirmek için çalışıyorum. Teknoloji dünyasında sınırların olmadığını, hayal gücünün en büyük güç olduğunu düşünüyorum.""")
with right:
    st_lottie(animasyon, speed=1, loop=True, height=300)
