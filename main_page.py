import streamlit as st
# st.markdown("""# Sahte Yorum Tespiti""")

# Page Definitions for the Navigation Demo App
pages = [
    st.Page("app_pages/main_page.py", title="Ana Sayfa", icon="🏠"),
    st.Page("app_pages/app.py", title="Yorum Analizi", icon="💬"),
    st.Page("app_pages/feedback.py", title="Geri Bildirim", icon="⭐"),
    st.Page("app_pages/whoami.py", title="Ben Kimim?", icon="👤"),
    st.Page("app_pages/SSS.py", title="Sıkça Sorulan Sorular", icon="❓"),
    st.Page("app_pages/HowItWorks.py", title="Nasıl Çalışır?", icon="⚙️"),

]

# Adding pages to the sidebar navigation using st.navigation
pg = st.navigation(pages, position="top", expanded=True)
# Running the app
pg.run()


# st.markdown("""## Sahte yorum tahmin etme Sitemize hoş geldiniz""")