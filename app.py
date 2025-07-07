import streamlit as st
import pandas as pd
import pymongo

st.markdown("""# Sahte Yorum Tespidi

Ne kullandım:
- NLP: 
- Model: `XGBosst`
- Data: `Kaggle` and `Web Scraping`
""")





catagory = st.selectbox(
    "Select a Catagory",
    ("Kindle Store", "Books", "Pet Supplies", "Home and Kitchen", "Electronics", "Sports and Outdoors", "Tools and Home Improvement", "Clothing Shoes and Jewelry", "Toys and Games", "Movies and TV")
)

review = st.text_area("Enter Review", key="comment")
st.button("Submit")
#if st.button("Submit"):


st.checkbox("Store in MongoDB for Supporting Us. (Everyone Can See it)")
