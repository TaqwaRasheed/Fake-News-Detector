import streamlit as st
import pickle
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack 
import joblib
import numpy as np

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🧠",
    layout="centered"
)

model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

page_bg_color = """
<style>
body {
background-color: #f0f2f6;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(news):
    cleaned = clean_text(news)
    x = vectorizer.transform([cleaned])
    sentiment = TextBlob(cleaned).sentiment.polarity
    sentiment_feature = [[sentiment]]
    
    x_final = hstack((x, sentiment_feature))
    
    prediction = model.predict(x_final)
    probability = model.predict_proba(x_final)  
    confidence = np.max(probability) * 100
    
    sentiment_label = 'Positive Tone' if sentiment > 0 else ('Negative Tone' if sentiment < 0 else 'Neutral Tone')
    
    return prediction[0], confidence, sentiment, sentiment_label

label_name = {
    0: 'Barely True',
    1: 'False',
    2: 'Half True',
    3: 'Mostly True',
    4: 'Pants on Fire!',
    5: 'True'
}

st.title("Fake News Detection & Sentiment Analysis")

st.info("👋 **Welcome to the Fake News Detection and Sentiment Analysis App!** \n\n"
        "Type in a news headline below, and I'll tell you if it's *Real* or *Fake* "
        "and also show you the *Sentiment Tone* based on the text. Let's find the truth together! 🕵️‍♂️")


st.write("Enter a news headline and find out if it's **Fake** or **Real**, and what its **Sentiment** is! 🎯")

user_input = st.text_area("Enter News Headline:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline!") 
    else:
        with st.spinner('Detective AI on the case....'):
            label, confidence, sentiment_score, sentiment_label = predict_news(user_input)
        
        predicted_label = label_name[label]
        
        if predicted_label in ['False', 'Pants on Fire!']:
            st.markdown(f"<h2 style='color: red;'>Prediction: {predicted_label}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: green;'>Prediction: {predicted_label}</h2>", unsafe_allow_html=True)
        
        st.subheader(f"Prediction Confidence: {confidence:.2f}%")
        st.subheader(f"Sentiment Score: {sentiment_score:.2f} ({sentiment_label})")

