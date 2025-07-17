import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import joblib
import requests

#load model and encoders
model = joblib.load('model1.pkl')
le = joblib.load('company_encoder.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
feature_names = joblib.load('feature_names.pkl')

def mentions_someone(text):
    return int(bool(re.search(r'<mention>', text)))

def preprocess_input(company, text, has_media):
    sentiment = TextBlob(text).sentiment.polarity
    word_count = len(text.split())
    char_count = len(text)
    hour = pd.Timestamp.now().hour
    hashtag_count = len(re.findall(r'#\w+', text))
    mentions_anyone = mentions_someone(text)
    company_encoded = le.transform([company])[0]

    #base features
    base = {
        'word_count': word_count,
        'char_count': char_count,
        'hour': hour,
        'sentiment': sentiment,
        'hashtag_count': hashtag_count,
        'has_media': int(has_media),
        'mentions_anyone': mentions_anyone,
        'company_encoded': company_encoded
    }

    #TF-IDF
    tfidf_vec = vectorizer.transform([text])
    tfidf_features = dict(zip(vectorizer.get_feature_names_out(), tfidf_vec.toarray()[0]))

    #merge base + tfidf
    all_features = {**base, **tfidf_features}

    #create df with same order
    input_df = pd.DataFrame([all_features])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    return input_df
def get_ai_generated_tweet(prompt):
    try:
        res = requests.post(
            "http://localhost:5000/generate_ai",
            json={"prompt": prompt},
            timeout=10
        )
        return res.json().get("tweet", "Failed to generate tweet.")
    except Exception as e:
        return f"Error: {e}"

#app
st.title("AI Tweet Generator + Like Predictor")

company = st.selectbox("Company", le.classes_.tolist())
has_media = st.checkbox("Attach media?", value=True)

prompt = st.text_input("Enter a prompt to guide tweet generation:", f"{company} launches")

if st.button("Generate Tweet + Predict Likes"):
    with st.spinner("Generating tweet..."):
        tweet = get_ai_generated_tweet(prompt)

    st.markdown(f"AI-Generated Tweet:\n>{tweet}")

    input_df = preprocess_input(company, tweet, has_media)
    log_likes = model.predict(input_df)[0]
    predicted_likes = int(np.expm1(log_likes))

    st.success(f"Predicted Likes: **{predicted_likes}**")

#manual input
st.markdown("---")
st.markdown("Or test a custom tweet:")

custom_tweet = st.text_area("Enter your own tweet")

if st.button("Predict Likes for Custom Tweet"):
    input_df = preprocess_input(company, custom_tweet, has_media)
    log_likes = model.predict(input_df)[0]
    predicted_likes = int(np.expm1(log_likes))
    st.success(f"Predicted Likes: **{predicted_likes}**")

def get_ai_generated_tweet(prompt):
        res = requests.post(
            "http://localhost:5000/generate_ai",
            json={"prompt": prompt}
        )
        return res.json().get("tweet", "Failed to generate tweet.")