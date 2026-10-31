import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# Download stopwords if required
nltk.download("stopwords", quiet=True)


# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Text preprocessing
stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"no", "nor", "not"}

stemmer = SnowballStemmer("english")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


# Prediction function
def predict_sentiment(review):
    cleaned_review = preprocess_text(review)

    review_tf = tfidf.transform([cleaned_review])

    prediction = model.predict(review_tf)[0]

    if prediction == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"


# Streamlit interface
st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write(
    "Enter a restaurant review to predict "
    "whether its sentiment is positive or negative."
)

review = st.text_area(
    "Enter your review here:",
    placeholder="Example: The food was delicious and the service was excellent."
)


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        result = predict_sentiment(review)

        if result.startswith("Positive"):
            st.success(f"Sentiment: {result}")
        else:
            st.error(f"Sentiment: {result}")