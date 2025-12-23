import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
import joblib
import os

st.set_page_config(page_title="News Article Classifier", layout="centered")

st.title("News Article Classification App")
st.write("Paste any news article content and get its predicted category.")

@st.cache_resource
def load_model():
    # Define model save path
    model_path = "news_classifier.pkl"

    # Check if model already exists
    if os.path.exists(model_path):
        return joblib.load(model_path)

    # Fetch dataset
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    categories = newsgroups.target_names

    # Pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(newsgroups.data, newsgroups.target)

    # Save model and categories
    joblib.dump((model, categories), model_path)

    return model, categories

# Load model
model, categories = load_model()

# User input
article = st.text_area("Paste the news article here:", height=250)

if st.button("Classify"):
    if article.strip() == "":
        st.warning("Please enter some content before classification.")
    else:
        prediction = model.predict([article])[0]
        st.success(f"Predicted Category: **{categories[prediction]}**")

st.write("Streamlit app loaded!")
