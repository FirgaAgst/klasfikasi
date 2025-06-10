import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("amazon_reviews.csv")

# Preprocessing
X = df["Review Text"]
y = df["Sentiment"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_vec = vectorizer.fit_transform(X)

# Train KNN model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Streamlit UI
st.title("Klasifikasi Sentimen Ulasan Produk")
st.write("Masukkan komentar dan rating, lalu sistem akan menentukan apakah sentimennya positif, netral, atau negatif.")

# User input
review_text = st.text_area("Komentar Produk")
rating = st.slider("Rating Produk (1 - 5)", 1, 5, 3)

if st.button("Klasifikasikan"):
    # Aturan berbasis rating
    if rating < 3:
        prediction = "Negatif"
    elif rating == 3:
        prediction = "Netral"
    else:
        prediction = "Positif"

    st.subheader("Hasil Klasifikasi")
    st.write(f"Sentimen: **{prediction}**")
