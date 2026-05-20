import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/chatgpt_reviews_clean_balanced.csv"
    )

    df["content"] = df["content"].astype(str)

    return df


@st.cache_resource
def build_model(texts):

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    return vectorizer, X


df = load_data()

vectorizer, X = build_model(
    df["content"]
)


st.title("Поиск похожих отзывов")
st.subheader("Приложение ChatGPT, на основе TF-IDF.")

query = st.text_input(
    "Введите фразу:"
)

top_n = st.slider(
    "Количество результатов:",
    3,
    20,
    5
)


if query:

    query_vector = vectorizer.transform(
        [query]
    )

    similarity = cosine_similarity(
        query_vector,
        X
    ).flatten()

    indices = np.argsort(
        similarity
    )[::-1]

    indices = indices[
        similarity[indices] > 0
    ]

    indices = indices[:top_n]

    st.subheader("Найденные отзывы")

    if len(indices) == 0:

        st.warning(
            "Совпадений не найдено"
        )

    for idx in indices:

        st.write(
            f"Cosine similarity: {similarity[idx]:.3f}, Score: {df.iloc[idx]["score"]}"
        )

        st.info(
            df.iloc[idx]["content"]
        )