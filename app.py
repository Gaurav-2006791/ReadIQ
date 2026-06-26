import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Dataset
# -----------------------------
books = pd.read_csv("books.csv")

# Combine features for recommendation
books["combined_features"] = (
    books["Genre"].fillna("") + " " +
    books["Author"].fillna("") + " " +
    books["Description"].fillna("")
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(books["combined_features"])

# Cosine Similarity
similarity = cosine_similarity(feature_vectors)


def recommend_books(book_title):
    if not book_title.strip():
        return "Please enter a book title.", ""

    book_title = book_title.lower()

    matches = books[books["Title"].str.lower() == book_title]

    if matches.empty:
        return "Book not found in dataset.", ""

    index = matches.index[0]

    similarity_scores = list(enumerate(similarity[index]))

    sorted_books = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = ""

    for i, score in sorted_books:
        recommendations += (
            f"📚 {books.iloc[i]['Title']}\n"
            f"Author : {books.iloc[i]['Author']}\n"
            f"Genre  : {books.iloc[i]['Genre']}\n\n"
        )

    summary = books.iloc[index]["Description"]

    return recommendations, summary


with gr.Blocks(title="Smart Book Recommendation System") as demo:

    gr.Markdown(
        """
# 📚 Smart Book Recommendation & Summarization System

### AI/ML Internship Project

Algorithms Used:
- TF-IDF Vectorization
- Cosine Similarity
- Content-Based Recommendation
        """
    )

    book = gr.Textbox(
        label="Enter Book Title",
        placeholder="Example: Python Basics"
    )

    btn = gr.Button("Recommend")

    output1 = gr.Textbox(
        label="Recommended Books",
        lines=10
    )

    output2 = gr.Textbox(
        label="Book Summary",
        lines=8
    )

    btn.click(
        fn=recommend_books,
        inputs=book,
        outputs=[output1, output2]
    )

demo.launch()
