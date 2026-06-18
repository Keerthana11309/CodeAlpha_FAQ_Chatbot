import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔥 NEW CHATBOT VERSION")
st.write("Ask your question below 👇")

faqs = {
    "What is AI?": "AI means Artificial Intelligence where machines simulate human intelligence.",
    "What is machine learning?": "Machine learning is a part of AI where systems learn from data.",
    "What is Python?": "Python is used for AI, web development, data science and automation.",
    "What is NLP?": "NLP is Natural Language Processing that helps computers understand human language.",
    "What is Deep Learning?": "Deep Learning is a subset of machine learning that uses neural networks.",
    "What is Java?": "Java is a popular object-oriented programming language.",
    "What is HTML?": "HTML is used to create the structure of web pages.",
    "What is CSS?": "CSS is used to style and design web pages.",
    "What is JavaScript?": "JavaScript is used to add interactivity to web pages.",
    "What is Web Development?": "Web development is building websites using HTML, CSS, JavaScript and backend technologies."
}

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_question):
    user_question = user_question.lower()

    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, question_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.8:
        return "Sorry, I can only answer questions available in the FAQ database."

    return answers[best_match]

user_input = st.text_input("Ask your question")

if user_input:
    st.success("Bot Answer:")
    st.write(get_answer(user_input))