import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis Web App", page_icon="🦜", layout="centered", initial_sidebar_state="collapsed")

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model = TFBertForSequenceClassification.from_pretrained('model/')
    tokenizer = BertTokenizer.from_pretrained('model/')
    return model, tokenizer

model, tokenizer = load_model()

# Function to predict sentiment and score
def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="tf", padding=True, truncation=True, max_length=128)
    logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    
    # Calculate sentiment score
    score = tf.nn.softmax(logits).numpy()[0]
    
    return "Positive" if predicted_class == 1 else "Negative", score

# Initialize session state to store results if not already initialized
if 'results' not in st.session_state:
    st.session_state.results = []

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300..800;1,300..800&family=Pacifico&family=Volkhov:ital,wght@0,400;0,700;1,400;1,700&display=swap');
            
            
            
    .st-markdown-container {
        background-color: #252A34;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 2rem;
        font-size: 1.2rem;
    }

    .stMarkdown p {
        font-size: 1.2rem;
    }

    .review-label, .compare-label, .upload-label {
        font-size: 1.2rem;
        color: #FFFFFF;
    }

    .stTextArea textarea {
        width: 100%;
        height: 170px;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }

    .main-title {
        font-size: 2.5rem;
        color: #FFD700;
        font-family: "Pacifico", cursive;
        font-weight: 200;
        font-style: normal;
    }

    .section-title {
        font-size: 1.5rem;
        color: #87CEFA;
    }

    .score-positive {
        color: #2EB086;
    }

    .score-negative {
        color: #de3f04;
    }
    .st-emotion-cache-1qg05tj, .st-emotion-cache-uef7qa { 
            display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
st.markdown("<h1 class='main-title'>✨ Discover the Sentiment Behind Text</h1>", unsafe_allow_html=True)
st.write("Enter a product review to find out if the sentiment is positive or negative:")

# User input for single review
st.markdown("<label class='review-label'>✍️ Review Text</label>", unsafe_allow_html=True)
user_review = st.text_area("", "", placeholder="Type your review here...")

# Predict sentiment button
if st.button("Predict Sentiment"):
    if user_review:
        sentiment, score = predict_sentiment(user_review)
        
        # Convert scores to percentages
        score_percentage = score * 100

        # Display sentiment result
        if sentiment == "Positive":
            st.success(f"The sentiment of the review is: **{sentiment}** 😊", icon="✅")
        else:
            st.error(f"The sentiment of the review is: **{sentiment}** 😟", icon="❌")

        # Display sentiment score in one line
        st.markdown(f"<h3 class='section-title'>Sentiment Score: &nbsp &nbsp<span class='score-positive'>Positive: {score_percentage[1]:.2f}%</span> &nbsp &nbsp <span class='score-negative'>Negative: {score_percentage[0]:.2f}%</span></h3>", unsafe_allow_html=True)

        # Breakdown of sentiment components
        breakdown = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Percentage": score_percentage})
        fig = px.bar(breakdown, x='Sentiment', y='Percentage', color='Sentiment', title="Sentiment Breakdown")
        st.plotly_chart(fig)

        # Save result for export
        st.session_state.results.append({"Review": user_review, "Sentiment": sentiment, "Positive Score": score_percentage[1], "Negative Score": score_percentage[0]})

    else:
        st.warning("Please enter a review to analyze.")

# Comparison feature in main content section
st.markdown("<h2 class='section-title'>🔍 Compare Multiple Reviews</h2>", unsafe_allow_html=True)
st.markdown("<label class='compare-label'>Enter multiple reviews (comma-separated)</label>", unsafe_allow_html=True)
compare_reviews = st.text_area("", "", key="compare_reviews", placeholder="Review 1, Review 2, ...")

if st.button("Compare Sentiments", key="compare_button"):
    if compare_reviews:
        reviews = [rev.strip() for rev in compare_reviews.split(",")]
        comparison_results = []
        for rev in reviews:
            sentiment, score = predict_sentiment(rev)
            
            # Convert scores to percentages
            score_percentage = score * 100
            
            comparison_results.append({"Review": rev, "Sentiment": sentiment, "Positive Score": score_percentage[1], "Negative Score": score_percentage[0]})

        comparison_df = pd.DataFrame(comparison_results)
        st.write(comparison_df)

        # Visualization for comparison results
        fig = px.bar(comparison_df, x='Review', y=['Positive Score', 'Negative Score'], barmode='group', title="Sentiment Comparison")
        st.plotly_chart(fig)

        # Save comparison results to session state for export
        st.session_state.comparison_results = comparison_results

# File upload for bulk review analysis in main content section
st.markdown("<h2 class='section-title'>📂 Upload Reviews for Analysis</h2>", unsafe_allow_html=True)
st.markdown("<label class='upload-label'>Choose a CSV file</label>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Reviews:")
    st.write(df.head())
    
    if st.button("Analyze Uploaded Reviews"):
        review_texts = df['Review'].tolist()
        results = []
        for review in review_texts:
            sentiment, score = predict_sentiment(review)
            
            # Convert scores to percentages
            score_percentage = score * 100
            
            results.append({"Review": review, "Sentiment": sentiment, "Positive Score": score_percentage[1], "Negative Score": score_percentage[0]})
        
        result_df = pd.DataFrame(results)
        st.write("Sentiment Analysis Results:")
        st.write(result_df)
        
        # Visualization of sentiment distribution
        sentiment_counts = result_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig)

# About section in the sidebar
st.sidebar.header("About This App 📖")
st.sidebar.write("""
    Welcome to the Sentiment Analysis Web App! This application uses a powerful BERT model to classify the sentiment of user reviews as either positive or negative. Perfect for gaining insights into customer feedback.
    
    **Features:**
    - ✨ Analyze individual reviews
    - 🧐 Compare sentiments of multiple reviews
    - 📊 Bulk review analysis through file upload
""")
