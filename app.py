import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

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
    
    # Calculate sentiment score (simplified for demo purposes)
    score = tf.nn.softmax(logits).numpy()[0]
    
    return "Positive" if predicted_class == 1 else "Negative", score

# Initialize session state to store results if not already initialized
if 'results' not in st.session_state:
    st.session_state.results = []

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []

# Streamlit app layout
st.title("Sentiment Analysis Web App")
st.write("Enter a product review to find out if the sentiment is positive or negative:")

# User input for single review
user_review = st.text_area("Review Text", "")

# Predict sentiment button
if st.button("Predict Sentiment"):
    if user_review:
        sentiment, score = predict_sentiment(user_review)
        
        # Display sentiment result
        if sentiment == "Positive":
            st.success(f"The sentiment of the review is: **{sentiment}**", icon="✅")
            sentiment_color = "green"
        else:
            st.error(f"The sentiment of the review is: **{sentiment}**", icon="❌")
            sentiment_color = "red"

        # Display sentiment score
        st.markdown(f"<h3 style='color:{sentiment_color};'>Sentiment Score: {score[1]:.2f} (Positive), {score[0]:.2f} (Negative)</h3>", unsafe_allow_html=True)

        # Breakdown of sentiment components
        breakdown = {
            "Positive": score[1],
            "Negative": score[0],
            "Neutral": 1 - (score[0] + score[1])  # Neutral score calculation
        }
        breakdown_df = pd.DataFrame(list(breakdown.items()), columns=["Sentiment", "Percentage"])
        st.bar_chart(breakdown_df.set_index("Sentiment"))

        # Save result for export
        st.session_state.results.append({"Review": user_review, "Sentiment": sentiment, "Positive Score": score[1], "Negative Score": score[0]})

    else:
        st.warning("Please enter a review to analyze.")

# Comparison feature (Example with predefined reviews)
st.subheader("Comparison Feature")
compare_reviews = st.text_area("Enter multiple reviews (comma-separated)", "")
if st.button("Compare Sentiments"):
    if compare_reviews:
        reviews = [rev.strip() for rev in compare_reviews.split(",")]
        comparison_results = []
        for rev in reviews:
            sentiment, score = predict_sentiment(rev)
            comparison_results.append({"Review": rev, "Sentiment": sentiment, "Positive Score": score[1], "Negative Score": score[0]})

        comparison_df = pd.DataFrame(comparison_results)
        st.write(comparison_df)

        # Save comparison results to session state for export
        st.session_state.comparison_results = comparison_results

# Export results section
if st.button("Export Results"):
    if st.session_state.results or st.session_state.comparison_results:  # Check if there are any results to export
        # Combine individual results and comparison results into one DataFrame for export
        export_data = pd.DataFrame(st.session_state.results + st.session_state.comparison_results)  # Combine both results
        
        export_file = "sentiment_analysis_results.csv"  # Name of the file to export
        
        # Convert the DataFrame to CSV format and provide it for download
        st.download_button(
            label="Download CSV",
            data=export_data.to_csv(index=False).encode('utf-8'),  # Encode the DataFrame to CSV
            file_name=export_file,
            mime='text/csv'  # Set MIME type for CSV
        )
    else:
        st.warning("No results available to export.")

# About section
st.sidebar.header("About")
st.sidebar.write("""
    This app uses a BERT model for sentiment analysis, classifying user reviews as positive or negative.
    - Model trained on a dataset of verified reviews.
    - Provides insights into customer feedback.
""")
