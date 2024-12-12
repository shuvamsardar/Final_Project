import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis Web App", page_icon="🦜", layout="centered", initial_sidebar_state="collapsed")

@st.cache_resource
def load_model():
    model = TFBertForSequenceClassification.from_pretrained('model/')
    tokenizer = BertTokenizer.from_pretrained('model/')
    return model, tokenizer

model, tokenizer = load_model()

def plot_sentiment_distribution(reviews_df):
    sentiment_count = reviews_df['Sentiment'].value_counts().reset_index()
    sentiment_count.columns = ['Sentiment', 'Count']
    fig = px.bar(sentiment_count, x='Sentiment', y='Count', color='Sentiment', title="Sentiment Distribution of Reviews")
    st.plotly_chart(fig)


def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="tf", padding=True, truncation=True, max_length=128)
    logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
    score = tf.nn.softmax(logits).numpy()[0]
    if score[1] > 0.65:
        sentiment = "Positive"
    elif score[0] > 0.65:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score

def shorten_text(text, max_len=10):
    if len(text) > max_len:
        return text[:max_len] + '...'
    return text

if 'results' not in st.session_state:
    st.session_state.results = []

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []

st.markdown("""
    <style>
    .st-markdown-container {
        background-color: #252A34;
        color: #FFFFFF;
        border-radius: 10px;
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
        color:  #16f2b3;
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
    .score-neutral {
        color: #FFDD57;
    }
    .st-emotion-cache-1qg05tj, .st-emotion-cache-uef7qa { 
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Discover the Sentiment Behind Text</h1>", unsafe_allow_html=True)
st.write("Enter a product review to find out if the sentiment is positive, negative, or neutral:")

st.markdown("<label class='review-label'>✍️ Review Text</label>", unsafe_allow_html=True)
user_review = st.text_area("", "", placeholder="Type your review here...")

if st.button("Predict Sentiment"):
    if user_review:
        with st.spinner('Analyzing...'):
            sentiment, score = predict_sentiment(user_review)
            score_percentage = score * 100
            if sentiment == "Positive":
                st.success(f"The sentiment of the review is: **{sentiment}** 😊", icon="✅")
            elif sentiment == "Negative":
                st.error(f"The sentiment of the review is: **{sentiment}** 😟", icon="❌")
            else:
                st.info(f"The sentiment of the review is: **{sentiment}** 😐", icon="ℹ️")
            st.markdown(f"<h3 class='section-title'>Sentiment Score: &nbsp;<span class='score-positive'>Positive: {score_percentage[1]:.2f}%</span> &nbsp;<span class='score-negative'>Negative: {score_percentage[0]:.2f}%</span></h3>", unsafe_allow_html=True)
            breakdown = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Percentage": score_percentage})
            fig = px.bar(breakdown, x='Sentiment', y='Percentage', color='Sentiment', title="Sentiment Breakdown")
            st.plotly_chart(fig)
            st.session_state.results.append({"Review": user_review, "Sentiment": sentiment, "Positive Score": score_percentage[1], "Negative Score": score_percentage[0]})
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("<h2 class='section-title'>🔍 Compare Multiple Reviews</h2>", unsafe_allow_html=True)
st.markdown("<label class='compare-label'>Enter multiple reviews (comma-separated)</label>", unsafe_allow_html=True)
compare_reviews = st.text_area("", "", key="compare_reviews", placeholder="Review 1, Review 2, ...")

if st.button("Compare Sentiments", key="compare_button"):
    if compare_reviews:
        reviews = [rev.strip() for rev in compare_reviews.split(",")]
        comparison_results = []
        for rev in reviews:
            with st.spinner(f'Analyzing review: {rev}'):
                sentiment, score = predict_sentiment(rev)
                score_percentage = score * 100
                comparison_results.append({"Review": rev, "Sentiment": sentiment, "Positive Score": f"{score_percentage[1]:.2f}%", "Negative Score": f"{score_percentage[0]:.2f}%"})
        comparison_df = pd.DataFrame(comparison_results)
        st.write(comparison_df)
        comparison_df_melted = comparison_df.melt(id_vars=['Review', 'Sentiment'], value_vars=['Positive Score', 'Negative Score'], var_name='Type', value_name='Percentage')
        comparison_df_melted['Percentage'] = comparison_df_melted['Percentage'].str.rstrip('%').astype('float')
        comparison_df_melted['Review'] = comparison_df_melted['Review'].apply(shorten_text)
        fig = px.bar(comparison_df_melted, x='Review', y='Percentage', color='Type', barmode='group', title="Sentiment Comparison")
        
        # Rotate and shorten the axis text for better readability
        fig.update_layout(
            xaxis=dict(
                tickangle=0 # Rotate the labels 45 degrees
            )
        )
        st.plotly_chart(fig)
        st.session_state.comparison_results = comparison_results
        # Plot sentiment distribution
        plot_sentiment_distribution(comparison_df)
        

# # Section for uploading CSV and analyzing reviews
# st.markdown("<h2 class='section-title'>📂 Upload Reviews for Analysis</h2>", unsafe_allow_html=True)
# st.markdown("<label class='upload-label'>Choose a CSV file</label>", unsafe_allow_html=True)
# uploaded_file = st.file_uploader("", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("Uploaded Reviews:")
#     st.write(df.head())

#     if 'Review' not in df.columns:
#         st.error("The CSV file does not contain a 'Review' column. Please upload a file with a 'Review' column containing the text data you want to analyze.")
#     else:
#         if st.button("Analyze Uploaded Reviews"):
#             review_texts = df['Review'].tolist()
#             results = []
#             for review in review_texts:
#                 try:
#                     with st.spinner(f'Analyzing review: {review}'):
#                         sentiment, score = predict_sentiment(review)
#                         score_percentage = [s * 100 for s in score]
#                         results.append({"Review": review, "Sentiment": sentiment, "Positive Score": f"{score_percentage[1]:.2f}%", "Negative Score": f"{score_percentage[0]:.2f}%"})
#                 except ValueError as e:
#                     st.error(f"Error processing review '{review}': The review text must be a string. Please ensure all reviews are provided as text.")

#                     continue
            
#             result_df = pd.DataFrame(results)
#             st.write("Sentiment Analysis Results:")
#             st.write(result_df)
            
#             result_df_melted = result_df.melt(id_vars=['Review', 'Sentiment'], value_vars=['Positive Score', 'Negative Score'], var_name='Type', value_name='Percentage')
#             result_df_melted['Percentage'] = result_df_melted['Percentage'].str.rstrip('%').astype('float')
#             result_df_melted['Review'] = result_df_melted['Review'].apply(shorten_text)
            
#             fig = px.bar(result_df_melted, x='Review', y='Percentage', color='Type', barmode='group', title="Sentiment Distribution")
            
#             fig.update_layout(
#                 xaxis=dict(
#                     tickangle=0  # Rotate the labels 45 degrees
#                 )
#             )
#             st.plotly_chart(fig)




# Main app content in the sidebar
st.sidebar.header('📑 About This App')
st.sidebar.write("""
    Welcome to the Sentiment Analysis Web App! Using a robust BERT model, this app classifies user reviews as positive, negative, or neutral, providing valuable insights into customer feedback. Enjoy exploring your data! 🎉
""")

# Sidebar content
st.sidebar.header("💡 Example Reviews")

# Example Reviews in one place
st.sidebar.write("""
    **Positive Reviews:**
    - "This product exceeded my expectations! The quality is fantastic, and it's incredibly easy to use. Highly recommend it to anyone."
    - "Absolutely love this item! It has made my life so much easier, and the customer service was outstanding."

    **Negative Reviews:**
    - "I'm very disappointed with this product. It broke after just two uses, and the quality is terrible. Would not recommend."
    - "The worst purchase I've ever made. It didn't work as advertised, and the return process was a nightmare."

    **Neutral Reviews:**
    - "The product is okay. It does what it's supposed to, but nothing special. I might buy it again if it's on sale."
    - "While the features are great, the battery life is quite short. It’s a bit of a trade-off."
""")

