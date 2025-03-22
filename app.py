import streamlit as st
from dotenv import load_dotenv
import os
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# GROQ API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Llama 3 model using ChatGroq with the GROQ API
llama3_model = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# Define the prompt template for fake news detection
fake_news_template = """
You are a fake news detection system. Analyze the following news article and classify it as 'real' or 'fake' based on its content:
Article: {article_text}
"""

# Create prompt using LangChain
fake_news_prompt = PromptTemplate(template=fake_news_template, input_variables=["article_text"])

# Initialize the LangChain LLMChain with the Llama 3 model
fake_news_chain = LLMChain(llm=llama3_model, prompt=fake_news_prompt)

def classify_article(article_text):
    response = fake_news_chain.run({"article_text": article_text})
    return response

# Set up Streamlit UI with a wide layout
st.set_page_config(page_title="Fake News Detection System", layout="wide")

# Main content
st.markdown("""
    <style>
        .main-text {
            font-size: 2em;
            text-align: center;
            margin-top: 2em;
        }
        .input-section {
            margin-left: auto;
            margin-right: auto;
            width: 80%;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-text">Fake News Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="main-text">Enter the news article below to classify it as real or fake.</p>', unsafe_allow_html=True)

# News Article Input section with centered alignment and full-width style
st.markdown('<div class="input-section">', unsafe_allow_html=True)
news_article = st.text_area("", height=300)
st.markdown('</div>', unsafe_allow_html=True)

if st.button('Classify'):
    if news_article:
        with st.spinner("Classifying the news article..."):
            result = classify_article(news_article)
            st.subheader("Classification Result")
            st.write(result)
    else:
        st.write("Please enter a news article.")