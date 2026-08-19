from pydoc import pager

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Natural Language Tool Kit For 

nltk.download("punkt_tab"),
nltk.download("stopwords"),
nltk.download("average_perceptron_tagger_eng"),


# ------------------------------StreamLit Setup-------------------------------


st.set_page_config(page_title="Y-Resume",page_icon="📑",layout="wide")
st.markdown('''
Upload Your Resume And Check Your Potential Is it Worth to Addressed By HR

This Tool uses **TF-IDF + Cosine similarity+ Some Special BreakPoints** to Analyze your Resume Against Job Title
''')

with st.sidebar:
    st.header("About")
    st.info("""
    This  Tool Helps You:
    - Measure how your resume  matches a job Description
    - Identify Important Aspect
    - Improve Your Resume Based Upon Missing terms
""")
    st.header("How It Works")
    st.write("""
    This  Tool Helps You:

    1. Upload Your Resume (PDF)

    2. Tell Us Your Job Description

    3. Click **Analyze Resume**

    4. Review **Score** & **Suggestions**
    5
""")

# ----------------------------Helper-----------------------------------

#Extracting text from pdf Function
def extract_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for text in pdf_reader.pages:
            text = text+pager.extract_text()
            return text
    except Exception as e:
        st.error(f"Error While reading PDF: {e}")
        return ""

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]",'',text) # all the letter between a-z and A-Z only remains others will be removed completely
    text = re.sub(r"\s+",'',text).strip() #used to remove or strip white SPaces from text
    return text


# Removing Stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words) 