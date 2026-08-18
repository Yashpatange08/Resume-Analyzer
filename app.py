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