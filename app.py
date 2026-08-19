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

# Download required NLTK data quietly 
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    # Note: Corrected the tagger name here
    nltk.download("averaged_perceptron_tagger_eng") 


# ------------------------------StreamLit Setup-------------------------------

st.set_page_config(page_title="Y-Resume", page_icon="📑", layout="wide")

st.markdown('''
### Upload Your Resume And Check Your Potential
This Tool uses **TF-IDF + Cosine similarity** to Analyze your Resume Against a Job Title.
''')

with st.sidebar:
    st.header("About")
    st.info("""
    This Tool Helps You:
    - Measure how your resume matches a job Description
    - Identify Important Aspects
    - Improve Your Resume Based Upon Missing terms
    """)
    st.header("How It Works")
    st.write("""
    1. Upload Your Resume (PDF)
    2. Tell Us Your Job Description
    3. Click **Analyze match**
    4. Review **Score** & **Suggestions**
    """)

# ----------------------------Helper Functions-----------------------------------

# Extracting text from pdf Function
def extract_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        # Iterate through actual page objects
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + " "
        return full_text
    except Exception as e:
        st.error(f"Error While reading PDF: {e}")
        return ""

# Text Cleaning
def clean_text(text):
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", '', text) 
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", ' ', text).strip() 
    return text

# Removing Stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words) 

# Calculate Cosine Similarity
def calculate_similarities(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    # Corrected duplicate variable processing
    job_processed = remove_stopwords(clean_text(job_description)) 
    
    vectorizer = TfidfVectorizer()
    Tfid_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    
    # Corrected cosine similarity matrix indexing syntax
    score = cosine_similarity(Tfid_matrix[0:1], Tfid_matrix[1:2])[0][0] * 100
    
    return round(score, 2), resume_processed, job_processed

# Extract Keywords
def extract_keywords(text, num_keyword=100):
    words = word_tokenize(text)
    words = [w for w in words if len(w) > 2]
    tagged_words = pos_tag(words)
    nouns = [w for w, pos in tagged_words if pos.startswith("NN") or pos.startswith("JJ")]
    word_freq = Counter(nouns)
    # Corrected object method call
    return word_freq.most_common(num_keyword) 


# -------------------Main Application------------------------

def main():
    uploaded_file = st.file_uploader("Upload Your File Here (PDF)", type=['pdf'])
    job_Description = st.text_area("Tell Us About your Job Type", height=200)   

    # Streamlined button logic
    if st.button("Analyze match"):
        if not uploaded_file:
            st.warning("Please Upload your Resume")
        elif not job_Description:
            st.warning("Provide us your job Description") # Corrected st.warning typo
        else:
            with st.spinner("Analyzing Your Resume......"):
                resume_text = extract_text(uploaded_file)
                
                if not resume_text:
                    st.error("Could Not Able to extract the Information from Pdf")
                    return

                # Understanding similarity
                similarity_score, resume_processed, job_processed = calculate_similarities(resume_text, job_Description)

                # Result Output
                st.subheader("Result")
                st.metric("Match score", f"{similarity_score:.2f}%")

                # Gauge chart
                # Corrected plt.subplot -> plt.subplots
                fig, ax = plt.subplots(figsize=(6, 0.5)) 
                colors = ['#ff4b4b', "#fffb00", "#00fa21"]
                
                # Logic to pick color safely based on score
                color_index = min(int(similarity_score // 34), 2) 
                
                ax.barh([0], [similarity_score], color=colors[color_index])
                ax.set_xlim(0, 100)
                ax.set_xlabel("Match Percentage")
                ax.set_title("Resume Job Match")
                
                # Render plot in Streamlit
                st.pyplot(fig)

                # Final recommendations
                if similarity_score < 40:
                    st.warning("Low Match. Your resume aligned very poorly.")
                elif similarity_score < 70:
                    st.info("Good match. Your resume aligns fairly well.")
                else:
                    st.success("Excellent Match! Your resume strongly aligns.")

if __name__ == "__main__":
    main()