import streamlit as st
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Function to extract the candidate's name
def extract_name(text):
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            return line.strip()
    return "Name Not Found"

# Function to extract key details using NLP and regex
def extract_entities(text):
    doc = nlp(text)
    entities = {
        "NAME": extract_name(text),
        "EMAIL": "",
        "PHONE": "",
        "EDUCATION": [],
        "SKILLS": []
    }

    # Extract EMAIL using regex
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    email_matches = email_pattern.findall(text)
    if email_matches:
        entities["EMAIL"] = email_matches[0]

    # Extract PHONE using regex
    phone_pattern = re.compile(r'\(?\+?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}')
    phone_matches = phone_pattern.findall(text)
    if phone_matches:
        entities["PHONE"] = phone_matches[0]

    # Extract EDUCATION entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["EDUCATION"].append(ent.text)

    # Extract SKILLS from a "Skills" section
    skills_section = False
    for line in text.split("\n"):
        if "skills" in line.lower():
            skills_section = True
        elif skills_section:
            if line.strip() == "":
                break
            entities["SKILLS"].extend([skill.strip() for skill in line.split(",")])

    return entities

# Function to calculate job match score
def get_similarity(resume_text, job_desc):
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_desc])
    similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(similarity * 100, 2)



# Streamlit App
st.title("AI Resume Analyzer")

# Input job description
job_description = st.text_area("Enter Job Description", height=200)

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    resume_results = []

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Extract text
        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text:
            st.warning(f"Could not extract text from {uploaded_file.name}")
            continue

        # Extract details
        entities = extract_entities(resume_text)

        # Compute match score
        match_score = get_similarity(resume_text, job_description)

        # Store results
        resume_results.append({
            "Rank": idx,
            "Resume Name": uploaded_file.name,
            "Candidate Name": entities["NAME"],
            "Email": entities["EMAIL"],
            "Phone": entities["PHONE"],
            "Education": ", ".join(entities["EDUCATION"]),
            "Skills": ", ".join(entities["SKILLS"]),
            "Match Score (%)": match_score
        })

    # Convert results to DataFrame and rank
    df = pd.DataFrame(resume_results)
    df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    # Display ranked resumes
    st.subheader("🔹 Resume Ranking")
    st.dataframe(df)

    # Show top candidate
    st.success(f" **Best Match:** {df.iloc[0]['Candidate Name']} with {df.iloc[0]['Match Score (%)']}% match!")

