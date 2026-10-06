import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import contractions
import pdfplumber
import tempfile
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Page Config ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 16px 30px;
        width: 100%;
        font-size: 18px;
        transition: all 0.3s;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .score-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .score-number {
        font-size: 72px;
        font-weight: 800;
        margin: 10px 0;
    }
    .score-label {
        font-size: 14px;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .good { color: #00d2ff; }
    .ok { color: #f7b731; }
    .bad { color: #fc5c65; }
    
    .section-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        margin: 3px;
        font-weight: 500;
    }
    .pill-green { background: rgba(0, 210, 255, 0.15); color: #00d2ff; border: 1px solid rgba(0, 210, 255, 0.3); }
    .pill-red { background: rgba(252, 92, 101, 0.15); color: #fc5c65; border: 1px solid rgba(252, 92, 101, 0.3); }
    .pill-yellow { background: rgba(247, 183, 49, 0.15); color: #f7b731; border: 1px solid rgba(247, 183, 49, 0.3); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your resume • Paste the job description • Get instant AI-powered feedback</div>', unsafe_allow_html=True)

# --- Data Processing Functions ---
def text_cleaning(text: str) -> str:
    if pd.isnull(text) or not text:
        return ""
    text = str(text).lower().strip()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{1,3}[-./]?\d{1,3}[-./]?\d{1,4}\b', '', text)
    # Replace punctuation with space to prevent words from merging
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            text = ""
            for page in pdf.pages:
                text = " ".join([text, page.extract_text() or ""])
        return text.strip()
    finally:
        os.unlink(tmp_path)

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_model():
    # all-MiniLM-L6-v2 is specifically trained for semantic similarity
    # It produces MUCH more meaningful similarity scores than raw DistilBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def get_tfidf_keywords(jd_text, resume_text, top_n=25):
    """Extract important keywords using TF-IDF and compare."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2), 
            max_features=500,
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get JD keyword importance scores
        jd_scores = tfidf_matrix[0].toarray().flatten()
        resume_scores = tfidf_matrix[1].toarray().flatten()
        
        # Get top JD keywords by importance
        jd_keyword_indices = jd_scores.argsort()[::-1][:top_n * 2]
        
        matched = []
        missing = []
        
        for idx in jd_keyword_indices:
            keyword = feature_names[idx]
            jd_importance = jd_scores[idx]
            if jd_importance > 0:
                if resume_scores[idx] > 0:
                    matched.append((keyword, jd_importance))
                else:
                    missing.append((keyword, jd_importance))
        
        # Sort by importance and take top N
        matched = sorted(matched, key=lambda x: x[1], reverse=True)[:top_n]
        missing = sorted(missing, key=lambda x: x[1], reverse=True)[:top_n]
        
        return matched, missing
    except:
        return [], []


# --- Main App ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 📄 Your Resume")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

with col2:
    st.markdown("#### 💼 Job Description")
    jd_text = st.text_area(
        "Paste JD", 
        height=180, 
        placeholder="Paste the full job description here...\n\nExample: We are looking for a Software Engineer with 3+ years of experience in Python, AWS, and distributed systems...",
        label_visibility="collapsed"
    )

st.markdown("")

if st.button("⚡ Analyze Match"):
    if not uploaded_file:
        st.error("⚠️ Please upload your resume PDF.")
    elif not jd_text.strip():
        st.error("⚠️ Please paste a job description.")
    else:
        with st.spinner("🤖 Loading AI Model & Analyzing... (first run may take ~30s)"):
            model = load_model()
            # 1. Extract & Clean
            raw_resume = extract_text_from_pdf(uploaded_file)
            cleaned_resume = text_cleaning(raw_resume)
            cleaned_jd = text_cleaning(jd_text)
            
            # Validation for fake/empty JDs
            if len(cleaned_jd.split()) < 10:
                st.error("⚠️ This Job Description looks too short or invalid. Please paste a proper, detailed Job Description.")
                st.stop()
                
            if len(cleaned_resume) < 50:
                st.error("Could not extract enough text from your PDF. Please ensure it's a text-based PDF (not a scanned image).")
                st.stop()
            
            # 2. Compute Embeddings with Sentence Transformer
            resume_emb = model.encode(cleaned_resume)
            jd_emb = model.encode(cleaned_jd)
            
            # 3. Cosine Similarity
            sim_score = cosine_similarity([resume_emb], [jd_emb])[0][0]
            percent_score = round(float(sim_score * 100), 2)
            
            # 4. Keyword Analysis
            matched_kw, missing_kw = get_tfidf_keywords(cleaned_jd, cleaned_resume)
            
            # --- RESULTS ---
            st.divider()
            
            res_col1, res_col2 = st.columns([1, 2], gap="large")
            
            with res_col1:
                score_class = "good" if percent_score >= 60 else ("ok" if percent_score >= 40 else "bad")
                
                if percent_score >= 60:
                    verdict = "Strong Match ✨"
                elif percent_score >= 40:
                    verdict = "Moderate Match 🔶"
                else:
                    verdict = "Weak Match ⚠️"
                
                st.markdown(f"""
                <div class="score-container">
                    <div class="score-label">Similarity Score</div>
                    <div class="score-number {score_class}">{percent_score}%</div>
                    <div style="font-size: 18px; font-weight: 600; color: #ccc;">{verdict}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown("### 💡 Score Explanation")
                if percent_score >= 60:
                    st.markdown("""
                    Your resume demonstrates **strong semantic alignment** with this job description. 
                    The AI model found that the core themes, skills, and domain language in your resume 
                    closely match what the employer is looking for. You are likely a strong candidate for this role.
                    """)
                elif percent_score >= 40:
                    st.markdown("""
                    Your resume shows **partial alignment** with this job description. While there are 
                    overlapping areas, the AI model detected that several key requirements or skills 
                    mentioned in the JD are not well represented in your resume. See the recommendations below.
                    """)
                else:
                    st.markdown("""
                    Your resume has **low alignment** with this job description. The core skills, domain 
                    expertise, and experience the employer is seeking appear to be significantly different 
                    from what's presented in your resume. This role may not be the best fit, or your resume 
                    needs substantial tailoring.
                    """)
            
            st.divider()
            
            kw_col1, kw_col2 = st.columns(2, gap="large")
            
            with kw_col1:
                st.markdown("### ✅ Matching Keywords")
                st.caption("These important JD terms were found in your resume.")
                if matched_kw:
                    pills_html = ""
                    for word, score in matched_kw:
                        pills_html += f'<span class="pill pill-green">{word.title()}</span>'
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.write("No significant keyword matches found.")
            
            with kw_col2:
                st.markdown("### 🚀 Missing Keywords")
                st.caption("Consider adding these to your resume if you have this experience.")
                if missing_kw:
                    pills_html = ""
                    for word, score in missing_kw:
                        pills_html += f'<span class="pill pill-red">{word.title()}</span>'
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.write("No significant missing keywords detected.")
            
            st.divider()
            
            st.markdown("### 📝 Actionable Recommendations")
            
            if percent_score >= 60:
                st.markdown("""
                1. **Fine-tune your summary** — Add a targeted objective or summary statement that mirrors the JD's language.
                2. **Quantify achievements** — Back up your matching skills with metrics (e.g., "Improved X by 30%").
                3. **Tailor the order** — Move the most relevant experience and skills to the top of your resume.
                """)
            elif percent_score >= 40:
                if missing_kw:
                    top_missing = ", ".join([f"**{w.title()}**" for w, _ in missing_kw[:5]])
                    st.markdown(f"""
                    1. **Add missing skills** — The most impactful keywords to add are: {top_missing}. If you have experience with these, make sure they appear prominently.
                    2. **Use the JD's language** — Rephrase your bullet points to use the same terminology as the job description.
                    3. **Add a skills section** — Create a dedicated skills section listing the relevant technical and soft skills.
                    4. **Tailor your experience** — Rewrite your work experience to emphasize projects relevant to this role.
                    """)
                else:
                    st.markdown("""
                    1. **Rephrase your content** — Your skills may be there but described differently. Use the JD's exact terminology.
                    2. **Add more detail** — Expand on relevant projects and experiences.
                    """)
            else:
                st.markdown("""
                1. **Evaluate fit** — This role may require a significantly different skill set. Consider whether it's the right target.
                2. **Bridge the gap** — If you're transitioning careers, highlight transferable skills and relevant projects.
                3. **Upskill** — Consider courses or certifications in the key areas mentioned in the JD.
                4. **Network** — For roles with low resume match, networking and referrals become even more important.
                """)
