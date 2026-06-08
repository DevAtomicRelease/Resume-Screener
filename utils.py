"""
Utility functions for AI-Powered Resume Screening System.

Modules:
    - PDF text extraction (pdfplumber)
    - NLP preprocessing (NLTK)
    - Semantic embeddings (Sentence Transformers)
    - Similarity calculation (scikit-learn)
    - Skill extraction & comparison
    - AI summary generation (Groq)
"""

import re
import os
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import pdfplumber
# pyrefly: ignore [missing-import]
import nltk
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from nltk.corpus import stopwords
# pyrefly: ignore [missing-import]
from nltk.tokenize import word_tokenize
# pyrefly: ignore [missing-import]
from nltk.stem import WordNetLemmatizer
# pyrefly: ignore [missing-import]
from sklearn.metrics.pairwise import cosine_similarity
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

from config import (
    MODEL_NAME, SKILL_KEYWORDS, GROQ_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMPERATURE, NLTK_RESOURCES,
    WEIGHT_SEMANTIC, WEIGHT_SKILL, WEIGHT_KEYWORD,
    THRESHOLD_STRONG, THRESHOLD_MODERATE,
)


# ═══════════════════════════════════════════════════════════════════════════
#  NLTK Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_nltk_data():
    """Download required NLTK datasets if not already present."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_data()


# ═══════════════════════════════════════════════════════════════════════════
#  PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Concatenated text from all pages.
    """
    text_parts = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        st.warning(f"Could not extract text from {getattr(uploaded_file, 'name', 'file')}: {e}")
        return ""
    return "\n".join(text_parts)


# ═══════════════════════════════════════════════════════════════════════════
#  NLP PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for NLP analysis.

    Pipeline: lowercase → remove special chars → tokenize →
              remove stopwords → lemmatize → rejoin.
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r"[^a-z0-9\s+#.]", " ", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords & lemmatize
    processed = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    ]
    return " ".join(processed)


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load the sentence-transformer model (cached as singleton)."""
    return SentenceTransformer(MODEL_NAME)


def get_embeddings(texts: list, model: SentenceTransformer) -> np.ndarray:
    """
    Encode a list of texts into dense vector embeddings.

    Args:
        texts: List of strings to encode.
        model: Loaded SentenceTransformer model.

    Returns:
        NumPy array of shape (len(texts), 384).
    """
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIMILARITY & RANKING
# ═══════════════════════════════════════════════════════════════════════════

def calculate_similarity(resume_embeddings: np.ndarray, jd_embedding: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between each resume and the job description.

    Returns:
        1-D array of similarity scores (0-1 scale).
    """
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)
    similarities = cosine_similarity(resume_embeddings, jd_embedding).flatten()
    return similarities


def calculate_keyword_overlap(resume_text: str, jd_text: str) -> float:
    """Calculate raw keyword overlap ratio between resume and JD."""
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    if not jd_words:
        return 0.0
    overlap = resume_words & jd_words
    return len(overlap) / len(jd_words)


def compute_composite_score(
    semantic_score: float,
    skill_overlap_ratio: float,
    keyword_overlap: float,
) -> float:
    """
    Compute a weighted composite match score.

    Weights defined in config.py:
        Semantic: 60%, Skill: 30%, Keyword: 10%
    """
    composite = (
        WEIGHT_SEMANTIC * semantic_score
        + WEIGHT_SKILL * skill_overlap_ratio
        + WEIGHT_KEYWORD * keyword_overlap
    )
    return min(composite * 100, 100.0)  # Convert to percentage, cap at 100


def get_match_label(score: float) -> str:
    """Return a human-readable match label based on score."""
    if score >= THRESHOLD_STRONG:
        return "Strong Match"
    elif score >= THRESHOLD_MODERATE:
        return "Moderate Match"
    else:
        return "Weak Match"


def rank_candidates(candidate_data: list[dict]) -> pd.DataFrame:
    """
    Build a ranked DataFrame from candidate analysis results.

    Args:
        candidate_data: List of dicts with keys:
            name, score, matched_skills, missing_skills, extra_skills, label

    Returns:
        Sorted DataFrame with Rank column.
    """
    df = pd.DataFrame(candidate_data)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SKILL EXTRACTION & COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

# Synonym map: canonical skill name -> list of alternate forms found in resumes.
# Used when LLM extracts a term from the JD that may appear differently in resumes.
SKILL_SYNONYMS: dict[str, list[str]] = {
    "rag architectures": ["rag", "retrieval augmented generation", "retrieval-augmented", "rag-based", "rag pipeline"],
    "embeddings": ["embedding", "vector embedding", "text embedding", "sentence embedding"],
    "vector databases": ["vector db", "vector store", "pinecone", "weaviate", "chroma", "qdrant", "faiss"],
    "data preprocessing": ["preprocessing", "pre-processing", "data cleaning", "data wrangling", "data preparation"],
    "feature engineering": ["feature extraction", "feature selection", "feature creation"],
    "model integration": ["model deployment", "api integration", "model serving", "model integration"],
    "analytical thinking": ["analytical", "analytical skills", "data-driven", "problem analysis"],
    "cloud platforms": ["cloud", "aws", "azure", "gcp", "google cloud", "cloud computing", "cloud services"],
    "data structures": ["data structure", "algorithms", "dsa"],
    "llms": ["llm", "large language model", "language model", "gpt", "llama", "gemini", "claude"],
    "prompt engineering": ["prompt design", "prompting", "chain of thought", "few-shot", "prompt"],
    "fine-tuning": ["fine tuning", "finetuning", "lora", "qlora", "peft", "instruction tuning"],
    "natural language processing": ["nlp", "text processing", "text analytics", "language processing"],
    "machine learning": ["ml", "statistical learning", "predictive modeling", "supervised learning"],
    "deep learning": ["dl", "neural network", "neural networks", "cnn", "rnn", "lstm", "transformer"],
    "generative ai": ["genai", "gen ai", "generative model", "text generation", "image generation"],
    "problem-solving": ["problem solving", "problem-solving skills", "analytical problem"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "hugging face": ["huggingface", "transformers library", "hf"],
    "langchain": ["lang chain", "langchain framework"],
}


def _skill_found_in_text(skill: str, text_lower: str) -> bool:
    """
    Check whether a skill term (or any of its known synonyms) appears in text.

    Checks exact phrase for multi-word terms and word-boundary regex for
    single-word terms. Falls back to synonym lookup when the primary term
    is not found.
    """
    def _check(term: str) -> bool:
        if " " in term:
            return term in text_lower
        return bool(re.search(rf"\b{re.escape(term)}\b", text_lower))

    if _check(skill):
        return True
    # Try known synonyms / alternate surface forms
    for synonym in SKILL_SYNONYMS.get(skill, []):
        if _check(synonym):
            return True
    return False


def extract_skills(text: str, skill_list: list[str] = None) -> list[str]:
    """
    Extract skills from text by matching against a skill list.

    Uses case-insensitive whole-word / phrase matching with synonym fallback
    so that LLM-extracted JD terms ('embeddings', 'rag architectures', etc.)
    are found even when the resume uses alternate surface forms.
    """
    if skill_list is None:
        skill_list = SKILL_KEYWORDS
    text_lower = text.lower()
    found = [
        skill for skill in skill_list
        if _skill_found_in_text(skill, text_lower)
    ]
    return sorted(set(found))


def compare_skills(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compare resume skills against JD skills.

    Returns:
        dict with keys: matched, missing, extra, overlap_ratio
    """
    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    overlap_ratio = len(matched) / len(jd_set) if jd_set else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "overlap_ratio": overlap_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CANDIDATE NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_candidate_name(text: str) -> str:
    """
    Extract candidate name from resume text.

    Heuristic: first non-empty line that looks like a name
    (2-4 title-case words, no digits).
    Falls back to first non-empty line.
    """
    if not text:
        return "Unknown Candidate"

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines[:5]:  # Check first 5 non-empty lines
        # Skip lines that look like headers/titles
        if any(kw in line.lower() for kw in ["resume", "curriculum", "cv", "objective", "summary"]):
            continue
        # Name pattern: 2-4 words, mostly letters
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.'-]+$", w) for w in words):
            return line.title()

    # Fallback: first non-empty line
    return lines[0][:50] if lines else "Unknown Candidate"



def extract_skills_from_jd_via_llm(jd_text: str, api_key: str) -> list[str]:
    """
    Extract key technical and soft skills/requirements from a job description using Groq LLaMA 3.3.
    """
    try:
        # pyrefly: ignore [missing-import]
        from groq import Groq
        import json

        client = Groq(api_key=api_key)
        
        prompt = f"""Analyze the following Job Description and extract all required technical skills, tools, frameworks, programming languages, databases, cloud platforms, and soft skills or methodologies.
Return the result as a raw JSON object with a single key "skills" mapping to a list of strings representing the extracted requirements. Do not include any formatting or explanations outside the JSON object.

JOB DESCRIPTION:
{jd_text}
"""
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise technical skill extractor. You output ONLY valid JSON objects matching the schema: {\\\"skills\\\": [\\\"skill1\\\", \\\"skill2\\\", ...]}"},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        skills = data.get("skills", [])
        
        # Clean and format skills
        cleaned_skills = []
        for s in skills:
            s_clean = s.strip().lower()
            if s_clean:
                cleaned_skills.append(s_clean)
        
        return sorted(list(set(cleaned_skills)))
    except Exception as e:
        st.warning(f"Failed to extract skills via LLM: {e}. Falling back to static keyword extraction.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  AI SUMMARY GENERATION (GROQ)
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidate_summary(name: str, score: float, matched_skills: list[str], missing_skills: list[str], jd_text: str, resume_text: str, api_key: str) -> str:
    """
    Generate an AI-powered candidate assessment using Groq.

    The LLM produces only the Strengths and Skill Gaps sections.
    The Recommendation is computed deterministically from the composite
    score and appended in code — it cannot be overridden by the model.
    """
    # ── Deterministic recommendation (never delegated to the LLM) ──────────
    if score >= THRESHOLD_STRONG:
        recommendation = (
            f"Strong Hire — composite score of {score:.1f}% meets the strong match "
            f"threshold (>= {THRESHOLD_STRONG}%). Recommend for technical interview."
        )
    elif score >= THRESHOLD_MODERATE:
        recommendation = (
            f"Consider for Interview — composite score of {score:.1f}% falls within "
            f"the moderate range ({THRESHOLD_MODERATE}%–{THRESHOLD_STRONG - 1}%). "
            f"Recommend a screening call to assess depth on missing areas."
        )
    else:
        recommendation = (
            f"Not Recommended — composite score of {score:.1f}% is below the minimum "
            f"threshold of {THRESHOLD_MODERATE}%. Significant skill and domain gaps "
            f"exist for this role."
        )

    try:
        # pyrefly: ignore [missing-import]
        from groq import Groq
        client = Groq(api_key=api_key)
        resume_excerpt = resume_text[:2000]
        jd_excerpt = jd_text[:1500]

        prompt = f"""Evaluate this job application and provide a factual assessment of the candidate's fit.

COMPOSITE SCORE: {score:.1f}%
MATCHED SKILLS ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None detected'}
MISSING SKILLS ({len(missing_skills)}): {', '.join(missing_skills) if missing_skills else 'None'}

CANDIDATE: {name}

JOB DESCRIPTION (excerpt):
{jd_excerpt}

RESUME (excerpt):
{resume_excerpt}

Write ONLY the following two sections. Do NOT write a Recommendation section — it will be added separately.

**Strengths:**
(2-3 concise bullet points on relevant strengths this candidate has for this specific role)

**Skill Gaps:**
(2-3 concise bullet points on specific skills or experience this candidate lacks for this role)"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical recruiter writing factual candidate assessments. "
                        "Be specific to the role and resume provided. Do not write a Recommendation "
                        "section — it is handled by the system. Output only Strengths and Skill Gaps."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.3,
        )
        llm_output = response.choices[0].message.content.strip()

        # Strip any Recommendation section the model may have added anyway
        for marker in ["**Recommendation:**", "**Recommendation**", "Recommendation:"]:
            if marker in llm_output:
                llm_output = llm_output[: llm_output.index(marker)].strip()

        return f"{llm_output}\n\n**Recommendation:** {recommendation}"

    except Exception as e:
        return generate_fallback_summary(name, score, matched_skills, missing_skills)


def generate_fallback_summary(
    name: str,
    score: float,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """
    Generate a rule-based candidate summary (no API required).
    Recommendation is strictly driven by the composite score band.
    """
    # Strengths
    if matched_skills:
        strengths = f"- Demonstrates proficiency in: {', '.join(matched_skills[:8])}"
    else:
        strengths = "- No specific skill matches detected from the job requirements"

    # Gaps
    if missing_skills:
        gaps = f"- Missing key skills required for this role: {', '.join(missing_skills[:6])}"
    else:
        gaps = "- No significant skill gaps detected"

    # Recommendation — strictly follows the score band
    if score >= THRESHOLD_STRONG:
        rec = "Strong Hire — candidate meets or exceeds the required skill and semantic threshold. Recommend for technical interview."
    elif score >= THRESHOLD_MODERATE:
        rec = "Consider for Interview — candidate shows partial alignment. Recommend a screening call to assess depth on missing areas."
    else:
        rec = "Not Recommended — composite score is below the minimum threshold. Significant skill and domain gaps exist for this role."

    return f"**Strengths:**\n{strengths}\n\n**Skill Gaps:**\n{gaps}\n\n**Recommendation:** {rec}"


# ═══════════════════════════════════════════════════════════════════════════
#  FULL ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_candidates(
    resume_files: list,
    jd_file=None,
    jd_text: str = None,
    api_key: str = None,
    progress_callback=None,
) -> dict:
    """
    Run the full analysis pipeline.

    Args:
        resume_files: List of Streamlit UploadedFile objects (PDFs).
        jd_file: Optional Streamlit UploadedFile (JD PDF).
        jd_text: Optional raw JD text (from paste). Used if jd_file is None.
        api_key: Optional Groq API key for AI summaries.
        progress_callback: Optional callable(progress, status_text).

    Returns:
        dict with keys: ranked_df, candidates, jd_skills, jd_text
    """
    def update(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # Step 1: Get JD text (from PDF or pasted text)
    update(0.05, "Processing job description...")
    if jd_file is not None:
        jd_text = extract_text_from_pdf(jd_file)
    if not jd_text or not jd_text.strip():
        raise ValueError("No job description text found. Please upload a PDF or paste text.")
    jd_processed = preprocess_text(jd_text)

    # Extract JD skills dynamically if api_key is available, else fall back
    jd_skills = None
    if api_key:
        update(0.10, "Extracting skills from JD via Groq...")
        jd_skills = extract_skills_from_jd_via_llm(jd_text, api_key)

    if not jd_skills:
        jd_skills = extract_skills(jd_text)

    # Step 2: Load model
    update(0.15, "Loading AI model...")
    model = load_model()

    # Step 3: Generate JD embedding
    update(0.25, "Generating JD embedding...")
    jd_embedding = get_embeddings([jd_processed], model)

    # Step 4: Process each resume
    candidates = []
    n = len(resume_files)

    for i, resume_file in enumerate(resume_files):
        progress = 0.30 + (0.55 * (i / n))
        update(progress, f"Analyzing resume {i + 1}/{n}: {resume_file.name}...")

        # Extract text
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text.strip():
            continue

        # Extract name
        name = extract_candidate_name(resume_text)

        # Preprocess
        resume_processed = preprocess_text(resume_text)

        # Generate embedding
        resume_embedding = get_embeddings([resume_processed], model)

        # Calculate semantic similarity
        semantic_score = float(calculate_similarity(resume_embedding, jd_embedding)[0])

        # Extract & compare skills
        # 1. Match candidate skills specifically against the JD requirements
        resume_matching_skills = extract_skills(resume_text, skill_list=jd_skills)
        # 2. Extract candidate's general skills from the curated keyword list
        resume_general_skills = extract_skills(resume_text, skill_list=None)
        
        # 3. Compare candidate's matching skills with JD requirements
        skill_comparison = compare_skills(resume_matching_skills, jd_skills)
        
        # 4. Compute extra skills (general skills candidate has that are not required by JD)
        jd_skills_lower = set(s.lower() for s in jd_skills)
        extra = [s for s in resume_general_skills if s.lower() not in jd_skills_lower]
        skill_comparison["extra"] = sorted(list(set(extra)))

        # Keyword overlap
        keyword_overlap = calculate_keyword_overlap(resume_text, jd_text)

        # Composite score
        composite_score = compute_composite_score(
            semantic_score,
            skill_comparison["overlap_ratio"],
            keyword_overlap,
        )

        # Match label
        label = get_match_label(composite_score)

        candidates.append({
            "name": name,
            "filename": resume_file.name,
            "score": round(composite_score, 1),
            "semantic_score": round(semantic_score * 100, 1),
            "skill_overlap": round(skill_comparison["overlap_ratio"] * 100, 1),
            "matched_skills": skill_comparison["matched"],
            "missing_skills": skill_comparison["missing"],
            "extra_skills": skill_comparison["extra"],
            "label": label,
            "resume_text": resume_text,
        })

    if not candidates:
        raise ValueError("No valid resumes could be processed.")

    # Step 5: Generate AI summaries
    update(0.88, "Generating AI assessments...")
    for candidate in candidates:
        if api_key:
            candidate["summary"] = generate_candidate_summary(
                candidate["name"],
                candidate["score"],
                candidate["matched_skills"],
                candidate["missing_skills"],
                jd_text,
                candidate["resume_text"],
                api_key,
            )
        else:
            candidate["summary"] = generate_fallback_summary(
                candidate["name"],
                candidate["score"],
                candidate["matched_skills"],
                candidate["missing_skills"],
            )

    # Step 6: Rank
    update(0.95, "Ranking candidates...")
    ranked_df = rank_candidates(candidates)

    update(1.0, "Analysis complete.")

    return {
        "ranked_df": ranked_df,
        "candidates": sorted(candidates, key=lambda c: c["score"], reverse=True),
        "jd_skills": jd_skills,
        "jd_text": jd_text,
    }
