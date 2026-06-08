# AI-Powered Resume Screening and Candidate Ranking System
## Project Report

**Author:** Muhammad Ghulam Jillani  
**Domain:** Machine Learning / Natural Language Processing / Generative AI  
**Stack:** Python, Streamlit, Sentence Transformers, Groq API, NLTK, scikit-learn  
**Date:** June 2026

---

## 1. Executive Summary

This project implements an end-to-end AI-powered resume screening system that automates the initial candidate evaluation process. Given a job description and a set of candidate resumes in PDF format, the system extracts text, applies NLP preprocessing, computes semantic similarity using deep learning embeddings, performs skill gap analysis, and generates a ranked shortlist with AI-written hiring assessments per candidate.

The system is designed to demonstrate practical application of NLP, machine learning, semantic search, and generative AI — making it directly relevant to roles in Data Science, ML Engineering, NLP Engineering, and GenAI development.

---

## 2. Problem Statement

Manual resume screening is time-consuming, inconsistent, and prone to bias. Recruiters often spend 6-8 seconds on an initial resume scan, and high-volume hiring makes it practically impossible to give every candidate a thorough evaluation.

Traditional keyword-matching approaches (used by most ATS systems) fail to capture semantic equivalence — a candidate who writes "built recommendation pipelines using collaborative filtering" may be as qualified as one who writes "machine learning model development," but keyword-only systems will rank them differently based on surface-level term presence.

This project addresses these gaps by combining:
- Semantic understanding (deep learning embeddings)
- Structured skill comparison (LLM-extracted requirements vs. resume content)
- AI-generated holistic assessments (Groq LLaMA 3.3 70B)

---

## 3. System Architecture

```
                        User Uploads
                            |
              +-------------+-------------+
              |                           |
       Job Description               Resume PDFs
       (PDF or text)               (one or more)
              |                           |
              v                           v
     +------------------+     +------------------+
     | PDF Text Extract  |     | PDF Text Extract  |
     | (pdfplumber)      |     | (pdfplumber)      |
     +--------+---------+     +--------+---------+
              |                         |
              v                         v
     +------------------+     +------------------+
     | NLP Preprocessing |     | NLP Preprocessing |
     | - lowercase       |     | - lowercase       |
     | - tokenize        |     | - tokenize        |
     | - stopword remove |     | - stopword remove |
     | - lemmatize       |     | - lemmatize       |
     +--------+---------+     +--------+---------+
              |                         |
              v                         v
     +------------------+     +------------------+
     | Sentence Encoding |     | Sentence Encoding |
     | all-MiniLM-L6-v2  |     | all-MiniLM-L6-v2  |
     | 384-dim vector    |     | 384-dim vector    |
     +--------+---------+     +--------+---------+
              |                         |
              +----------+--------------+
                         |
                         v
              +--------------------+
              | Cosine Similarity  |  -> Semantic Score (60%)
              +--------------------+
                         |
              +--------------------+
              | LLM Skill Extract  |  Groq LLaMA 3.3
              | (JD requirements)  |
              +--------------------+
                         |
              +--------------------+
              | Skill Matching     |  -> Skill Score (30%)
              | (synonym-aware)    |
              +--------------------+
                         |
              +--------------------+
              | Keyword Overlap    |  -> Keyword Score (10%)
              +--------------------+
                         |
                         v
              +--------------------+
              | Composite Score    |
              | Ranking & Labels   |
              +--------------------+
                         |
              +--------------------+
              | Groq AI Summary    |
              | per Candidate      |
              +--------------------+
                         |
                         v
              +--------------------+
              | Streamlit Dashboard|
              | Charts + Profiles  |
              +--------------------+
```

---

## 4. ML Pipeline

### 4.1 Text Extraction

PDF text is extracted using `pdfplumber`, which handles multi-column layouts and preserves reading order better than raw PDF parsers. Each page's text is concatenated sequentially.

Candidate names are extracted via a heuristic scan of the first five non-empty lines — targeting 2-4 title-case words with no digits. This handles standard resume formats reliably.

### 4.2 NLP Preprocessing

The preprocessing pipeline standardizes both resume and JD text before embedding:

| Step | Description |
|---|---|
| Lowercase | Normalizes casing |
| Special character removal | Retains alphanumeric, spaces, `+`, `#`, `.` |
| Tokenization | NLTK punkt tokenizer (word-level) |
| Stopword removal | NLTK English stopword corpus |
| Lemmatization | WordNet lemmatizer reduces words to base form |

Example: `"developed machine learning models"` → `"develop machine learn model"`

### 4.3 Semantic Embedding

Both preprocessed texts are encoded using `sentence-transformers/all-MiniLM-L6-v2`, a lightweight but high-quality model fine-tuned for semantic similarity tasks.

- Output: 384-dimensional dense vectors
- Normalization: L2-normalized embeddings (cosine similarity = dot product)
- Inference: CPU-compatible, ~50ms per document

### 4.4 Cosine Similarity

Cosine similarity between the JD embedding and each resume embedding measures semantic alignment:

```
similarity = (A · B) / (||A|| × ||B||)
```

This captures conceptual overlap — two documents discussing "data preprocessing pipelines" and "feature engineering workflows" will score high even with minimal exact keyword overlap.

### 4.5 Dynamic Skill Extraction (Groq LLM)

When a Groq API key is present, the system sends the raw JD text to `llama-3.3-70b-versatile` with a structured prompt requesting a JSON list of required skills, tools, frameworks, and competencies.

The LLM extracts skills appropriate to the specific role — for an AI/ML internship, this yields terms like `embeddings`, `rag architectures`, `langchain`, `prompt engineering`, while for a backend role it would yield `rest apis`, `postgresql`, `docker`, etc.

If no API key is provided, a curated static keyword list of 70+ terms from `config.py` is used as a fallback.

### 4.6 Synonym-Aware Skill Matching

Skill matching uses a two-stage lookup:

1. **Primary match**: exact phrase (multi-word) or word-boundary regex (single word)
2. **Synonym fallback**: a `SKILL_SYNONYMS` dictionary maps canonical JD terms to their common resume surface forms

Examples of synonym mappings:

| JD Term (LLM extracted) | Matched aliases |
|---|---|
| `rag architectures` | rag, retrieval augmented generation, rag-based |
| `embeddings` | embedding, vector embedding, sentence embedding |
| `data preprocessing` | preprocessing, data cleaning, data wrangling |
| `cloud platforms` | cloud, aws, azure, gcp, google cloud |
| `llms` | llm, large language model, gpt, llama, gemini |
| `scikit-learn` | sklearn, scikit learn |

This significantly reduces false negatives where a candidate has a skill but uses a different surface form.

### 4.7 Composite Scoring

The final match score is a weighted combination of three signals:

```
score = (0.60 × semantic_similarity × 100)
      + (0.30 × skill_overlap_ratio × 100)
      + (0.10 × keyword_overlap_ratio × 100)
```

Weight rationale:
- **60% semantic**: Most important — captures conceptual understanding
- **30% skill**: Structured comparison against JD requirements
- **10% keyword**: Broad signal, diluted intentionally to avoid pure keyword gaming

Score thresholds:

| Label | Range | Interpretation |
|---|---|---|
| Strong Match | >= 75% | Recommend for technical interview |
| Moderate Match | >= 58% | Consider for screening call |
| Weak Match | < 58% | Significant gaps for this role |

### 4.8 AI Assessment Generation

For each candidate, the system sends a structured prompt to Groq containing:
- Match score and extracted skills
- Excerpts from the resume (first 2000 chars) and JD (first 1500 chars)

The model is prompted to generate exactly three sections:
- **Strengths** — 3-4 bullet points on core competencies
- **Skill Gaps** — 1-2 bullets framing missing skills as growth areas
- **Recommendation** — one sentence with a hiring decision

A rule-based fallback generates the same structure without API calls.

---

## 5. Features

| Feature | Implementation |
|---|---|
| Batch PDF upload | Streamlit `st.file_uploader(accept_multiple_files=True)` |
| JD as PDF or text | Tab-based input with two modes |
| Real-time progress | `st.progress()` with step-level callbacks |
| Candidate ranking | Sorted DataFrame with progress bar columns |
| Score breakdown | Semantic %, Skill %, composite % per candidate |
| Radar chart | Multi-dimensional fit visualization (Plotly) |
| Skill tags (color coded) | Matched (green), Missing (red), Extra (blue) |
| ATS badge | Pass / Review / Fail based on composite score |
| AI assessment | Groq LLaMA 3.3 70B or rule-based fallback |
| Analytics tab | Score distribution donut, skill frequency bar chart |
| Pipeline summary | Strong / Moderate / Weak / Total candidate counts |
| Session state | Results persist across sidebar interactions |

---

## 6. Technology Stack

| Layer | Technology | Version / Detail |
|---|---|---|
| Language | Python | 3.11+ |
| UI Framework | Streamlit | 1.58 |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 (384-dim) |
| Similarity | scikit-learn | cosine_similarity |
| NLP | NLTK | punkt, stopwords, wordnet |
| PDF Processing | pdfplumber | page-level text extraction |
| Generative AI | Groq API | llama-3.3-70b-versatile |
| Data | Pandas, NumPy | DataFrame operations |
| Visualization | Plotly | Bar, Donut, Radar charts |
| Config | python-dotenv | .env for API keys |

---

## 7. Project Structure

```
resume_screener/
├── app.py               # Streamlit UI — layout, charts, CSS, session state
├── utils.py             # Full ML/NLP pipeline: extraction, embedding, scoring, AI summary
├── config.py            # Model name, thresholds, skill keyword list, weights
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Excludes venv, model files, uploaded PDFs, secrets
├── README.md            # Setup and usage documentation
├── Project_Report.md    # This document
├── .streamlit/
│   └── config.toml      # Dark theme, layout settings
└── resumes/
    └── .gitkeep         # Directory placeholder (actual files gitignored)
```

---

## 8. Evaluation

### 8.1 Test Cases

Three resume-JD pairs were evaluated:

| Resume | JD | Score | Label | Assessment |
|---|---|---|---|---|
| Jillani Resume (AI/ML background) | AIML Internship JD | 66.2% | Moderate Match | Accurate — strong AI/ML background, minor gaps in cloud/RAG experience |
| Vishal Sharma Resume | AIML Internship JD | 56.4% | Moderate Match | Borderline — general software skills, less AI/ML depth |
| Web Development Resume | AIML Internship JD | 37% | Weak Match | Accurate — web dev stack (PHP, Ruby, JS) against AI/ML JD |

### 8.2 Observations

1. **Semantic score is generally higher than composite** — the model correctly recognizes that software engineering and AI/ML share conceptual space (both deal with code, algorithms, systems), but the skill overlap component pulls weak-match candidates down appropriately.

2. **False negatives reduced after synonym matching** — before the fix, terms like `embeddings`, `rag architectures`, and `cloud platforms` were being marked as missing even when the resume contained equivalent terms. The synonym map corrected this.

3. **LLM assessment tends to be optimistic** — the system prompt is configured to look for potential and transferable skills. This is intentional for internship-level roles where growth trajectory matters, but the composite score provides an objective counterpoint.

4. **Threshold calibration** — raising `THRESHOLD_MODERATE` from 50% to 58% tightens the Moderate/Weak boundary, ensuring candidates with strong semantic alignment but weak skill coverage (55-57%) are correctly classified as Weak rather than Moderate.

---

## 9. Limitations

| Limitation | Description |
|---|---|
| Scanned PDFs | Image-based or scanned PDFs produce no extractable text |
| Name detection | Heuristic line-scan may fail on non-standard resume layouts |
| Model scope | all-MiniLM-L6-v2 is general-purpose; domain-specific models would improve AI/ML role accuracy |
| Synonym coverage | The synonym map covers common variants; rare or highly domain-specific aliases may still be missed |
| LLM assessment bias | The optimistic recruiter prompt may over-recommend candidates for borderline scores |
| Single JD at a time | Each analysis run is scoped to one job description |
| No persistent storage | Results exist only in Streamlit session state; refreshing clears them |

---

## 10. Future Improvements

| Enhancement | Description |
|---|---|
| Domain-specific models | Fine-tune or use `e5-large` / `instructor-xl` for higher precision on technical roles |
| Resume parsing | Structured extraction (name, education, experience years) using spaCy NER |
| Feedback loop | Allow recruiters to mark results as accurate/inaccurate and retrain weights |
| PDF export | Generate downloadable candidate ranking report in PDF format |
| Multi-JD comparison | Screen one resume pool against multiple job descriptions simultaneously |
| Persistent storage | SQLite or PostgreSQL backend for storing results across sessions |
| ATS format detection | Warn when a resume uses tables, graphics, or multi-column layouts that reduce extraction quality |
| Confidence scoring | Report uncertainty bands alongside match scores |

---

## 11. How to Run

### Prerequisites

- Python 3.11+
- A Groq API key (free tier available at https://console.groq.com/keys)

### Setup

```bash
# Clone or navigate to the project directory
cd resume_screener

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env           # Windows
cp .env.example .env             # macOS/Linux
# Edit .env and set GROQ_API_KEY=your_key_here

# Run the application
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Usage

1. Upload a Job Description — as a PDF file or paste the text directly
2. Upload one or more candidate resume PDFs
3. Click **Analyze Candidates**
4. Review the ranked results, skill tags, radar charts, and AI assessments
5. Use the Analytics section for pipeline-level insights

---

## 12. Concepts Demonstrated

This project covers the following ML/AI concepts relevant to Data Scientist and ML Engineer roles:

| Concept | Where Applied |
|---|---|
| NLP preprocessing | Text cleaning, tokenization, stopword removal, lemmatization |
| Word embeddings | Sentence Transformers (transformer-based dense retrieval) |
| Semantic similarity | Cosine similarity on normalized embedding vectors |
| Information retrieval | Skill extraction and document-to-document matching |
| Generative AI | LLM prompt engineering for structured output (JSON skills, free-text assessment) |
| RAG concepts | JD skill extraction mirrors retrieval-augmented grounding |
| Scoring functions | Weighted composite scoring with interpretable components |
| Thresholding | Label classification based on calibrated score bands |
| Data pipelines | End-to-end pipeline from raw PDF to ranked structured output |
| Interactive ML UI | Streamlit dashboard with real-time progress and visualization |

---

*This report documents the system as built. All scores, thresholds, and model choices reflect deliberate design decisions made during development.*
