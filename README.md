# AI-Powered Resume Screening & Candidate Ranking System

An intelligent Streamlit dashboard that analyzes resumes against job descriptions using NLP, Sentence Transformers, and Generative AI to rank candidates and generate structured hiring recommendations.

<img width="1784" height="834" alt="image" src="https://github.com/user-attachments/assets/1fe1b9e5-8576-4bcc-8ce3-f5488a15bbe2" />

---

## Overview

This project automates the initial stage of resume screening by combining multiple signals — semantic understanding, skill coverage, and keyword relevance — into a single composite score for each candidate. It is built around real NLP and ML concepts rather than simple keyword counting, making it suitable for roles that require candidates with experience in data science, machine learning, or AI engineering.

<img width="1783" height="864" alt="Img01" src="https://github.com/user-attachments/assets/5fceda49-3ab8-4b07-81c0-4100769d214c" />

---

## Features

| Feature | Description |
|---|---|
| PDF Resume Upload | Upload multiple resumes in PDF format for batch processing |
| Job Description Analysis | Upload a JD PDF or paste text; skills are extracted via LLM |
| Semantic Matching | Sentence Transformers (all-MiniLM-L6-v2) measure conceptual alignment |
| Cosine Similarity | Mathematical scoring of resume-JD embedding distance |
| Dynamic Skill Extraction | Groq LLaMA 3.3 pulls required skills directly from the JD at runtime |
| Candidate Ranking | Composite score: 60% semantic + 30% skill overlap + 10% keyword match |
| AI Summaries | Per-candidate assessment: strengths, skill gaps, and hiring recommendation |
| Analytics Dashboard | Score distribution chart, skill frequency, and pipeline summary |
| ATS Simulation | Applicant Tracking System compatibility scoring |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| NLP | NLTK — tokenization, lemmatization, stopword removal |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Similarity | scikit-learn cosine similarity |
| Generative AI | Groq API — LLaMA 3.3 70B |
| PDF Processing | pdfplumber |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly |

<img width="1761" height="859" alt="img02" src="https://github.com/user-attachments/assets/e5f5b3b7-b1ad-4003-aedd-9441be189cc9" />

---

## Scoring Pipeline

```
Resume Text       Job Description
    |                   |
    v                   v
Preprocessing       Preprocessing
    |                   |
    v                   v
Embedding           Embedding
    |                   |
    +----> Cosine Similarity ----> Semantic Score (60%)
                        |
            LLM Skill Extraction
                        |
            Skill Match vs Resume ----> Skill Score (30%)
                        |
            Raw Keyword Overlap  ----> Keyword Score (10%)
                        |
                        v
               Composite Score (%)
```

Score thresholds:

| Label | Range |
|---|---|
| Strong Match | >= 75% |
| Moderate Match | >= 39% |
| Weak Match | < 39% |

---

## Project Structure

```
resume_screener/
├── app.py               # Streamlit UI and layout
├── utils.py             # NLP, ML, and AI pipeline functions
├── config.py            # Model settings, skill list, scoring weights
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .streamlit/
│   └── config.toml      # Theme configuration
├── resumes/             # Sample resumes and job descriptions
├── models/              # Reserved for custom model files
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Groq API key (optional but recommended)

Copy the example env file and add your key. A free key is available at https://console.groq.com/keys.

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_key_here
```

Without a key, the system still runs using rule-based fallback summaries and static skill extraction.

### 4. Run the application

```bash
streamlit run app.py
```

---

## How It Works

### Step 1: Text Extraction

PDF text is extracted using `pdfplumber`. The system reads each page and concatenates the output. Candidate names are detected by scanning the first five non-empty lines for a 2-4 word title-case pattern.

### Step 2: NLP Preprocessing

Text goes through a standard pipeline: lowercase conversion, removal of special characters, word tokenization (NLTK punkt), stopword removal, and WordNet lemmatization.

### Step 3: Semantic Embedding

Both the preprocessed resume and job description are encoded into 384-dimensional vectors using `all-MiniLM-L6-v2`. This model captures contextual meaning rather than surface-level word overlap.

### Step 4: Skill Extraction

When a Groq API key is present, the LLM extracts required skills directly from the JD text at runtime. The system then checks the resume for those skills using exact-phrase and word-boundary matching, with a synonym map to handle alternate surface forms (e.g., "rag" matches "rag architectures", "sklearn" matches "scikit-learn").

If no API key is provided, the system falls back to a curated static keyword list defined in `config.py`.

### Step 5: Composite Scoring

```python
score = (0.60 * semantic_similarity)
      + (0.30 * skill_overlap_ratio)
      + (0.10 * keyword_overlap_ratio)
```

### Step 6: AI Assessment

The LLM generates a structured per-candidate summary covering strengths, skill gaps, and a hiring recommendation. When no API key is available, a rule-based summary is generated from the matched and missing skill lists.

---

## Resume Guidelines

For best extraction results, resumes should:

- Place the candidate name on the first or second line, as a standalone 2-4 word entry
- Include a dedicated Skills or Technical Skills section
- Use ATS-friendly formatting — avoid tables, columns, or image-based text
- List tools, frameworks, and languages explicitly rather than embedding them only in sentences

---

## Known Limitations

- Scanned PDFs or image-based resumes are not supported; text must be machine-readable
- Skill matching relies on surface-form lookup with a synonym map; it does not infer skills from project descriptions
- The semantic model (all-MiniLM-L6-v2) is a lightweight general-purpose model; domain-specific models would improve accuracy on specialized roles
- Candidate name extraction uses a heuristic line scan; non-standard resume layouts may produce incorrect names

---

## License

MIT License. Free to use, modify, and distribute.
