"""
Configuration settings for the AI Resume Screening System.
"""

# ── Sentence Transformer Model ──────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"

# ── Groq AI Settings ────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.4

# ── Similarity Thresholds ───────────────────────────────────────────────────
THRESHOLD_STRONG = 75    # >= 75% -> Strong Match
THRESHOLD_MODERATE = 58  # >= 58% -> Moderate Match
# < 58% -> Weak Match

# ── Curated Skill Keywords ──────────────────────────────────────────────────
SKILL_KEYWORDS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "go",
    "rust", "scala", "kotlin", "swift", "ruby", "php", "matlab",
    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "data engineering",
    "artificial intelligence", "generative ai", "llm", "transformers",
    "neural networks", "reinforcement learning", "feature engineering",
    # Frameworks & Libraries
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "opencv", "hugging face", "langchain", "streamlit", "fastapi",
    "flask", "django", "react", "angular", "vue", "node.js", "spring boot",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ci/cd", "jenkins", "github actions", "mlops",
    "airflow", "kafka", "spark", "hadoop",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "snowflake", "bigquery",
    # Tools & Practices
    "git", "linux", "api", "rest", "graphql", "microservices",
    "agile", "scrum", "jira", "power bi", "tableau", "excel",
    # Soft Skills (common in JDs)
    "communication", "leadership", "problem solving", "teamwork",
    "project management", "analytical thinking",
]

# ── NLP Preprocessing Settings ──────────────────────────────────────────────
NLTK_RESOURCES = ["punkt_tab", "stopwords", "wordnet"]

# ── ATS Scoring Weights ─────────────────────────────────────────────────────
WEIGHT_SEMANTIC = 0.60   # Semantic similarity weight
WEIGHT_SKILL = 0.30      # Skill overlap weight
WEIGHT_KEYWORD = 0.10    # Raw keyword overlap weight
