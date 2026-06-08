"""
AI-Powered Resume Screening & Candidate Ranking System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A Streamlit dashboard for analyzing resumes against job descriptions
using NLP, Sentence Transformers, and Generative AI.
"""

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

from utils import analyze_candidates, load_model
from config import THRESHOLD_STRONG, THRESHOLD_MODERATE

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Plotly dark template ────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(240,240,245,0.7)", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
    margin=dict(l=0, r=0, t=30, b=0),
)


# ═══════════════════════════════════════════════════════════════════════════
#  APPLE GLASS EFFECT — CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    st.html("""
    <style>
    /* ── Global ─────────────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }
    .stApp {
        background: linear-gradient(160deg, #08080d 0%, #0E1117 35%, #111120 100%);
    }

    /* ── Scrollbar ──────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(168,85,247,0.25); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(168,85,247,0.45); }

    /* ── Glass Card ─────────────────────────────────────────── */
    .glass {
        background: rgba(255,255,255,0.025);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    .glass:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(168,85,247,0.15);
        box-shadow: 0 8px 32px rgba(168,85,247,0.06);
    }

    /* ── Hero ────────────────────────────────────────────────── */
    .hero {
        text-align: center;
        padding: 32px 20px 24px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, rgba(168,85,247,0.06) 0%, rgba(139,92,246,0.03) 50%, transparent 100%);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #A855F7 0%, #c084fc 35%, #e8e0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 4px;
        letter-spacing: -0.03em;
    }
    .hero p {
        font-size: 0.88rem;
        color: rgba(240,240,245,0.6);
        margin: 0;
    }

    /* ── Section Title ──────────────────────────────────────── */
    .sec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f0f5;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Upload Area ────────────────────────────────────────── */
    .upload-card {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
    }
    .upload-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(240,240,245,0.65);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ── Metric Row ─────────────────────────────────────────── */
    .m-row { display: flex; gap: 10px; }
    .m-card {
        flex: 1;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
    }
    .m-val { font-size: 1.5rem; font-weight: 700; color: #f0f0f5; }
    .m-lbl {
        font-size: 0.68rem; color: rgba(240,240,245,0.55);
        text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; font-weight: 500;
    }

    /* ── Score Circle ───────────────────────────────────────── */
    .sc { width: 84px; height: 84px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 1.3rem;
          font-weight: 700; margin: 0 auto 8px; color: #f0f0f5; }
    .sc-g { background: rgba(34,197,94,0.12); border: 2px solid rgba(34,197,94,0.35); box-shadow: 0 0 16px rgba(34,197,94,0.08); }
    .sc-y { background: rgba(234,179,8,0.12); border: 2px solid rgba(234,179,8,0.35); box-shadow: 0 0 16px rgba(234,179,8,0.08); }
    .sc-r { background: rgba(239,68,68,0.12); border: 2px solid rgba(239,68,68,0.35); box-shadow: 0 0 16px rgba(239,68,68,0.08); }

    /* ── Skill Tags ─────────────────────────────────────────── */
    .tag { display: inline-block; padding: 3px 10px; border-radius: 16px;
           font-size: 0.72rem; font-weight: 500; margin: 2px 2px; }
    .tag-g { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); color: #86efac; }
    .tag-r { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: #fca5a5; }
    .tag-b { background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.2); color: #93c5fd; }

    /* ── Spotlight ───────────────────────────────────────────── */
    .spot {
        background: linear-gradient(135deg, rgba(168,85,247,0.07), rgba(139,92,246,0.03), transparent);
        border: 1px solid rgba(168,85,247,0.12);
        border-radius: 16px; padding: 24px; text-align: center;
    }
    .spot-badge {
        display: inline-block; background: linear-gradient(135deg, #A855F7, #7C3AED);
        color: white; padding: 3px 12px; border-radius: 16px;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    }

    /* ── Summary Box ────────────────────────────────────────── */
    .sum-box {
        background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px; padding: 16px; margin-top: 12px;
        font-size: 0.84rem; line-height: 1.7; color: rgba(240,240,245,0.75);
    }

    /* ── ATS Badge ──────────────────────────────────────────── */
    .ats { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; }
    .ats-g { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25); color: #86efac; }
    .ats-y { background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.25); color: #fde68a; }
    .ats-r { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); color: #fca5a5; }

    /* Analyze button (primary) only — scoped strictly so file uploader buttons are NOT affected */
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #7C3AED, #A855F7) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 10px 20px !important; font-weight: 600 !important; font-size: 0.9rem !important;
        transition: all 0.3s ease !important; width: 100% !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 20px rgba(168,85,247,0.3) !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="stBaseButton-primary"]:active { transform: translateY(0) !important; }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px; margin-bottom: 6px;
    }
    /* Hide the raw icon text (ligature names like 'keyboard_arrow_right')
       that appear when Material Symbols font does not load */
    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
        display: none !important;
    }
    /* Expander summary layout */
    div[data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 10px 14px !important;
        cursor: pointer !important;
        list-style: none !important;
        user-select: none !important;
        position: relative !important;
    }
    div[data-testid="stExpander"] summary::-webkit-details-marker,
    div[data-testid="stExpander"] summary::marker { display: none !important; }
    /* CSS chevron arrow replacing the Material icon */
    div[data-testid="stExpander"] summary::before {
        content: '' !important;
        display: inline-block !important;
        width: 7px !important;
        height: 7px !important;
        border-right: 2px solid rgba(168,85,247,0.7) !important;
        border-bottom: 2px solid rgba(168,85,247,0.7) !important;
        transform: rotate(-45deg) !important;
        transition: transform 0.2s ease !important;
        flex-shrink: 0 !important;
        margin-right: 2px !important;
    }
    div[data-testid="stExpander"][open] summary::before,
    details[open] div[data-testid="stExpander"] summary::before {
        transform: rotate(45deg) !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7C3AED, #A855F7, #c084fc);
        border-radius: 6px;
    }

    section[data-testid="stSidebar"] {
        background: rgba(14,17,23,0.97);
        border-right: 1px solid rgba(255,255,255,0.03);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: rgba(255,255,255,0.02);
        border-radius: 10px; padding: 3px;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 7px; padding: 6px 14px; font-weight: 500; font-size: 0.85rem; }

    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 8px !important; color: #f0f0f5 !important; font-size: 0.85rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(168,85,247,0.35) !important;
        box-shadow: 0 0 0 1px rgba(168,85,247,0.15) !important;
    }

    /* Plotly chart containers */
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }

    /* Hide defaults */
    #MainMenu, footer, header { visibility: hidden; }

    /* File uploader refinement */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    /* Dropzone: column layout so button + instructions stack cleanly */
    [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        background: rgba(255,255,255,0.02) !important;
        border: 1px dashed rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 20px 16px !important;
        min-height: 90px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(168,85,247,0.3) !important;
        background: rgba(168,85,247,0.02) !important;
    }
    /* Reorder: show button first, then instructions */
    [data-testid="stFileUploaderDropzone"] > span { order: 1 !important; }
    [data-testid="stFileUploaderDropzone"] > div[data-testid="stFileUploaderDropzoneInstructions"] { order: 2 !important; }
    /* Browse/Upload button — compact pill style, NOT full width */
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
        width: auto !important;
        min-width: 90px !important;
        background: rgba(168,85,247,0.12) !important;
        border: 1px solid rgba(168,85,247,0.35) !important;
        border-radius: 8px !important;
        color: #c084fc !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 6px 18px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0 !important;
        transform: none !important;
        box-shadow: none !important;
        transition: background 0.2s ease, border-color 0.2s ease !important;
    }
    /* Hide the Material icon inside the Upload browse button — keep text only */
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(168,85,247,0.22) !important;
        border-color: rgba(168,85,247,0.55) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    /* Restore borderless icon buttons (add/delete file cards) — do NOT re-style them */
    [data-testid="stFileUploader"] button[data-testid="stBaseButton-borderlessIcon"] {
        background: transparent !important;
        border: none !important;
        color: rgba(240,240,245,0.6) !important;
        padding: 4px !important;
        width: auto !important;
        min-width: unset !important;
        transform: none !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploader"] button[data-testid="stBaseButton-borderlessIcon"]:hover {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(240,240,245,0.9) !important;
        border-radius: 4px !important;
        transform: none !important;
        box-shadow: none !important;
    }
    /* Ensure icon font inside borderless buttons renders at proper size */
    [data-testid="stFileUploader"] button[data-testid="stBaseButton-borderlessIcon"] [data-testid="stIconMaterial"] {
        display: inline-flex !important;
        font-size: 18px !important;
        line-height: 1 !important;
    }
    /* Instructions text below button */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 0.72rem !important;
        color: rgba(240,240,245,0.55) !important;
    }

    </style>
    """)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def tags(skills, cls):
    if not skills:
        return '<span style="color:rgba(240,240,245,0.55); font-size:0.8rem;">None detected</span>'
    return "".join(f'<span class="tag {cls}">{s}</span>' for s in skills)

def score_circle(score):
    c = "sc-g" if score >= THRESHOLD_STRONG else "sc-y" if score >= THRESHOLD_MODERATE else "sc-r"
    return f'<div class="sc {c}">{score:.0f}%</div>'

def ats_badge(score):
    if score >= THRESHOLD_STRONG:
        return '<span class="ats ats-g">ATS PASS</span>'
    elif score >= THRESHOLD_MODERATE:
        return '<span class="ats ats-y">REVIEW</span>'
    return '<span class="ats ats-r">FAIL</span>'


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def chart_ranking(candidates):
    """Horizontal bar chart of candidate match scores."""
    names = [c["name"] for c in reversed(candidates)]
    scores = [c["score"] for c in reversed(candidates)]
    colors = []
    for s in scores:
        if s >= THRESHOLD_STRONG:
            colors.append("rgba(34,197,94,0.7)")
        elif s >= THRESHOLD_MODERATE:
            colors.append("rgba(234,179,8,0.7)")
        else:
            colors.append("rgba(239,68,68,0.7)")

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="auto",
        textfont=dict(color="white", size=12, family="-apple-system, sans-serif"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(140, len(candidates) * 60 + 40),
        xaxis=dict(title="Match Score (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.03)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", automargin=True),
        bargap=0.3,
    )
    return fig


def chart_score_distribution(candidates):
    """Donut chart showing score distribution."""
    strong = sum(1 for c in candidates if c["score"] >= THRESHOLD_STRONG)
    moderate = sum(1 for c in candidates if THRESHOLD_MODERATE <= c["score"] < THRESHOLD_STRONG)
    weak = sum(1 for c in candidates if c["score"] < THRESHOLD_MODERATE)

    labels = ["Strong Match", "Moderate Match", "Weak Match"]
    values = [strong, moderate, weak]
    colors = ["rgba(34,197,94,0.7)", "rgba(234,179,8,0.7)", "rgba(239,68,68,0.7)"]

    # Filter out zero values
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return None
    labels, values, colors = zip(*filtered)

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="rgba(14,17,23,1)", width=2)),
        textinfo="label+value",
        textfont=dict(size=11, color="rgba(240,240,245,0.8)"),
        hovertemplate="<b>%{label}</b><br>%{value} candidates<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        showlegend=False,
        annotations=[dict(text="Score<br>Split", x=0.5, y=0.5, font_size=13, font_color="rgba(240,240,245,0.4)", showarrow=False)],
    )
    return fig


def chart_skills_frequency(candidates):
    """Bar chart of most common skills across all resumes."""
    from collections import Counter
    all_skills = []
    for c in candidates:
        all_skills.extend(c["matched_skills"])
        all_skills.extend(c["extra_skills"])
    counts = Counter(all_skills).most_common(10)
    if not counts:
        return None

    skills, freqs = zip(*counts)
    fig = go.Figure(go.Bar(
        x=list(freqs), y=list(skills), orientation="h",
        marker=dict(
            color=[f"rgba(168,85,247,{0.3 + 0.5 * (f / max(freqs))})" for f in freqs],
            line=dict(width=0),
        ),
        text=list(freqs),
        textposition="auto",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(counts) * 32 + 40),
        bargap=0.25,
    )
    fig.update_xaxes(title="Frequency", gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.03)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)", automargin=True, categoryorder="total ascending")
    return fig


def chart_radar(candidate, jd_skills):
    """Radar chart for a single candidate's multi-dimensional fit."""
    categories = ["Semantic Match", "Skill Coverage", "ATS Score", "Keyword Relevance"]
    values = [
        candidate["semantic_score"],
        candidate["skill_overlap"],
        min(candidate["score"], 100),
        min(candidate.get("score", 0) * 0.85, 100),  # approx keyword proxy
    ]
    values.append(values[0])  # close the loop
    categories.append(categories[0])

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(168,85,247,0.1)",
        line=dict(color="rgba(168,85,247,0.6)", width=2),
        marker=dict(size=5, color="#A855F7"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=250,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.04)",
                           tickfont=dict(size=9, color="rgba(240,240,245,0.3)")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           tickfont=dict(size=10, color="rgba(240,240,245,0.5)")),
        ),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.html("""
        <div style="text-align:center; padding:10px 0 4px;">
            <div style="font-size:1.1rem; font-weight:700;
                        background:linear-gradient(135deg,#A855F7,#f0f0f5);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">Settings</div>
        </div>
        """)
        st.markdown("---")

        st.markdown("##### Scoring Weights")
        st.markdown("""
| Signal | Weight |
|--------|--------|
| Semantic Similarity | 60% |
| Skill Overlap | 30% |
| Keyword Match | 10% |
        """)
        st.markdown("---")

        st.markdown("##### Thresholds")
        st.markdown(f"""
- **Strong**: >= {THRESHOLD_STRONG}%
- **Moderate**: >= {THRESHOLD_MODERATE}%
- **Weak**: < {THRESHOLD_MODERATE}%
        """)
        st.markdown("---")

        with st.expander("About"):
            st.markdown("""
**AI Resume Screener** uses:
- Sentence Transformers
- Cosine Similarity
- Groq AI (optional)
- NLTK for NLP
            """)
        with st.expander("Tech Stack"):
            st.markdown("""
- **Model**: all-MiniLM-L6-v2
- **NLP**: NLTK
- **ML**: scikit-learn
- **Viz**: Plotly
- **PDF**: pdfplumber
            """)


# ═══════════════════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════════════════

def render_upload_section():
    col_jd, col_res = st.columns(2, gap="medium")

    with col_jd:
        st.html("""
        <div class="upload-card">
            <div class="upload-label">Job Description</div>
        </div>
        """)
        jd_tab1, jd_tab2 = st.tabs(["Upload PDF", "Paste Text"])
        with jd_tab1:
            jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], key="jd_upload", label_visibility="collapsed")
            if jd_file:
                st.html(f'<div style="padding:6px 0;font-size:0.8rem;color:rgba(34,197,94,0.8);">Loaded: {jd_file.name}</div>')
        with jd_tab2:
            jd_text_input = st.text_area(
                "Paste JD",
                height=180,
                placeholder="Paste the full job description text here...",
                key="jd_text",
                label_visibility="collapsed",
            )
            if jd_text_input:
                wc = len(jd_text_input.split())
                st.html(f'<div style="padding:4px 0;font-size:0.78rem;color:rgba(240,240,245,0.55);">{wc} words</div>')

    with col_res:
        st.html("""
        <div class="upload-card">
            <div class="upload-label">Candidate Resumes</div>
        </div>
        """)
        resume_files = st.file_uploader(
            "Upload resume PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_upload",
            label_visibility="collapsed",
        )
        if resume_files:
            file_names = ", ".join(f.name for f in resume_files)
            st.html(f"""
            <div style="padding:8px 12px; margin-top:6px;
                        background:rgba(34,197,94,0.05); border:1px solid rgba(34,197,94,0.12);
                        border-radius:8px; font-size:0.78rem; color:rgba(240,240,245,0.55);">
                <strong style="color:#86efac;">{len(resume_files)}</strong> file{'s' if len(resume_files) != 1 else ''} loaded &mdash; {file_names}
            </div>
            """)

    # Analyze button
    st.markdown("")
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        analyze_clicked = st.button("Analyze Candidates", use_container_width=True, type="primary")

    return jd_file, jd_text_input, resume_files, analyze_clicked


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def render_results(results):
    candidates = results["candidates"]
    jd_skills = results["jd_skills"]
    ranked_df = results["ranked_df"]

    st.html('<div style="height:8px;"></div>')

    # ── Rankings Chart ───────────────────────────────────────────────────
    st.html('<div class="sec-title">Candidate Rankings</div>')

    st.plotly_chart(chart_ranking(candidates), use_container_width=True, config={"displayModeBar": False})

    # ── Rankings Table ───────────────────────────────────────────────────
    display_df = ranked_df[["rank", "name", "score", "semantic_score", "skill_overlap", "label"]].copy()
    display_df.columns = ["Rank", "Candidate", "Overall %", "Semantic %", "Skill Match %", "Verdict"]
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Overall %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Semantic %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Skill Match %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    # ── Top Candidate ────────────────────────────────────────────────────
    if candidates:
        top = candidates[0]
        st.html(f"""
        <div class="spot">
            <div class="spot-badge">Top Candidate</div>
            <div style="font-size:1.3rem; font-weight:700; color:#f0f0f5; margin:8px 0;">{top['name']}</div>
            {score_circle(top['score'])}
            <div style="margin-top:8px;">{tags(top['matched_skills'][:10], 'tag-g')}</div>
            <div style="margin-top:8px;">{ats_badge(top['score'])}</div>
        </div>
        """)

    # ── JD Skills ────────────────────────────────────────────────────────
    if jd_skills:
        st.html(f"""
        <div class="glass" style="margin-top:16px;">
            <div style="font-size:0.72rem; font-weight:600; color:rgba(240,240,245,0.6);
                        text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
                JD Required Skills ({len(jd_skills)})
            </div>
            <div>{tags(jd_skills, 'tag-b')}</div>
        </div>
        """)

    # ── Detailed Profiles ────────────────────────────────────────────────
    st.html('<div class="sec-title">Candidate Profiles</div>')

    for i, c in enumerate(candidates):
        rank_label = f"#{i+1}"
        with st.expander(f"{rank_label} — {c['name']}  ·  {c['score']:.1f}%  ·  {c['label']}", expanded=(i == 0)):
            col_score, col_radar, col_skills = st.columns([1, 1.2, 1.8])

            with col_score:
                st.html(f"""
                <div style="text-align:center; padding-top:8px;">
                    {score_circle(c['score'])}
                    <div style="margin:6px 0;">{ats_badge(c['score'])}</div>
                    <div style="margin-top:12px; text-align:left;">
                        <div style="font-size:0.68rem; color:rgba(240,240,245,0.6); text-transform:uppercase;
                                    letter-spacing:0.05em; margin-bottom:4px; font-weight:600;">Breakdown</div>
                        <div style="font-size:0.8rem; color:rgba(240,240,245,0.7); line-height:1.8;">
                            Semantic: <b>{c['semantic_score']}%</b><br>
                            Skills: <b>{c['skill_overlap']}%</b><br>
                            <span style="color:rgba(240,240,245,0.55);">{c['filename']}</span>
                        </div>
                    </div>
                </div>
                """)

            with col_radar:
                fig = chart_radar(c, jd_skills)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with col_skills:
                st.html(f"""
                <div style="margin-bottom:10px;">
                    <div style="font-size:0.68rem; font-weight:600; color:rgba(240,240,245,0.6);
                                text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                        Matched ({len(c['matched_skills'])})
                    </div>
                    {tags(c['matched_skills'], 'tag-g')}
                </div>
                <div style="margin-bottom:10px;">
                    <div style="font-size:0.68rem; font-weight:600; color:rgba(240,240,245,0.6);
                                text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                        Missing ({len(c['missing_skills'])})
                    </div>
                    {tags(c['missing_skills'], 'tag-r')}
                </div>
                <div>
                    <div style="font-size:0.68rem; font-weight:600; color:rgba(240,240,245,0.6);
                                text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                        Extra ({len(c['extra_skills'])})
                    </div>
                    </div>
                    {tags(c['extra_skills'], 'tag-b')}
                </div>
                """)

            # AI Summary
            st.html(f"""
            <div class="sum-box">
                <div style="font-size:0.68rem; font-weight:600; color:#c084fc;
                            text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
                    AI Assessment
                </div>
                {c['summary'].replace(chr(10), '<br>')}
            </div>
            """)

    # ── Analytics ────────────────────────────────────────────────────────
    render_analytics(candidates, jd_skills)


def render_analytics(candidates, jd_skills):
    st.html('<div class="sec-title">Analytics</div>')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Score Distribution**")
        fig = chart_score_distribution(candidates)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("**Top Skills Across Resumes**")
        fig = chart_skills_frequency(candidates)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No skill data to display.")

    # Pipeline summary
    strong = sum(1 for c in candidates if c["score"] >= THRESHOLD_STRONG)
    moderate = sum(1 for c in candidates if THRESHOLD_MODERATE <= c["score"] < THRESHOLD_STRONG)
    weak = sum(1 for c in candidates if c["score"] < THRESHOLD_MODERATE)

    st.html(f"""
    <div class="glass" style="margin-top:12px;">
        <div style="font-size:0.72rem; font-weight:600; color:rgba(240,240,245,0.6);
                    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px;">
            Pipeline Summary
        </div>
        <div class="m-row">
            <div class="m-card">
                <div class="m-val" style="color:#86efac;">{strong}</div>
                <div class="m-lbl">Strong</div>
            </div>
            <div class="m-card">
                <div class="m-val" style="color:#fde68a;">{moderate}</div>
                <div class="m-lbl">Moderate</div>
            </div>
            <div class="m-card">
                <div class="m-val" style="color:#fca5a5;">{weak}</div>
                <div class="m-lbl">Weak</div>
            </div>
            <div class="m-card">
                <div class="m-val" style="color:#c084fc;">{len(candidates)}</div>
                <div class="m-lbl">Total</div>
            </div>
        </div>
    </div>
    """)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    inject_custom_css()
    render_sidebar()

    # Hero
    st.html("""
    <div class="hero">
        <h1>AI Resume Screener</h1>
        <p>Intelligent candidate ranking &middot; NLP &middot; Sentence Transformers &middot; Groq AI</p>
    </div>
    """)

    # Upload
    jd_file, jd_text_input, resume_files, analyze_clicked = render_upload_section()

    # Analyze
    if analyze_clicked:
        has_jd_file = jd_file is not None
        has_jd_text = bool(jd_text_input and jd_text_input.strip())

        if not has_jd_file and not has_jd_text:
            st.error("Please upload a JD PDF or paste JD text.")
            return
        if not resume_files:
            st.error("Please upload at least one resume PDF.")
            return

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            api_key = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(pct, msg):
            progress_bar.progress(min(pct, 1.0))
            status_text.markdown(f"*{msg}*")

        try:
            results = analyze_candidates(
                resume_files=resume_files,
                jd_file=jd_file,
                jd_text=jd_text_input if has_jd_text else None,
                api_key=api_key,
                progress_callback=progress_callback,
            )
            st.session_state["results"] = results
            progress_bar.empty()
            status_text.empty()
        except ValueError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error: {e}")
            return
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Unexpected error: {e}")
            return

    # Results or empty state
    if "results" in st.session_state:
        render_results(st.session_state["results"])
    elif not analyze_clicked:
        st.html("""
        <div class="glass" style="text-align:center; padding:36px 20px; margin-top:8px;">
            <div style="font-size:1rem; font-weight:600; color:rgba(240,240,245,0.6); margin-bottom:4px;">
                Ready to Screen
            </div>
            <div style="font-size:0.82rem; color:rgba(240,240,245,0.6); max-width:400px; margin:0 auto; line-height:1.5;">
                Upload a job description and resumes above,<br>
                then click <strong style="color:#A855F7;">Analyze Candidates</strong>
            </div>
        </div>
        """)


if __name__ == "__main__":
    main()
